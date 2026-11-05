"""Coverage-preserving prompt batches and fold-fitted soft text matching."""

import numpy as np
import pandas as pd


WORD_BINS = ("<60", "60-99", "100-159", ">=160")
MATCH_FEATURES = ("length", "curly_quotes", "em_dash", "colon", "newline")
FEATURE_SIZES = (4, 2, 2, 2, 2)
MIN_WEIGHT = 0.5
MAX_WEIGHT = 2.0


def prompt_batches(frame, seed, batch_size=3):
    """Return positional rows and a centered-loss flag, using each row once.

    Eligible prompt groups supply distinct-label batches first. If only two
    labels remain, a two-row centered batch is retained. Single-label leftovers
    and ineligible groups supply ordinary mixed batches, including the last
    incomplete batch. No row is duplicated to fill a batch.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    rng = np.random.default_rng(seed)
    if batch_size == 1:
        return [([int(row)], False) for row in rng.permutation(len(frame))]
    groups = frame[["prompt_id", "model"]].reset_index(drop=True)
    evidence_column = "group_eligible" if "group_eligible" in frame else "group_is_strong"
    eligible = frame[evidence_column].fillna(False).to_numpy(dtype=bool, copy=True)
    eligible &= groups.prompt_id.fillna("").str.strip().ne("").to_numpy()
    batches = []
    mixed = np.flatnonzero(~eligible).tolist()
    for positions in groups.loc[eligible].groupby("prompt_id", sort=True).groups.values():
        part = groups.loc[positions]
        by_label = {
            label: rng.permutation(rows.to_numpy(dtype=int)).tolist()
            for label, rows in part.groupby("model", sort=True).groups.items()
        }
        while len(by_label) >= 2:
            labels = list(by_label)
            rng.shuffle(labels)
            labels.sort(key=lambda label: len(by_label[label]), reverse=True)
            selected = labels[:batch_size]
            rows = [int(by_label[label].pop()) for label in selected]
            batches.append((rows, True))
            for label in selected:
                if not by_label[label]:
                    del by_label[label]
        mixed.extend(row for rows in by_label.values() for row in rows)
    rng.shuffle(mixed)
    batches.extend((mixed[start:start + batch_size], False)
                   for start in range(0, len(mixed), batch_size))
    return [batches[index] for index in rng.permutation(len(batches))]


def plan_summary(frame, batches):
    """Describe the actual planned exposure using positional row indices."""
    rows = np.asarray([row for batch, _ in batches for row in batch], dtype=np.int64)
    if ((rows < 0) | (rows >= len(frame))).any():
        raise ValueError("A batch contains a row position outside the frame.")
    counts = np.bincount(rows, minlength=len(frame))
    exposed = frame.iloc[rows]
    missing = np.flatnonzero(counts == 0)
    duplicates = np.flatnonzero(counts > 1)
    centered_rows = sum(len(batch) for batch, same in batches if same)
    summary = {
        "input_rows": len(frame), "batches": len(batches),
        "row_exposures": len(rows), "unique_positions": int(np.count_nonzero(counts)),
        "unique_ids": int(exposed.id.nunique()) if "id" in exposed else int(np.count_nonzero(counts)),
        "coverage_fraction": float(np.count_nonzero(counts) / len(frame)) if len(frame) else 1.0,
        "missing_rows": len(missing), "missing_indices": missing.tolist(),
        "duplicate_exposures": int(np.maximum(counts - 1, 0).sum()),
        "duplicate_indices": duplicates.tolist(),
        "centered_batches": sum(bool(same) for _, same in batches),
        "centered_rows": centered_rows,
        "centered_row_fraction": centered_rows / len(rows) if len(rows) else 0.0,
        "batch_sizes": {str(size): int(count) for size, count in
                        pd.Series([len(batch) for batch, _ in batches], dtype=int).value_counts().items()},
    }
    for column in ("model", "source_dataset", "word_bin"):
        if column in exposed:
            summary[f"rows_per_{column}"] = exposed.groupby(column).size().to_dict()
    return summary


def _matching_features(frame):
    text = frame.text.fillna("").astype(str)
    words = frame.n_words.to_numpy() if "n_words" in frame else text.str.split().str.len().to_numpy()
    return np.column_stack([
        np.searchsorted([60, 100, 160], words, side="right"),
        text.str.contains("[“”‘’]", regex=True).to_numpy(dtype=int),
        text.str.contains("—", regex=False).to_numpy(dtype=int),
        text.str.contains(":", regex=False).to_numpy(dtype=int),
        text.str.contains(r"[\r\n]", regex=True).to_numpy(dtype=int),
    ]).astype(np.int64)


def _bounded_mean_one(scores):
    """Scale positive scores to mean one while retaining fixed weight bounds."""
    low = MIN_WEIGHT / scores.max()
    high = MAX_WEIGHT / scores.min()
    for _ in range(60):
        scale = (low + high) / 2
        if np.clip(scale * scores, MIN_WEIGHT, MAX_WEIGHT).mean() > 1:
            high = scale
        else:
            low = scale
    return np.clip(((low + high) / 2) * scores, MIN_WEIGHT, MAX_WEIGHT)


def matching_weights(external, official_fit, strength=0.5):
    """Fit bounded class-conditional weights using only official_fit targets.

    Five Laplace-smoothed marginal density ratios are combined geometrically,
    then shrunk by strength. Each external class has mean weight one, so these
    weights adjust within-class style rather than class priors. They are used
    as loss weights and do not resample or remove training texts.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and lie in [0, 1].")
    source_features = _matching_features(external)
    target_features = _matching_features(official_fit)
    source_labels = external.model.to_numpy()
    target_labels = official_fit.model.to_numpy()
    weights = np.ones(len(external), dtype=np.float64)
    reports = []
    for label in sorted(external.model.unique()):
        source_positions = np.flatnonzero(source_labels == label)
        target_positions = np.flatnonzero(target_labels == label)
        source = source_features[source_positions]
        target = target_features[target_positions]
        log_ratios = np.zeros(len(source), dtype=np.float64)
        if len(target):
            for column, categories in enumerate(FEATURE_SIZES):
                source_count = np.bincount(source[:, column], minlength=categories)
                target_count = np.bincount(target[:, column], minlength=categories)
                source_probability = (source_count + 1) / (len(source) + categories)
                target_probability = (target_count + 1) / (len(target) + categories)
                log_ratios += np.log(target_probability[source[:, column]]
                                     / source_probability[source[:, column]])
            scores = np.exp(strength * log_ratios / len(MATCH_FEATURES))
            weights[source_positions] = _bounded_mean_one(scores)
        class_weights = weights[source_positions]
        marginals = {}
        for column, (name, categories) in enumerate(zip(MATCH_FEATURES, FEATURE_SIZES)):
            raw = np.bincount(source[:, column], minlength=categories) / len(source)
            matched = (np.bincount(source[:, column], weights=class_weights, minlength=categories)
                       / class_weights.sum())
            target_rate = (np.bincount(target[:, column], minlength=categories) / len(target)
                           if len(target) else None)
            marginals[name] = {
                "source": raw.tolist(), "weighted": matched.tolist(),
                "target": target_rate.tolist() if target_rate is not None else None,
                "source_tv": float(np.abs(raw - target_rate).sum() / 2) if len(target) else None,
                "weighted_tv": float(np.abs(matched - target_rate).sum() / 2) if len(target) else None,
            }
        effective_rows = float(class_weights.sum() ** 2 / np.square(class_weights).sum())
        reports.append({
            "model": str(label), "external_rows": len(source), "target_rows": len(target),
            "target_available": bool(len(target)),
            "weight_min": float(class_weights.min()), "weight_max": float(class_weights.max()),
            "weight_mean": float(class_weights.mean()),
            "lower_bound_rows": int(np.isclose(class_weights, MIN_WEIGHT).sum()),
            "upper_bound_rows": int(np.isclose(class_weights, MAX_WEIGHT).sum()),
            "bound_weight_fraction": float((np.isclose(class_weights, MIN_WEIGHT)
                                             | np.isclose(class_weights, MAX_WEIGHT)).mean()),
            "effective_sample_size": effective_rows, "effective_sample_fraction": effective_rows / len(source),
            "length_unweighted_tv": marginals["length"]["source_tv"],
            "length_weighted_tv": marginals["length"]["weighted_tv"],
            "marginals": marginals,
        })
    report = {
        "method": "Class-conditional geometric mean of Laplace-smoothed marginal density ratios",
        "strength": float(strength), "laplace_alpha": 1.0,
        "features": list(MATCH_FEATURES), "length_categories": list(WORD_BINS),
        "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT], "class_mean_weight": 1.0,
        "external_rows": len(external), "target_fit_rows": len(official_fit),
        "target_prompt_count": int(official_fit.prompt_id.nunique()) if "prompt_id" in official_fit else None,
        "target_folds": sorted(int(fold) for fold in official_fit.fold.unique()) if "fold" in official_fit else None,
        "target_only_labels": sorted(set(target_labels) - set(source_labels)),
        "per_class": reports,
        "weighting_scope": "Within-class loss reweighting only. Class balancing is supplied separately by the trainer.",
    }
    return weights.astype(np.float32), report
