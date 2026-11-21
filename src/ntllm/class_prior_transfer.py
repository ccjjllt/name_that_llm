"""Compare final class-mass adjustments under label-frequency changes on Kaggle."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax


METHODS = ("unchanged", "public_sinkhorn", "training_prior", "clipped_sinkhorn", "inverse_marginal_040")
LENGTH_BINS = ((0, 70, "<70"), (70, 120, "70-119"), (120, 200, "120-199"), (200, np.inf, "200+"))
SINKHORN_STEPS = 100
TARGET_MIX = 0.35
SINKHORN_MIX = 0.30
PRIOR_PSEUDOCOUNT = 10.0
PRIOR_STRENGTH = 0.50
LOGIT_CAP = 0.15


def load_oof(path, train_path, score):
    train = pd.read_csv(train_path, usecols=["id", "prompt_id", "model", "text"],
                        dtype={"id": str, "prompt_id": str})
    labels = sorted(train["model"].unique())
    columns = [f"{score}__{label}" for label in labels]
    scores = pd.read_parquet(path, columns=["id", "prompt_id", "model", "fold", *columns])
    scores["id"] = scores["id"].astype(str)
    if len(labels) != 18 or len(scores) != len(train) or not train["id"].is_unique or not scores["id"].is_unique:
        raise ValueError("Expected all official passages and eighteen model classes")
    scores = scores.set_index("id").loc[train["id"]]
    for column in ("prompt_id", "model"):
        if not np.array_equal(scores[column].astype(str), train[column].astype(str)):
            raise ValueError(f"OOF and official data disagree on {column}")
    folds = scores["fold"].to_numpy(dtype=int)
    if len(np.unique(folds)) != 5 or pd.Series(folds).groupby(train["prompt_id"]).nunique().max() != 1:
        raise ValueError("Expected the original five folds with each prompt held out together")
    raw = scores[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("The input scores must be finite")
    target = pd.Categorical(train["model"], categories=labels).codes.astype(int)
    words = train["text"].str.split().str.len().to_numpy(dtype=int)
    return log_softmax(raw, axis=1), target, folds, words, labels


def sinkhorn_offsets(logp, training_mass):
    probability = np.exp(logp)
    target = (1 - TARGET_MIX) * probability.mean(axis=0) + TARGET_MIX * training_mass
    offsets = np.zeros(probability.shape[1])
    for _ in range(SINKHORN_STEPS):
        current = softmax(logp + offsets, axis=1).mean(axis=0)
        offsets += np.log(target / current.clip(1e-300))
        offsets -= offsets.mean()
    # The row multiplier of the balanced matrix cancels in each row's argmax.
    return SINKHORN_MIX * offsets


def predict(logp, training_counts, protected):
    laplace = training_counts + 1
    adjustments = sinkhorn_offsets(logp, laplace / laplace.sum())
    prior = training_counts + PRIOR_PSEUDOCOUNT
    prior /= prior.sum()
    unchanged = logp.argmax(axis=1)
    marginal = np.exp(logp).mean(axis=0).clip(1e-6)
    centered = (logp - 0.40 * np.log(marginal)).argmax(axis=1)
    keep = np.isin(unchanged, protected)
    centered[keep] = unchanged[keep]
    return {
        "unchanged": unchanged,
        "public_sinkhorn": (logp + adjustments).argmax(axis=1),
        "training_prior": (logp - PRIOR_STRENGTH * np.log(prior)).argmax(axis=1),
        "clipped_sinkhorn": (logp + np.clip(adjustments, -LOGIT_CAP, LOGIT_CAP)).argmax(axis=1),
        "inverse_marginal_040": centered,
    }


def cohorts(target, folds):
    yield "full", 0, np.arange(len(target))
    classes = len(np.unique(target))
    scenarios = {
        "moderate": np.geomspace(0.55, 1.0, classes),
        "strong": np.geomspace(0.15, 1.0, classes),
        "three_missing": np.r_[np.zeros(3), np.geomspace(0.25, 1.0, classes - 3)],
    }
    # Rotate class roles to cover each label at every frequency without assuming a hidden layout.
    role_order = np.random.default_rng(21).permutation(classes)
    parts = {(fold, label): np.flatnonzero((folds == fold) & (target == label))
             for fold in np.unique(folds) for label in range(classes)}
    for repeat in range(classes):
        rng = np.random.default_rng(758 + repeat)
        shuffled = {key: rng.permutation(rows) for key, rows in parts.items()}
        roles = np.roll(role_order, repeat)
        for scenario, template in scenarios.items():
            rates = template[roles]
            selected = []
            for (_, label), rows in shuffled.items():
                if rates[label] > 0 and len(rows):
                    selected.append(rows[:max(1, int(round(rates[label] * len(rows))))])
            yield scenario, repeat + 1, np.sort(np.concatenate(selected))


def outcomes(target, guess, unchanged, classes):
    counts = np.bincount(target, minlength=classes)
    observed = counts > 0
    guessed = guess == target
    previous = unchanged == target
    changed = guess != unchanged
    correct_by_class = np.bincount(target[guessed], minlength=classes)
    previous_by_class = np.bincount(target[previous], minlength=classes)
    recall = correct_by_class[observed] / counts[observed]
    delta = (correct_by_class[observed] - previous_by_class[observed]) / counts[observed]
    return {
        "rows": len(target),
        "classes_present": int(observed.sum()),
        "balanced_accuracy": float(recall.mean()),
        "delta_ba": float(delta.mean()),
        "changed": int(changed.sum()),
        "corrected": int((guessed & ~previous).sum()),
        "lost": int((previous & ~guessed).sum()),
        "wrong_to_wrong": int((changed & ~guessed & ~previous).sum()),
        "predicted_absent": int((~observed[guess]).sum()),
    }


def evaluate_cohort(logp, target, folds, words, priors, rows, labels):
    y = target[rows]
    fold = folds[rows]
    length = words[rows]
    n_classes = len(labels)
    protected = [labels.index(name) for name in ("glm-5.3", "glm-5.3-flash")]
    results = {method: np.empty(len(rows), dtype=int) for method in METHODS}
    for key, prior in priors.items():
        within = fold == key
        for method, decisions in predict(logp[rows[within]], prior, protected).items():
            results[method][within] = decisions
    unchanged = results["unchanged"]
    counts = np.bincount(y, minlength=n_classes)
    detail = {}
    for method, guesses in results.items():
        detail[method] = {
            **outcomes(y, guesses, unchanged, n_classes),
            "folds": {str(key): outcomes(y[fold == key], guesses[fold == key], unchanged[fold == key], n_classes)
                      for key in priors},
            "length": {
                name: outcomes(y[keep], guesses[keep], unchanged[keep], n_classes)
                for low, high, name in LENGTH_BINS if (keep := (length >= low) & (length < high)).any()
            },
            "class_recall": {
                label: (float(np.mean(guesses[y == j] == j)) if counts[j] else None)
                for j, label in enumerate(labels)
            },
            "predicted_counts": dict(zip(labels, np.bincount(guesses, minlength=n_classes).tolist())),
        }
    return {"retained_counts": dict(zip(labels, counts.tolist())), "methods": detail}


def directions(deltas):
    values = np.asarray(deltas)
    return {"positive": int((values > 1e-12).sum()), "negative": int((values < -1e-12).sum()),
            "flat": int((np.abs(values) <= 1e-12).sum())}


def aggregate(entries):
    grouped = {}
    for scenario in dict.fromkeys(entry["scenario"] for entry in entries):
        selected = [row for row in entries if row["scenario"] == scenario]
        grouped[scenario] = {}
        for method in METHODS:
            metrics = [entry["methods"][method] for entry in selected]
            deltas = [m["delta_ba"] for m in metrics]
            summary = {
                "cohorts": len(metrics),
                "mean_rows": float(np.mean([m["rows"] for m in metrics])),
                "mean_ba": float(np.mean([m["balanced_accuracy"] for m in metrics])),
                "mean_delta_ba": float(np.mean(deltas)),
                "min_delta_ba": min(deltas),
                "max_delta_ba": max(deltas),
                "cohort_direction": directions(deltas),
                "fold_direction": directions([v["delta_ba"] for m in metrics for v in m["folds"].values()]),
                "fold_mean_delta": {
                    fold: float(np.mean([m["folds"][fold]["delta_ba"] for m in metrics]))
                    for fold in metrics[0]["folds"]
                },
                "length": {},
            }
            for field in ("changed", "corrected", "lost", "wrong_to_wrong", "predicted_absent"):
                summary[f"mean_{field}"] = float(np.mean([m[field] for m in metrics]))
            for _, _, name in LENGTH_BINS:
                band = [m["length"][name] for m in metrics if name in m["length"]]
                if band:
                    summary["length"][name] = {
                        "mean_rows": float(np.mean([m["rows"] for m in band])),
                        "mean_ba": float(np.mean([m["balanced_accuracy"] for m in band])),
                        "mean_delta_ba": float(np.mean([m["delta_ba"] for m in band])),
                    }
            grouped[scenario][method] = summary
    return grouped


def study(oof, train, score):
    logp, target, folds, words, labels = load_oof(oof, train, score)
    priors = {fold: np.bincount(target[folds != fold], minlength=len(labels)) for fold in np.unique(folds)}
    entries = []
    for scenario, repeat, rows in cohorts(target, folds):
        entry = evaluate_cohort(logp, target, folds, words, priors, rows, labels)
        entries.append({"scenario": scenario, "repeat": repeat, **entry})
    return {"rows": len(target), "labels": labels, "summary": aggregate(entries), "cohorts": entries}


def write_results(oof, train, scores, output):
    inputs = {score: study(oof, train, score) for score in scores}
    record = {
        "input_parquet": str(oof),
        "scope": [
            "Downstream decisions on saved Native plain OOF scores and Native plain plus GLM OOF scores; original upstream scaling and routing are fixed.",
            "Adjustments are fit on the unlabeled scores of each held-out fold or use counts of the other folds' labels only.",
            "True labels choose sampled rows and evaluate decisions; they do not enter query adjustments.",
            "Balanced accuracy averages over classes present in the evaluated rows. Rotations overlap and are not independent.",
        ],
        "configuration": {"target_marginal_mix": TARGET_MIX, "sinkhorn_log_mix": SINKHORN_MIX,
                          "sinkhorn_iterations": SINKHORN_STEPS, "target_training_pseudocount": 1,
                          "prior_pseudocount": PRIOR_PSEUDOCOUNT,
                          "prior_strength": PRIOR_STRENGTH, "clipped_logit_cap": LOGIT_CAP,
                          "inverse_marginal_log_strength": 0.40, "inverse_marginal_protects_glm_winners": True},
        "inputs": inputs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "class_prior_transfer.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    rows = []
    for score, result in inputs.items():
        for scenario, methods in result["summary"].items():
            for method, values in methods.items():
                scalars = {key: value for key, value in values.items() if not isinstance(value, dict)}
                rows.append({"score": score, "scenario": scenario, "method": method, **scalars,
                             **{f"fold_{key}": value for key, value in values["fold_direction"].items()}})
    table = pd.DataFrame(rows)
    table.to_csv(output / "class_prior_transfer.csv", index=False)
    print(table[["score", "scenario", "method", "mean_ba", "mean_delta_ba", "mean_corrected", "mean_lost",
                 "fold_positive", "fold_negative"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--train", type=Path, default=Path("/kaggle/input/competitions/name-that-llm/train.csv"))
    parser.add_argument("--score", action="append", help="OOF score prefix, repeat for multiple inputs")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if args.oof is None:
        args.oof = next(Path("/kaggle/input").rglob("glm_dynamics_oof.parquet"))
    scores = args.score or ["reference", "family_mass_glm_posterior"]
    write_results(args.oof, args.train, scores, args.output)


if __name__ == "__main__":
    main()
