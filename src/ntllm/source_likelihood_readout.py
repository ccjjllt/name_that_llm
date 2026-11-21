"""Diagnose existing routed source-likelihood scores without running a model."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def load_predictions(train_path, reference_path, source_path):
    train = pd.read_csv(train_path, dtype={"id": str, "prompt_id": str}, usecols=["id", "prompt_id", "model"])
    labels = sorted(train.model.unique())
    columns = [f"family_mass_glm_posterior__{label}" for label in labels]
    reference = pd.read_parquet(reference_path, columns=["id", "fold", *columns])
    reference["id"] = reference.id.astype(str)
    frame = train.merge(reference, on="id", validate="one_to_one")
    if len(frame) != len(train) or frame.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("The reference must include every official row with prompts confined to their folds")
    scores = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Reference logits contain non-finite scores")
    expected = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    label_index = pd.Index(labels)
    paths = sorted(Path(source_path).glob("native_candidates_fold*.npz"))
    found, rows, candidates, ratio = set(), [], [], []
    ids = pd.Index(frame.id)
    for path in paths:
        fold = int(path.stem.rsplit("fold", 1)[1])
        with np.load(path, allow_pickle=False) as values:
            chosen = ids.get_indexer(values["ids"].astype(str))
            names = values["candidates"].astype(str)
            strength = values["log_ratios"].astype(np.float64)
        candidate = label_index.get_indexer(names.ravel()).reshape(names.shape)
        if (
            (chosen < 0).any() or np.unique(chosen).size != len(chosen)
            or not (frame.fold.to_numpy()[chosen] == fold).all()
            or not np.array_equal(candidate, expected[chosen])
            or strength.shape != candidate.shape or not np.isfinite(strength).all()
        ):
            raise ValueError(f"Source and reference rows or candidate order differ in {path.name}")
        found.add(fold)
        rows.append(chosen)
        candidates.append(candidate)
        ratio.append(strength)
    if found != set(frame.fold.astype(int)):
        raise ValueError(f"Source artifacts cover folds {sorted(found)}, expected {sorted(set(frame.fold))}")
    return frame, labels, scores, expected, np.concatenate(rows), np.concatenate(candidates), np.concatenate(ratio)


def fit_decoy_bias(candidates, ratios, truth, classes, shrinkage=8.0):
    """Estimate author offsets using score differences where both labels are wrong."""
    design, response = [], []
    for left, right in combinations(range(candidates.shape[1]), 2):
        wrong = (candidates[:, left] != truth) & (candidates[:, right] != truth)
        indices = np.flatnonzero(wrong)
        x = np.zeros((len(indices), classes))
        x[np.arange(len(indices)), candidates[indices, left]] = 1
        x[np.arange(len(indices)), candidates[indices, right]] = -1
        design.append(x)
        response.append(ratios[indices, left] - ratios[indices, right])
    x, y = np.concatenate(design), np.concatenate(response)
    return np.linalg.solve(x.T @ x + shrinkage * np.eye(classes), x.T @ y), len(y)


def crossfit_bias(candidates, ratios, truth, folds, prompts, labels):
    corrected = np.empty_like(ratios)
    parameters = {}
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        parameters[str(fold)] = []
        for train, test in GroupKFold(2).split(rows, groups=prompts[rows]):
            fit, valid = rows[train], rows[test]
            bias, n_decoy = fit_decoy_bias(candidates[fit], ratios[fit], truth[fit], len(labels))
            offsets = bias[candidates[valid]]
            corrected[valid] = ratios[valid] - offsets + offsets[:, :1]
            parameters[str(fold)].append({
                "fit_prompts": int(np.unique(prompts[fit]).size),
                "validation_prompts": int(np.unique(prompts[valid]).size),
                "decoy_pairs": n_decoy,
                "offsets": dict(zip(labels, bias.tolist())),
            })
    return corrected, parameters


def rate(values):
    return float(np.mean(values)) if len(values) else None


def auc(target, scores):
    return float(roc_auc_score(target, scores)) if np.unique(target).size == 2 else None


def capture(target, top):
    hit = top == target[:, None]
    return {
        "rows": len(target),
        "reference_correct": int(hit[:, 0].sum()),
        "top3_has_truth": int(hit.any(axis=1).sum()),
        "reference_wrong_truth_in_top3": int((~hit[:, 0] & hit[:, 1:].any(axis=1)).sum()),
        "top3_hit_rate": rate(hit.any(axis=1)),
        "top3_balanced_hit_rate": rate([hit[target == label].any(axis=1).mean() for label in np.unique(target)]),
    }


def conditional_pairs(folds, target, labels, candidates, base, raw, adjusted):
    """Compare the reference winner with each rival whenever one of them is true."""
    n_rivals = candidates.shape[1] - 1
    anchor = np.repeat(candidates[:, 0], n_rivals)
    rival = candidates[:, 1:].ravel()
    truth = np.repeat(target, n_rivals)
    fold = np.repeat(folds, n_rivals)
    eligible = (anchor == truth) | (rival == truth)
    positive = rival == truth
    margin = (base[:, 1:] - base[:, :1]).ravel()
    ll_margin = (raw[:, 1:] - raw[:, :1]).ravel()
    centered = (adjusted[:, 1:] - adjusted[:, :1]).ravel()
    family = np.array([label.split("-", 1)[0] for label in labels])

    def summarize(rows):
        y = positive[rows]
        source = ll_margin[rows]
        adjusted_source = centered[rows]
        return {
            "pairs": int(rows.sum()), "truth_is_rival": int(y.sum()),
            "reference_auc": auc(y, margin[rows]),
            "source_auc": auc(y, source),
            "source_bias_corrected_auc": auc(y, adjusted_source),
            "reference_plus_source_auc": auc(y, margin[rows] + 0.25 * source),
            "reference_plus_corrected_source_auc": auc(y, margin[rows] + 0.25 * adjusted_source),
            "source_prefers_truth_on_reference_errors": rate(source[y] > 0),
            "source_prefers_truth_when_reference_correct": rate(source[~y] < 0),
            "corrected_source_prefers_truth_on_reference_errors": rate(adjusted_source[y] > 0),
            "corrected_source_prefers_truth_when_reference_correct": rate(adjusted_source[~y] < 0),
        }

    def scopes(rows):
        return {
            "all": summarize(rows),
            "gemini": summarize(rows & (family[truth] == "gemini")),
            "gemini_within_family": summarize(rows & (family[anchor] == "gemini") & (family[rival] == "gemini")),
            "gpt": summarize(rows & (family[truth] == "gpt")),
            "gpt_within_family": summarize(rows & (family[anchor] == "gpt") & (family[rival] == "gpt")),
        }

    return {"overall": scopes(eligible), "by_fold": {str(k): scopes(eligible & (fold == k)) for k in np.unique(fold)}}


def rerank(scores, rows, candidates, ratios, scale):
    result = scores.copy()
    indices = (rows[:, None], candidates)
    initial = scores[indices]
    modified = initial + scale * ratios
    result[indices] = modified - logsumexp(modified, axis=1, keepdims=True) + logsumexp(initial, axis=1, keepdims=True)
    return result.argmax(axis=1)


def study(train_path, reference_path, source_path):
    frame, labels, scores, expected, rows, candidates, ratios = load_predictions(train_path, reference_path, source_path)
    truth = pd.Index(labels).get_indexer(frame.model)
    initial = scores.argmax(axis=1)
    folds = frame.fold.to_numpy()
    fixed, bias = crossfit_bias(candidates, ratios, truth[rows], folds[rows], frame.prompt_id.to_numpy()[rows], labels)
    pairs = conditional_pairs(folds[rows], truth[rows], labels, candidates, scores[rows[:, None], candidates], ratios, fixed)

    def decision_metrics(prediction):
        return {
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "fixed_predictions": int(((initial != truth) & (prediction == truth)).sum()),
            "spoiled_predictions": int(((initial == truth) & (prediction != truth)).sum()),
            "by_fold": {str(k): balanced_accuracy_score(truth[folds == k], prediction[folds == k])
                        for k in np.unique(folds)},
            "gemini_recall": rate([rate(prediction[truth == i] == i) for i, name in enumerate(labels) if name.startswith("gemini-")]),
            "gpt_recall": rate([rate(prediction[truth == i] == i) for i, name in enumerate(labels) if name.startswith("gpt-")]),
        }

    result = {
        "candidate_capture": {"all": capture(truth, expected), "scored": capture(truth[rows], candidates)},
        "conditional_pairs": pairs,
        "shrunken_author_offsets": bias,
        "decisions": {"reference": decision_metrics(initial)},
    }
    for name, value in (("raw", ratios), ("bias_corrected", fixed)):
        for scale in (0.25, 0.50, 1.0):
            result["decisions"][f"{name}_{scale}"] = decision_metrics(rerank(scores, rows, candidates, value, scale))
    return result


def unique_input(root, name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one mounted {name}, found {len(matches)}; specify its path explicitly")
    return matches[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="Official train.csv")
    parser.add_argument("--reference", type=Path, help="glm_dynamics_oof.parquet with family_mass_glm_posterior columns")
    parser.add_argument("--source", type=Path, help="Directory with five native_candidates_fold*.npz files")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/source_likelihood_readout.json"))
    args = parser.parse_args()
    root = Path("/kaggle/input")
    train = args.train or next((path for path in (
        root / "competitions/name-that-llm/train.csv", root / "name-that-llm/train.csv",
    ) if path.is_file()), None)
    if train is None:
        raise FileNotFoundError("Mount the official Name That LLM competition or specify --train")
    reference = args.reference or unique_input(root, "glm_dynamics_oof.parquet")
    source = args.source or unique_input(root, "native_candidates_fold0.npz").parent
    result = study(train, reference, source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("candidate_capture", "decisions")}, indent=2))
