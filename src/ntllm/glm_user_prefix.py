"""Compare real GLM likelihoods with matched text windows and compute budgets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

from glm_public_features import one as token_features
from likelihood_refine import transfer_glm_posterior
from likelihood_train import DIFFERENCES, GLM, fit_head, head_logp, normalize_temperature


MODES = {
    "max_empty_160": (("old",), 160, 13),
    "max_empty_320": (("old",), 320, 13),
    "low_user_160": (("prefix_part1", "prefix_part2"), 160, 30),
}


def model_features(caches, name, train, window, truncate):
    frames = []
    lengths = train.set_index("id").text.fillna("").str.len()
    for directory in caches:
        feat = pd.read_csv(directory / f"{name}_features.csv", dtype={"id": str}).set_index("id")
        if truncate:
            with np.load(directory / f"{name}_scores.npz", allow_pickle=False) as raw:
                offsets = raw["offsets"]
                scores = raw["scores"]
            if not np.array_equal(np.diff(offsets), feat["n_tok"]):
                raise ValueError(f"Token rows differ from {name} features")
            feat = pd.DataFrame(
                [token_features(scores[first:min(last, first + window)], lengths.at[row_id])
                 for row_id, first, last in zip(feat.index, offsets[:-1], offsets[1:])],
                index=feat.index,
            )
        elif feat["n_tok"].max() > window:
            raise ValueError(f"{name} features exceed the declared window")
        frames.append(feat)
    combined = pd.concat(frames, verify_integrity=True)
    if set(combined.index) != set(train.id) or not np.isfinite(combined.to_numpy()).all():
        raise ValueError(f"{name} features must match all official ids exactly")
    return combined.loc[train.id]


def feature_block(cache, mode, train):
    folders, window, overhead = MODES[mode]
    paths = [cache / name for name in folders]
    flash = model_features(paths, "flash", train, window, mode == "max_empty_160")
    full = model_features(paths, "glm53", train, window, mode == "max_empty_160")
    joined = pd.concat([flash.add_prefix("flash:"), full.add_prefix("glm53:")], axis=1)
    for key in DIFFERENCES:
        joined[f"d:{key}"] = flash[key] - full[key]
    costs = np.maximum(flash.n_tok.to_numpy(), full.n_tok.to_numpy()) + overhead
    return joined, costs


def held_out_member(features, target, folds):
    scores = np.empty((len(target), len(np.unique(target))))
    for fold in np.unique(folds):
        fit = folds != fold
        head = fit_head(features.iloc[fit], target[fit])
        scores[~fit] = head_logp(head, features.iloc[~fit])
    return scores


def select_rows(base, folds, columns, costs, coverage, budget):
    mass = np.exp(base[:, columns]).sum(axis=1)
    picked = []
    for fold in np.unique(folds):
        subset = np.flatnonzero(folds == fold)
        ranked = subset[np.argsort(-mass[subset], kind="stable")][:int(len(subset) * coverage)]
        if budget is not None:
            ceiling = budget * len(subset) / 4000
            ranked = ranked[np.cumsum(costs[ranked]) <= ceiling]
        picked.append(ranked)
    return np.concatenate(picked)


def metrics(target, prediction, baseline, labels, folds):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "fixed": int(((baseline != target) & (prediction == target)).sum()),
        "lost": int(((baseline == target) & (prediction != target)).sum()),
        "recall": {model: float((prediction[target == model] == model).mean()) for model in labels},
        "by_fold": {str(i): float(balanced_accuracy_score(target[folds == i], prediction[folds == i]))
                    for i in np.unique(folds)},
    }


def compare(args):
    train = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str})
    labels = sorted(train.model.unique())
    columns = [f"reference__{label}" for label in labels]
    scores = pd.read_parquet(args.reference, columns=["id", "fold", *columns])
    scores["id"] = scores.id.astype(str)
    train = train.merge(scores, on="id", validate="one_to_one")
    if train.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("A prompt crosses the original evaluation folds")
    base = log_softmax(train[columns].to_numpy(), axis=1)
    folds = train.fold.to_numpy()
    target = train.model.to_numpy()
    names = np.asarray(labels)
    baseline = names[base.argmax(axis=1)]
    glm_columns = [labels.index(label) for label in GLM]
    weight = 1.0 / normalize_temperature(base)
    modes = {}
    artifact = {"ids": train.id.to_numpy(dtype=str), "folds": folds}
    for name in MODES:
        block, costs = feature_block(args.cache, name, train)
        ll = held_out_member(block, target, folds)
        strategy = {}
        for case, coverage, budget in (("same_top15", 0.15, None), ("same_top30", 0.30, None),
                                       ("same_130k_tokens_per_4000", 0.30, 130_000)):
            rows = select_rows(base, folds, glm_columns, costs, coverage, budget)
            result = {"rows": len(rows), "token_positions": int(costs[rows].sum()), "weights": {}}
            for strength in (1.0, 1.5):
                updated = base.copy()
                combined = base[rows] + strength * weight * ll[rows]
                updated[rows] = transfer_glm_posterior(base[rows], combined, labels)
                prediction = names[updated.argmax(axis=1)]
                result["weights"][str(strength)] = metrics(target, prediction, baseline, labels, folds)
            strategy[case] = result
        artifact[f"{name}_logp"] = ll
        artifact[f"{name}_token_positions"] = costs
        modes[name] = strategy
    out = {"reference": metrics(target, baseline, baseline, labels, folds),
           "entropic_blend_weight": weight, "modes": modes, "labels": labels}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "glm_user_prefix_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(args.output / "glm_user_prefix_oof.npz", **artifact)
    for name, cases in modes.items():
        for case, result in cases.items():
            print(name, case, result["rows"], result["token_positions"],
                  {k: round(v["balanced_accuracy"], 5) for k, v in result["weights"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    compare(parser.parse_args())
