"""Compare GLM prefix choices after retaining two real answers per prompt."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from glm_prefix_routing import find_input, select_rows
from likelihood_refine import transfer_glm_posterior
from likelihood_train import GLM, normalize_temperature


MODES = ("max_empty_160", "max_empty_320", "low_user_160")


def paired_sample(groups, target, seed, imbalanced):
    rng = np.random.default_rng(seed)
    probability = rng.permutation(np.geomspace(0.35, 1.0, len(np.unique(target))))
    keep = []
    for group in groups:
        if len(group) <= 2:
            keep.extend(group)
        else:
            weights = probability[target[group]] if imbalanced else None
            weights = weights / weights.sum() if imbalanced else None
            keep.extend(rng.choice(group, 2, replace=False, p=weights))
    return np.sort(keep)


def balanced_accuracy(truth, guess):
    counts = np.bincount(truth)
    covered = counts > 0
    correct = np.bincount(truth[guess == truth], minlength=len(counts))
    return float(np.mean(correct[covered] / counts[covered]))


def compare():
    table = pd.read_parquet(find_input("glm_dynamics_oof.parquet"))
    table["id"] = table.id.astype(str)
    labels = sorted(table.model.unique())
    official = pd.read_csv(find_input("train.csv"), dtype={"id": str}, usecols=["id", "text"]).set_index("id")
    with np.load(find_input("glm_user_prefix_oof.npz"), allow_pickle=False) as artifact:
        ids = artifact["ids"].astype(str)
        if len(ids) != len(table) or set(ids) != set(table.id):
            raise ValueError("The two sets of held-out predictions cover different passages")
        table = table.set_index("id", verify_integrity=True).loc[ids]
        folds = table.fold.to_numpy()
        if not np.array_equal(artifact["folds"], folds) or table.groupby("prompt_id").fold.nunique().max() != 1:
            raise ValueError("The original prompt folds must match")
        base = log_softmax(table[[f"reference__{label}" for label in labels]].to_numpy(), axis=1)
        prior = 1.0 / normalize_temperature(base)
        old = base.argmax(axis=1)
        mass = np.exp(base[:, [labels.index(k) for k in GLM]]).sum(axis=1)
        chosen = {}
        costs = {}
        for mode in MODES:
            likelihood = artifact[f"{mode}_logp"]
            costs[mode] = artifact[f"{mode}_token_positions"]
            revised = transfer_glm_posterior(base, base + prior * likelihood, labels)
            chosen[mode] = revised.argmax(axis=1)

    target = pd.Categorical(table.model, categories=labels).codes
    word_count = official.loc[ids].text.fillna("").str.split().str.len().to_numpy()
    by_prompt = list(table.groupby("prompt_id", sort=False).indices.values())
    result = {}
    for scenario, imbalanced in (("uniform", False), ("unequal_class_frequencies", True)):
        seeds = []
        for seed in range(20):
            rows = paired_sample(by_prompt, target, seed, imbalanced)
            short = word_count[rows] < 100
            truth = target[rows]
            default = old[rows]
            values = {"seed": seed, "rows": len(rows), "short_rows": int(short.sum()),
                      "reference_ba": balanced_accuracy(truth, default), "modes": {}}
            for mode in MODES:
                selected = select_rows(mass[rows], folds[rows], costs[mode][rows], 0.30, 130_000)
                outcome = np.where(selected, chosen[mode][rows], default)
                values["modes"][mode] = {
                    "ba": balanced_accuracy(truth, outcome),
                    "short_ba": balanced_accuracy(truth[short], outcome[short]),
                    "scored_rows": int(selected.sum()),
                    "token_positions": int(costs[mode][rows][selected].sum()),
                }
            seeds.append(values)
        overview = {}
        for key in ("ba", "short_ba"):
            new = np.array([seed["modes"]["low_user_160"][key] for seed in seeds])
            comparisons = {}
            for old_mode in MODES[:-1]:
                original = np.array([seed["modes"][old_mode][key] for seed in seeds])
                differences = new - original
                comparisons[old_mode] = {
                    "old_mean": float(original.mean()), "new_mean": float(new.mean()),
                    "mean_delta": float(differences.mean()),
                    "wins": int((differences > 1e-10).sum()),
                    "ties": int((np.abs(differences) <= 1e-10).sum()),
                }
            overview[key] = comparisons
        result[scenario] = {"summary": overview, "samples": seeds}
        print(scenario, json.dumps(overview), flush=True)
    return result


if __name__ == "__main__":
    result = compare()
    Path("/kaggle/working/glm_prefix_sparse_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
