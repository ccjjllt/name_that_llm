"""Measure which Kimi corrections are reachable at different scoring budgets."""

import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from kimi_real_likelihood import KIMI, choose_rows, read_reference


def compare_routes(frame, labels, logits):
    kimi = labels.index(KIMI)
    truth = np.array([labels.index(label) for label in frame.model])
    folds = frame.fold.to_numpy()
    original = logits.argmax(axis=1)
    non_kimi_columns = np.array([i for i in range(len(labels)) if i != kimi])
    rival = non_kimi_columns[logits[:, non_kimi_columns].argmax(axis=1)]
    promote = (truth == kimi) & (original != kimi)
    demote = (original == kimi) & (truth == rival)
    correct = original == truth
    class_count = np.bincount(truth)
    class_weight = 1.0 / (len(labels) * class_count[truth])

    words = frame.text.fillna("").str.split().str.len().to_numpy()
    token_cost = np.minimum(128, np.maximum(16, 1.45 * words))
    top = np.argsort(-logits, axis=1, kind="stable")[:, :3]
    eligible = (top == kimi).any(axis=1)
    lead = logits[:, kimi] - logits[np.arange(len(logits)), rival]
    pk = softmax(logits, axis=1)[:, kimi]
    priorities = {
        "margin_per_token": np.exp(-np.abs(lead)) / np.sqrt(token_cost),
        "binary_uncertainty": pk * (1 - pk),
    }

    def select(priority, count):
        picks = []
        parts = np.unique(folds)
        for part in parts:
            subset = np.flatnonzero((folds == part) & eligible)
            order = np.argsort(-priority[subset], kind="stable")
            picks.append(subset[order[:count // len(parts)]])
        return np.concatenate(picks)

    def summarize(rows):
        fix = (promote | demote)[rows]
        rivals = {
            labels[i]: {
                "promote_kimi": int((promote[rows] & (rival[rows] == i)).sum()),
                "replace_kimi": int((demote[rows] & (rival[rows] == i)).sum()),
            }
            for i in non_kimi_columns
            if ((rival[rows] == i) & fix).any()
        }
        return {
            "rows": len(rows),
            "estimated_token_positions": float(token_cost[rows].sum()),
            "correct_before": int(correct[rows].sum()),
            "promote_kimi": int(promote[rows].sum()),
            "replace_kimi": int(demote[rows].sum()),
            "oracle_gain_ba": float(class_weight[rows][fix].sum()),
            "rivals": rivals,
        }

    results = {}
    for count in (75, 125, 200, 360):
        nearest, _ = choose_rows(logits, folds, kimi, count)
        results[str(count)] = {"nearest_margin": summarize(nearest)}
        for name, priority in priorities.items():
            results[str(count)][name] = summarize(select(priority, count))
    return {"all_top_three": summarize(np.flatnonzero(eligible)), "budgets": results}


def main():
    root = Path("/kaggle/input")
    train = next(root.glob("**/name-that-llm/train.csv"))
    reference = next(root.rglob("glm_dynamics_oof.parquet"))
    frame, labels, logits = read_reference(train, reference)
    results = compare_routes(frame, labels, logits)
    Path("/kaggle/working/kimi_route_budget.json").write_text(json.dumps(results, indent=2) + "\n")
    for count, methods in results["budgets"].items():
        for method, row in methods.items():
            print(count, method, row["estimated_token_positions"], row["promote_kimi"],
                  row["replace_kimi"], round(row["oracle_gain_ba"], 5), flush=True)


if __name__ == "__main__":
    main()
