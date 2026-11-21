"""Compare two GLM calibration residuals on fixed Low/user-prefix candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, logsumexp

from glm_user_prefix import feature_block, select_rows
from likelihood_refine import transfer_glm_posterior
from likelihood_train import GLM, fit_head, head_logp


# The prior Native + Low/user-prefix experiment used this weight on identical inputs.
BLEND_WEIGHT = 2.706821502316715
RIDGE = 0.25


def design(base, member, labels, glm):
    other = np.array([i for i in range(len(labels)) if i not in glm])
    updated = transfer_glm_posterior(base, base + BLEND_WEIGHT * member, labels)
    native_gap = logsumexp(base[:, glm], axis=1) - base[:, other].max(axis=1)
    blended_gap = logsumexp(updated[:, glm], axis=1) - updated[:, other].max(axis=1)
    family = np.column_stack((native_gap, blended_gap - native_gap))
    siblings = np.column_stack((base[:, glm[0]] - base[:, glm[1]],
                               BLEND_WEIGHT * (member[:, glm[0]] - member[:, glm[1]])))
    family_odds = logsumexp(updated[:, glm], axis=1) - logsumexp(updated[:, other], axis=1)
    sibling_odds = updated[:, glm[0]] - updated[:, glm[1]]
    return updated, (family, family_odds), (siblings, sibling_odds)


def fit_residual(design, target, weight):
    x, offset = design
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -4, 4)))
    weights = weight / weight.sum()
    if len(np.unique(target)) != 2:
        raise ValueError("Both outcomes are needed to fit a calibration residual")

    def objective(coef):
        logits = offset + x @ coef
        loss = weights @ (np.logaddexp(0, logits) - target * logits)
        gradient = x.T @ (weights * (expit(logits) - target))
        return loss + 0.5 * RIDGE * (coef @ coef), gradient + RIDGE * coef

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Calibration failed: {fit.message}")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": fit.x.tolist()}


def residual(head, x):
    x = np.clip((x - head["mean"]) / head["scale"], -4, 4)
    return head["coef"][0] + x @ head["coef"][1:]


def apply_residuals(updated, glm, sibling_odds, family_delta, sibling_delta):
    result = updated.copy()
    family = logsumexp(updated[:, glm], axis=1) + family_delta
    sibling = sibling_odds + sibling_delta
    result[:, glm[0]] = family - np.logaddexp(0, -sibling)
    result[:, glm[1]] = family - np.logaddexp(0, sibling)
    return result


def score_member(features, target, labels, fit, predict):
    head = fit_head(features.iloc[fit], target[fit])
    if head["classes"] != labels:
        raise ValueError("An inner likelihood split is missing a class or its order differs")
    return head_logp(head, features.iloc[predict])


def metrics(prediction, target, rows, labels):
    y, guess = target[rows], prediction[rows]
    recall = {label: float((guess[y == i] == i).mean()) for i, label in enumerate(labels) if np.any(y == i)}
    return {"rows": int(rows.sum()), "balanced_accuracy": float(np.mean(list(recall.values()))), "recall": recall}


def summarize(outputs, labels, target, folds, word_count, chosen, costs):
    result = {"selected": int(chosen.sum()), "token_positions": int(costs[chosen].sum()),
              "selected_short": int((chosen & (word_count < 100)).sum()), "strategies": {}}
    for name, candidate in outputs.items():
        base = outputs["native"].argmax(axis=1)
        prediction = np.where(chosen, candidate.argmax(axis=1), base)
        result["strategies"][name] = {
            "all": metrics(prediction, target, np.ones(len(target), dtype=bool), labels),
            "under_100_words": metrics(prediction, target, word_count < 100, labels),
            "by_fold": {str(fold): metrics(prediction, target, folds == fold, labels) for fold in np.unique(folds)},
        }
    return result


def compare(train_path, reference_path, cache):
    train = pd.read_csv(train_path, dtype={"id": str, "prompt_id": str})
    labels = sorted(train.model.unique())
    glm = [labels.index(label) for label in GLM]
    columns = [f"reference__{label}" for label in labels]
    reference = pd.read_parquet(reference_path, columns=["id", "fold", *columns])
    reference["id"] = reference.id.astype(str)
    if set(train.id) != set(reference.id) or len(labels) != 18:
        raise ValueError("Native and official input labels or IDs differ")
    train = train.merge(reference, on="id", validate="one_to_one", sort=False)
    folds = train.fold.to_numpy()
    if len(np.unique(folds)) != 5 or train.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("The original five prompt folds must remain intact")
    base = log_softmax(train[columns].to_numpy(), axis=1)
    target = train.model.to_numpy()
    target_idx = pd.Categorical(target, categories=labels).codes
    family_target = np.isin(target_idx, glm)
    features, costs = feature_block(cache, "low_user_160", train)
    selected = np.zeros(len(folds), dtype=bool)
    selected[select_rows(base, folds, glm, costs, 0.30, 130_000)] = True
    expanded = np.zeros_like(selected)
    expanded[select_rows(base, folds, glm, costs, 0.30, None)] = True

    outputs = {"native": base}
    outputs.update({name: np.empty_like(base) for name in ("original", "family_only", "siblings_only", "both")})
    fitted = {}
    for outer in np.unique(folds):
        train_rows = folds != outer
        meta_rows = np.flatnonzero(train_rows & selected)
        meta_scores = np.empty((len(meta_rows), len(labels)))
        for inner in np.unique(folds[train_rows]):
            predict = folds[meta_rows] == inner
            fit = np.flatnonzero(train_rows & (folds != inner))
            meta_scores[predict] = score_member(features, target, labels, fit, meta_rows[predict])

        _, family, siblings = design(base[meta_rows], meta_scores, labels, glm)
        within = family_target[meta_rows]
        class_counts = np.bincount(target_idx[train_rows], minlength=len(labels))
        weights = 1.0 / class_counts[target_idx[meta_rows]]
        family_head = fit_residual(family, within.astype(float), weights)
        sibling_head = fit_residual(tuple(column[within] for column in siblings),
                                    (target_idx[meta_rows][within] == glm[0]).astype(float), weights[within])
        fitted[str(outer)] = {"family": family_head, "siblings": sibling_head,
                              "training_candidates": len(meta_rows), "training_glm": int(within.sum())}

        outer_rows = np.flatnonzero(~train_rows)
        outer_member = score_member(features, target, labels, np.flatnonzero(train_rows), outer_rows)
        updated, family, siblings = design(base[outer_rows], outer_member, labels, glm)
        family_delta = residual(family_head, family[0])
        sibling_delta = residual(sibling_head, siblings[0])
        outputs["original"][outer_rows] = updated
        outputs["family_only"][outer_rows] = apply_residuals(updated, glm, siblings[1], family_delta, 0.0)
        outputs["siblings_only"][outer_rows] = apply_residuals(updated, glm, siblings[1], 0.0, sibling_delta)
        outputs["both"][outer_rows] = apply_residuals(updated, glm, siblings[1], family_delta, sibling_delta)

    words = train.text.fillna("").str.split().str.len().to_numpy()
    return {
        "blend_weight": BLEND_WEIGHT, "ridge": RIDGE, "fold_heads": fitted,
        "native_reference": "Frozen public Native prompt OOF. GLM heads are nested within each outer training split.",
        "same_130k_tokens_per_4000": summarize(outputs, labels, target_idx, folds, words, selected, costs),
        "same_top30_frozen_heads": summarize(outputs, labels, target_idx, folds, words, expanded, costs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/glm_prefix_calibration_metrics.json"))
    args = parser.parse_args()
    report = compare(args.train, args.reference, args.cache)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name in ("same_130k_tokens_per_4000", "same_top30_frozen_heads"):
        summary = report[name]
        scores = {key: (round(v["all"]["balanced_accuracy"], 6), round(v["under_100_words"]["balanced_accuracy"], 6))
                  for key, v in summary["strategies"].items()}
        print(name, summary["selected"], summary["token_positions"], scores, flush=True)


if __name__ == "__main__":
    main()
