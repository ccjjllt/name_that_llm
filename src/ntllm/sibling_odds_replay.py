"""Replay the final decision cell of RanJeet Gupta's Sibling Odds v1 (372780350).

Inputs are the frozen Dennis v1 public-cohort trace. GLM selection, likelihood
evaluation, family residual fitting and Fable protection stay fixed. An external
reader can contribute 18 logits before the sibling gate and typography rules.
"""

import numpy as np
import pandas as pd


def replay_predictions(trace, texts, *, external_logits=None, weight=0.0, sibling_odds=True, typography=True):
    """Return labels in trace ID order. Supply texts and optional logits indexed by ID."""
    ids = pd.Index(trace["ids"])
    labels = np.asarray(trace["labels"])
    if not ids.is_unique or not texts.index.is_unique or not ids.isin(texts.index).all():
        raise ValueError("Texts must contain every trace ID exactly once")
    if not 0 <= weight <= 1:
        raise ValueError("External reader weight must be between 0 and 1")
    refined = np.asarray(trace["sibling_logp"], dtype=np.float64)
    if weight:
        if external_logits is None:
            raise ValueError("A positive weight requires external reader logits")
        if (not external_logits.index.is_unique or not ids.isin(external_logits.index).all()
                or not external_logits.columns.is_unique or set(labels) != set(external_logits.columns)):
            raise ValueError("External reader logits must identify all trace IDs and labels")
        other = external_logits.loc[ids, labels].to_numpy(dtype=np.float64)
        if not np.isfinite(other).all():
            raise ValueError("External reader logits must be finite")
        refined = (1 - weight) * refined + weight * other

    winner = refined.argmax(axis=1)
    predictions = labels[winner].copy()
    if sibling_odds:
        delta = trace["glm_logp"] - trace["stack_logp"]
        adjusted = refined.copy()
        family = np.array([label.split("-")[0] for label in labels])
        for name in ("claude", "gpt", "gemini"):
            group = family == name
            within = delta[:, group]
            adjusted[:, group] += 0.55 * np.clip(within - within.mean(axis=1, keepdims=True), -3, 3)
        candidate = adjusted.argmax(axis=1)
        top = np.sort(refined, axis=1)
        accept = (family[winner] == family[candidate]) & (top[:, -1] - top[:, -2] < 0.8)
        predictions[accept] = labels[candidate[accept]]
    protected = trace["glm_winner"] | trace["fable_guard"]
    predictions[protected] = trace["glm"][protected]
    if not typography:
        return predictions

    # In the published cell, only a violated rule masks the probabilities. Each
    # rule sees the previous one's updated prediction and probability support.
    probs = np.exp(refined - refined.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    passages = texts.loc[ids].fillna("").astype(str)
    spaced_allowed = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
    rules = (
        (" — ", [label not in spaced_allowed for label in labels]),
        ("’", [label.startswith(("claude-", "glm-")) for label in labels]),
        ("—", [label.startswith("gemini-") for label in labels]),
    )
    for marker, disallowed in rules:
        rows = passages.str.contains(marker, regex=False).to_numpy(dtype=bool)
        rows = rows & np.isin(predictions, labels[disallowed])
        probs[np.ix_(rows, disallowed)] = 0.0
        total = probs.sum(axis=1)
        update = rows & (total > 0)
        probs[update] /= total[update, None]
        predictions[update] = labels[probs[update].argmax(axis=1)]
    return predictions
