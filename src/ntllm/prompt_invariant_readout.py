"""Compare raw and prompt-regularized frozen Qwen readouts on all 18 sources."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from .gemini_sentence_order import compare, official_oof
    from .gemini_tail_contrast import cached_features
else:
    from gemini_sentence_order import compare, official_oof
    from gemini_tail_contrast import cached_features


WEIGHT = 0.20


def prompt_regularize(train, test, prompt_ids):
    _, inverse, counts = np.unique(prompt_ids, return_inverse=True, return_counts=True)
    prompt_means = np.zeros((len(counts), train.shape[1]), dtype=train.dtype)
    np.add.at(prompt_means, inverse, train)
    prompt_means /= counts[:, None]
    prompt_means -= prompt_means.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(prompt_means @ prompt_means.T)
    keep = eigenvalues > eigenvalues[-1] * 1e-6
    energy = eigenvalues[keep]
    directions = (vectors[:, keep].T @ prompt_means) / np.sqrt(energy)[:, None]
    # Reparameterize the L2 prior to penalize score variance along training prompt means.
    shrink = 1.0 - 1.0 / np.sqrt(1.0 + energy / np.median(energy))
    return tuple(rows - ((rows @ directions.T) * shrink) @ directions for rows in (train, test))


def readout(train, test, y):
    head = LogisticRegression(C=16, max_iter=180)
    head.fit(train, y)
    return head.decision_function(test)


def evaluate(values, frame, labels, reference):
    y = pd.Categorical(frame.model, categories=labels).codes
    folds = frame.fold.to_numpy()
    words = frame.text.fillna("").str.split().str.len().to_numpy()
    old = reference.argmax(axis=1)
    raw, regularized = np.empty_like(reference), np.empty_like(reference)
    for fold in np.unique(folds):
        held = folds == fold
        scaler = StandardScaler()
        train = scaler.fit_transform(values[~held]) / math.sqrt(values.shape[1])
        test = scaler.transform(values[held]) / math.sqrt(values.shape[1])
        raw[held] = readout(train, test, y[~held])
        train, test = prompt_regularize(train, test, frame.prompt_id.to_numpy()[~held])
        regularized[held] = readout(train, test, y[~held])
        print("Fold", fold, "complete", flush=True)
    scores = {"raw": raw, "prompt_regularized": regularized}
    predicted = {name: logits.argmax(axis=1) for name, logits in scores.items()}
    predicted.update({f"blend_{name}": ((1 - WEIGHT) * reference + WEIGHT * logits).argmax(axis=1)
                      for name, logits in scores.items()})
    masks = {"all": np.ones(len(frame), dtype=bool), "under_100": words < 100, "100_plus": words >= 100}
    masks.update({f"fold_{fold}": folds == fold for fold in np.unique(folds)})
    variants = {}
    for name, pred in predicted.items():
        variants[name] = {subset: compare(y[mask], old[mask], pred[mask]) for subset, mask in masks.items()}
        variants[name]["recall"] = {label: float((pred[y == i] == i).mean()) for i, label in enumerate(labels)}
    predictions = frame[["id", "prompt_id", "model", "fold"]].copy()
    for name, logits in scores.items():
        predictions[[f"{name}__{label}" for label in labels]] = logits
    return variants, predictions


def main():
    frame, labels, reference = official_oof(Path("/kaggle/input"))
    cached, layers = cached_features(Path("/kaggle/input"), frame.id)
    values = cached["tail_dc"]
    del cached
    output = Path("/kaggle/working")
    with threadpool_limits(limits=4):
        variants, predictions = evaluate(values, frame, labels, reference)
    result = {"rows": len(frame), "layers": layers, "weight": WEIGHT, "variants": variants}
    (output / "prompt_invariant_readout.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_parquet(output / "prompt_invariant_readout_oof.parquet", index=False)
    for name, stats in variants.items():
        print(name, stats["all"], flush=True)


if __name__ == "__main__":
    main()
