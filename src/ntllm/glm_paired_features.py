"""Same-token GLM comparison following Kirill Tymchishin's Apache-2.0 Kaggle v24."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_trace(cache, tag, ids):
    pooled = pd.read_csv(cache / f"{tag}_features.csv", usecols=["id", "n_tok"], dtype={"id": str})
    if not pooled.id.is_unique or pooled.id.tolist() != ids:
        raise ValueError(f"{tag} CSV ids are not the requested ordered ids")
    with np.load(cache / f"{tag}_scores.npz", allow_pickle=False) as archive:
        scores, offsets, prefixes, tokens = (archive[key] for key in ("scores", "offsets", "n_prefix", "tokens"))
    counts = np.diff(offsets)
    if (
        scores.ndim != 2 or scores.shape[1] != 4
        or offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(scores)
        or prefixes.shape != (len(ids),) or tokens.shape != (len(scores),)
        or not np.array_equal(counts, pooled.n_tok.to_numpy()) or (counts <= 0).any()
        or not np.isfinite(scores).all()
    ):
        raise ValueError(f"{tag} scores, prefix counts, or token ids have an inconsistent layout")
    return scores, offsets, prefixes, tokens


def aligned_rows(first, second):
    _, left_offset, left_prefix, left_token = first
    _, right_offset, right_prefix, right_token = second
    rows = (left_prefix == right_prefix) & (np.diff(left_offset) == np.diff(right_offset))
    for i in np.flatnonzero(rows):
        rows[i] = np.array_equal(
            left_token[left_offset[i]:left_offset[i + 1]],
            right_token[right_offset[i]:right_offset[i + 1]],
        )
    return rows


def paired_statistics(flash, full):
    a, b = flash.astype(np.float64), full.astype(np.float64)
    delta = a[:, 0] - b[:, 0]
    average_entropy = (a[:, 1] + b[:, 1]) / 2
    easy = average_entropy <= np.median(average_entropy)
    hard = ~easy
    adjusted = (
        (a[:, 0] + a[:, 1]) / np.sqrt(np.maximum(a[:, 2] - a[:, 1] ** 2, 1e-6))
        - (b[:, 0] + b[:, 1]) / np.sqrt(np.maximum(b[:, 2] - b[:, 1] ** 2, 1e-6))
    )
    q10, q90 = np.quantile(delta, (0.1, 0.9))
    return {
        "paired:lp_spread": delta.std(),
        "paired:lp_win": (delta > 0).mean(),
        "paired:lp_easy": delta[easy].mean(),
        "paired:lp_hard": delta[hard].mean() if hard.any() else delta.mean(),
        "paired:lp_q10": q10,
        "paired:lp_q90": q90,
        "paired:adjusted_win": (adjusted > 0).mean(),
        "paired:adjusted_easy": adjusted[easy].mean(),
        "paired:adjusted_hard": adjusted[hard].mean() if hard.any() else adjusted.mean(),
        "paired:rank_win": (a[:, 3] < b[:, 3]).mean(),
    }


def paired_features(cache, ids, token_limit=320):
    """Return paired columns only when every requested row aligns."""
    ids = list(ids)
    if not ids or token_limit < 1:
        raise ValueError("Pairing requires a nonempty ordered id list and positive token limit")
    report = {"rows": len(ids), "aligned_rows": None, "coverage": None, "token_limit": token_limit}
    try:
        flash = read_trace(Path(cache), "flash", ids)
        full = read_trace(Path(cache), "glm53", ids)
    except (OSError, ValueError, KeyError) as error:
        return None, {**report, "mode": "reference_aggregate_features", "reason": str(error)}

    matched = aligned_rows(flash, full)
    report.update(aligned_rows=int(matched.sum()), coverage=float(matched.mean()))
    if not matched.all():
        return None, {**report, "mode": "reference_aggregate_features",
                      "reason": "GLM and Flash differ in prefix counts or scored token ids"}

    a, offsets, _, _ = flash
    b = full[0]
    values = [
        paired_statistics(a[first:min(last, first + token_limit)], b[first:min(last, first + token_limit)])
        for first, last in zip(offsets[:-1], offsets[1:])
    ]
    result = pd.DataFrame(values, index=pd.Index(ids, name="id"))
    if not np.isfinite(result.to_numpy()).all():
        return None, {**report, "mode": "reference_aggregate_features", "reason": "Non-finite paired statistics"}
    return result, {**report, "mode": "paired_tokens"}
