"""Direction C: is kimi-k3's cross-family collapse a data problem or a boundary problem?

Read-only diagnostics over three inputs:

* the external pool (``external_expanded.parquet``),
* the official 150-prompt set (``official.parquet``),
* the saved five-fold OOF probabilities (``oof/fold*/rows.parquet`` and
  ``oof/fold*/plain_centered_gaussian.parquet``).

Every function returns a plain ``dict`` so the whole run serialises to JSON.  The
module is written for a Kaggle CPU notebook (4 vCPU / 30 GB) and never writes to
the input data.  Local machines should only import it; see
``runs/kaggle_c_20261116/run_c.py`` for the entry point.

Design choices worth knowing when reading the output:

* The pool is sampled before any TF-IDF fit (class caps), so ``meta.sample``
  records exactly how many rows entered each stage.  Rates are estimates on that
  sample, not on all 238k rows, unless the key says ``full_pool``.
* Near-duplicate search uses hashed **character 5-grams**; cosine on binary
  vectors lower-bounds Jaccard, so ``cos >= 0.8`` is a *necessary* condition and
  the reported Jaccard is re-computed exactly on the candidate pairs.
* The separability model is a plain logistic regression on char_wb 3-5 TF-IDF.
  Its CV uses prompt-grouped folds and refits the vectoriser per fold, so no
  text from the validation prompt is ever seen by the vectoriser.
* Transfer to the official set trains on the pool only.  The official set is
  balanced by construction (150 per class) while the pool is not, so a 0.5
  threshold is meaningless there; AUC/PR-AUC and prevalence-matched thresholds
  are reported alongside.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # scipy is normally present on Kaggle; everything below has a numpy fallback.
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - exercised only on stripped images
    _scipy_stats = None

__all__ = [
    "DEFAULT_LABEL",
    "DiagnosticsConfig",
    "family_of",
    "locate_inputs",
    "load_tables",
    "run_all",
    "summarize",
    "write_results",
]

DEFAULT_LABEL = "kimi-k3"
FAMILY_ORDER = ("claude", "gemini", "glm", "gpt", "kimi", "qwen")

_WORD_RE = re.compile(r"[a-z0-9']+")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass
class DiagnosticsConfig:
    """Row caps and model knobs.  Every value is logged into ``meta.config``.

    The defaults are sized for a ~4 vCPU Kaggle CPU notebook and finish well
    inside an hour.  ``NTLLM_C_FAST=1`` shrinks the expensive stages.
    """

    label: str = DEFAULT_LABEL
    seed: int = 17
    n_folds: int = 5

    # kimi-vs-rest / kimi-vs-family CV
    sep_pos_cap: int = 4000
    sep_per_class_cap: int = 450

    # TF-IDF
    max_features: int = 200_000
    min_df: int = 3
    ngram_min: int = 3
    ngram_max: int = 5

    # near-duplicate sampling (hashed char 5-grams)
    near_kimi_queries: int = 2000
    near_per_class_cap: int = 800
    near_within_kimi_queries: int = 800
    near_within_kimi_db: int = 3000
    near_k: int = 5
    jaccard_pairs: int = 400

    # coverage: per-class pool is used in full when smaller than this
    cover_pool_cap: int = 8000

    # optional wall-clock guard for the family fits inside separability
    separability_minutes: float = 25.0

    # optional wall-clock guard for the transfer stage
    budget_minutes: float = 55.0

    @classmethod
    def fast(cls) -> "DiagnosticsConfig":
        return cls(
            sep_pos_cap=2000,
            sep_per_class_cap=250,
            near_kimi_queries=800,
            near_per_class_cap=300,
            near_within_kimi_queries=300,
            near_within_kimi_db=1200,
            jaccard_pairs=150,
            cover_pool_cap=3000,
            separability_minutes=15.0,
        )


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def family_of(model: str) -> str:
    """Provider family of a model label (``gpt-5.6-luna`` -> ``gpt``)."""
    name = str(model).strip().lower()
    head = re.split(r"[-_/.]", name, maxsplit=1)[0]
    if head in FAMILY_ORDER:
        return head
    for fam in FAMILY_ORDER:
        if name.startswith(fam):
            return fam
    return "other"


def _norm_text(text) -> str:
    """Lowercase + whitespace collapse.  Punctuation and casing of the original
    column are never modified - normalisation lives only in this copy."""
    return _WS_RE.sub(" ", str(text).lower().replace("\u00a0", " ")).strip()


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()


def _log_duration(log: Callable[[str], None], name: str, seconds: float) -> None:
    log(f"  [{name}] {seconds:.1f}s")


def _fmt_metric(x) -> str:
    return "n/a" if x is None else f"{float(x):.3f}"


def _jsonable(obj):
    """Recursively convert numpy/pandas scalars and drop non-finite floats."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


# --------------------------------------------------------------------------- #
# statistics with numpy fallback
# --------------------------------------------------------------------------- #


def _gser(a: float, x: float) -> float:
    """Lower regularised incomplete gamma series (Numerical Recipes)."""
    itmax, eps = 500, 3e-12
    ap, total, delta = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma continued fraction."""
    itmax, eps, fpmin = 500, 3e-12, 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = fpmin if abs(d) < fpmin else d
        c = b + an / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_sf(x: float, dof: int) -> float | None:
    if dof <= 0:
        return None
    if _scipy_stats is not None:
        return float(_scipy_stats.chi2.sf(x, dof))
    if x <= 0:
        return 1.0
    a = dof / 2.0
    hx = x / 2.0
    if hx < a + 1.0:
        return max(0.0, 1.0 - _gser(a, hx))
    return float(min(1.0, _gcf(a, hx)))


def chi2_contingency(table: Sequence[Sequence[float]]) -> dict:
    """Chi-square test of independence plus Cramer's V."""
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return dict(chi2=None, dof=None, p=None, cramers_v=None, n=0)
    keep_r = arr.sum(1) > 0
    keep_c = arr.sum(0) > 0
    arr = arr[keep_r][:, keep_c]
    total = float(arr.sum())
    if arr.shape[0] < 2 or arr.shape[1] < 2 or total <= 0:
        return dict(chi2=None, dof=None, p=None, cramers_v=None, n=int(total))
    expected = arr.sum(1, keepdims=True) @ arr.sum(0, keepdims=True) / total
    chi2 = float(((arr - expected) ** 2 / np.where(expected > 0, expected, 1.0)).sum())
    dof = int((arr.shape[0] - 1) * (arr.shape[1] - 1))
    v = math.sqrt(chi2 / (total * min(arr.shape[0] - 1, arr.shape[1] - 1)))
    return dict(chi2=chi2, dof=dof, p=_chi2_sf(chi2, dof), cramers_v=float(v), n=int(total))


def ks_2samp(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sample Kolmogorov-Smirnov.  ``D`` grows with distribution distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(D=None, p=None, n_a=int(len(a)), n_b=int(len(b)))
    if _scipy_stats is not None:
        res = _scipy_stats.ks_2samp(a, b, alternative="two-sided", method="asymp")
        return dict(D=float(res.statistic), p=float(res.pvalue), n_a=int(len(a)), n_b=int(len(b)))
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return dict(D=d, p=float(min(1.0, max(0.0, p))), n_a=int(len(a)), n_b=int(len(b)))


def auc_score(y: Sequence[int], score: Sequence[float]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney U), tolerant of ties."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    if _scipy_stats is not None:
        return float(_scipy_stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided").statistic) / (n1 * n0)
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y: Sequence[int], score: Sequence[float]) -> float | None:
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y == 1].mean())


def binary_metrics(y: Sequence[int], score: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(score, dtype=float)
    pred = p >= threshold
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    recall = float(pred[y == 1].mean()) if n_pos else None
    specificity = float((~pred[y == 0]).mean()) if n_neg else None
    precision = float(y[pred].mean()) if pred.any() else None
    ba = None
    if recall is not None and specificity is not None:
        ba = 0.5 * (recall + specificity)
    return dict(
        n=int(len(y)),
        n_pos=n_pos,
        n_neg=n_neg,
        auc=auc_score(y, p),
        average_precision=average_precision(y, p),
        threshold=float(threshold),
        recall=recall,
        specificity=specificity,
        precision=precision,
        balanced_accuracy=ba,
        predicted_positive=int(pred.sum()),
    )


def _bin_order(bins: Iterable) -> list:
    def key(b):
        nums = re.findall(r"\d+", str(b))
        return (int(nums[0]) if nums else -1, str(b))

    return sorted({str(b) for b in bins if str(b) not in ("nan", "None")}, key=key)


# --------------------------------------------------------------------------- #
# input discovery / loading
# --------------------------------------------------------------------------- #

_DEFAULT_ROOTS = ("/kaggle/input", "data", "runs")
_POOL_COLUMNS = (
    "text",
    "model",
    "prompt_id",
    "n_words",
    "word_bin",
    "source_dataset",
    "origin",
    "source_pool",
    "generation_status",
    "provenance_tier",
)
_OFFICIAL_COLUMNS = ("id", "prompt_id", "text", "model", "n_words", "word_bin", "fold")
_OOF_ROWS_COLUMNS = ("id", "prompt_id", "model", "n_words", "word_bin", "fold")


def _collect(roots: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            hits.extend(p for p in root.glob(pattern) if p.is_file())
    seen, unique = set(), []
    for path in hits:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _pick(
    hits: Sequence[Path],
    prefer: Sequence[str],
    what: str,
    roots: Sequence[Path],
    score_fn: Callable[[Path], float] | None = None,
) -> Path:
    if not hits:
        searched = ", ".join(f"{r}/**/{what}" for r in roots)
        raise FileNotFoundError(
            f"找不到 {what}。已搜索: {searched}\n"
            "Kaggle 上请确认该文件已作为 dataset 挂载到 /kaggle/input 下（目录结构会被保留）。"
        )

    def score(path: Path) -> float:
        text = str(path)
        bonus = sum((len(prefer) - i) * 10 for i, token in enumerate(prefer) if token in text)
        extra = score_fn(path) if score_fn is not None else 0.0
        return bonus + extra - len(text) / 1e4

    return sorted(hits, key=score, reverse=True)[0]


def _has_columns(path: Path, required: Sequence[str]) -> bool:
    """Schema-only check, so a stray parquet with the same name is not picked."""
    try:
        import pyarrow.parquet as pq

        return set(required) <= set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return False


def _num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def locate_inputs(roots: Sequence[str | Path] | None = None) -> dict:
    """Find the three inputs by glob, tolerating any Kaggle mount layout."""
    env_root = os.environ.get("NTLLM_DATA_ROOT")
    root_list = [Path(p) for p in (roots or ([env_root] if env_root else []))]
    root_list += [Path(p) for p in _DEFAULT_ROOTS]
    # Deduplicate roots while keeping order.  Only data/ and runs/ are scanned by
    # default so that a local .venv is never traversed.
    seen, uniq_roots = set(), []
    for root in root_list:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            uniq_roots.append(root)

    external_hits = [
        p for p in _collect(uniq_roots, ["**/external_expanded.parquet", "**/external_pool.parquet"])
        if _has_columns(p, ("text", "model", "prompt_id"))
    ]
    external = _pick(
        external_hits,
        ["q8_protocol"],
        "external_expanded.parquet",
        uniq_roots,
        score_fn=lambda p: math.log1p(_num_rows(p)),
    )

    official_hits = [
        p for p in _collect(uniq_roots, ["**/official.parquet"])
        if _has_columns(p, ("id", "prompt_id", "text", "model", "n_words", "word_bin", "fold"))
    ]
    official = _pick(
        official_hits,
        ["q8/official", "q8_official"],
        "official.parquet",
        uniq_roots,
        # The official set is 2,699 rows; prefer a candidate with that size.
        score_fn=lambda p: -abs(_num_rows(p) - 2699) / 100.0,
    )

    fold_hits = _collect(uniq_roots, ["**/oof/fold*/rows.parquet", "**/oof/fold*/plain_centered_gaussian.parquet"])
    by_oof: dict[Path, set] = {}
    for hit in fold_hits:
        oof_dir = hit.parent.parent
        by_oof.setdefault(oof_dir, set()).add(hit.parent.name)
    candidates = []
    for oof_dir, folds in by_oof.items():
        rows = [oof_dir / f / "rows.parquet" for f in folds]
        probs = [oof_dir / f / "plain_centered_gaussian.parquet" for f in folds]
        complete = sum(r.is_file() and p.is_file() for r, p in zip(rows, probs))
        candidates.append((complete, len(folds), oof_dir))
    if not candidates:
        raise FileNotFoundError(
            "找不到 OOF 目录（期望 **/oof/fold*/{rows,plain_centered_gaussian}.parquet）。"
            "Kaggle 上请确认 runs/q8_official_20261113/oof/ 已挂载。"
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    oof_dir = candidates[0][2]
    folds = sorted([p.name for p in oof_dir.glob("fold*") if p.is_dir()])
    if not folds:
        raise FileNotFoundError(f"OOF 目录 {oof_dir} 下没有 fold*/ 子目录。")

    return {
        "external": external,
        "official": official,
        "oof_dir": oof_dir,
        "folds": folds,
    }


def _read_parquet(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    keep = [c for c in columns if c in available]
    missing = [c for c in columns if c not in available]
    frame = pd.read_parquet(path, columns=keep)
    if missing:
        frame.attrs["missing_columns"] = missing
    return frame


def load_tables(paths: Mapping) -> dict:
    """Load the pool, the official set and the joined OOF frame."""
    pool = _read_parquet(Path(paths["external"]), _POOL_COLUMNS)
    official = _read_parquet(Path(paths["official"]), _OFFICIAL_COLUMNS)
    for name, frame in (("external", pool), ("official", official)):
        missing = [c for c in ("text", "model", "prompt_id") if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} 缺少必需列 {missing}；请检查挂载的文件是否选错。")
        frame["text"] = frame["text"].astype(str)
        frame["model"] = frame["model"].astype(str)
        frame["prompt_id"] = frame["prompt_id"].astype(str)
        if "n_words" in frame.columns:
            frame["n_words"] = pd.to_numeric(frame["n_words"], errors="coerce")
        else:
            frame["n_words"] = frame["text"].str.split().str.len()
    pool = pool[pool["text"].str.len() > 0].reset_index(drop=True)
    official = official[official["text"].str.len() > 0].reset_index(drop=True)
    if "id" not in official.columns:
        official["id"] = [f"off{i:05d}" for i in range(len(official))]
    official["id"] = official["id"].astype(str)

    oof_frames = []
    for fold in paths["folds"]:
        fold_dir = Path(paths["oof_dir"]) / fold
        rows_path = fold_dir / "rows.parquet"
        prob_path = fold_dir / "plain_centered_gaussian.parquet"
        if not (rows_path.is_file() and prob_path.is_file()):
            continue
        rows = _read_parquet(rows_path, _OOF_ROWS_COLUMNS)
        probs = pd.read_parquet(prob_path)
        prob_cols = [c for c in probs.columns if c.startswith("prob__")]
        keep = [c for c in ("id", "fold", "true_model", "pred_model") if c in probs.columns]
        probs = probs[keep + prob_cols]
        joined = rows.merge(probs, on=["id", "fold"], how="inner", validate="one_to_one")
        if len(joined) != len(probs):
            raise ValueError(f"{fold}: rows 与概率文件行数不一致 ({len(joined)} vs {len(probs)})。")
        oof_frames.append(joined)
    if not oof_frames:
        raise FileNotFoundError(f"{paths['oof_dir']} 下没有任何可用的 fold*/rows.parquet + 概率文件。")
    oof = pd.concat(oof_frames, ignore_index=True)

    return {"pool": pool, "official": official, "oof": oof}


# --------------------------------------------------------------------------- #
# 1. sample size, length and composition
# --------------------------------------------------------------------------- #


def _num_stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return dict(n=0)
    return dict(
        n=int(len(arr)),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
    )


def _word_bin_shares(frame: pd.DataFrame) -> dict:
    if "word_bin" not in frame:
        return {}
    counts = frame["word_bin"].dropna().astype(str).value_counts()
    total = counts.sum()
    return {str(k): float(v / total) for k, v in counts.items()} if total else {}


def length_diagnostics(pool: pd.DataFrame, official: pd.DataFrame, label: str) -> dict:
    """Length distributions of kimi vs every other class, in both datasets."""
    out: dict = {"pool": {}, "official": {}}

    for name, frame in (("pool", pool), ("official", official)):
        if frame.empty:
            out[name] = {"empty": True}
            continue
        kimi = frame[frame["model"] == label]
        rest = frame[frame["model"] != label]
        block = {
            "kimi": _num_stats(kimi["n_words"]),
            "rest": _num_stats(rest["n_words"]),
            "ks_kimi_vs_rest": ks_2samp(kimi["n_words"], rest["n_words"]),
            "word_bin_share_kimi": _word_bin_shares(kimi),
            "word_bin_share_rest": _word_bin_shares(rest),
            "mean_n_words_by_class": {
                str(k): float(v) for k, v in frame.groupby("model")["n_words"].mean().items()
            },
            "rows_by_class": {str(k): int(v) for k, v in frame["model"].value_counts().items()},
        }
        orders = _bin_order(set(block["word_bin_share_kimi"]) | set(block["word_bin_share_rest"]))
        if orders:
            table = [
                [block["word_bin_share_kimi"].get(b, 0.0) * len(kimi), block["word_bin_share_rest"].get(b, 0.0) * len(rest)]
                for b in orders
            ]
            block["word_bin_chi2"] = chi2_contingency(table)
            block["word_bin_order"] = orders
        # Per-class KS distance against kimi: which class looks like kimi in length?
        per_class = {}
        for cls, group in frame.groupby("model"):
            if str(cls) == label:
                continue
            per_class[str(cls)] = ks_2samp(kimi["n_words"], group["n_words"])
        block["ks_kimi_vs_class"] = dict(
            sorted(per_class.items(), key=lambda kv: (kv[1]["D"] if kv[1]["D"] is not None else 0.0))
        )
        out[name] = block

    return out


def _composition_block(frame: pd.DataFrame, is_kimi: np.ndarray, column: str, top: int = 8) -> dict:
    if column not in frame:
        return {"missing": True}
    values = frame[column].fillna("<NA>").astype(str)
    kimi_counts = values[is_kimi].value_counts()
    rest_counts = values[~is_kimi].value_counts()
    kimi_total = int(kimi_counts.sum())
    rest_total = int(rest_counts.sum())
    if kimi_total == 0 or rest_total == 0:
        return {"empty": True}
    kimi_share = kimi_counts / kimi_total
    rest_share = rest_counts / rest_total
    categories = list(kimi_share.index[:top])
    for cat in rest_share.index[:top]:
        if cat not in categories:
            categories.append(cat)
    all_cats = kimi_share.index.union(rest_share.index)
    tvd = float(
        0.5
        * (
            kimi_share.reindex(all_cats).fillna(0.0)
            - rest_share.reindex(all_cats).fillna(0.0)
        ).abs().sum()
    )
    table = [
        [float(kimi_counts.get(c, 0)), float(rest_counts.get(c, 0))]
        for c in categories
    ]
    other_row = [float(kimi_total - kimi_counts.reindex(categories).fillna(0).sum()),
                 float(rest_total - rest_counts.reindex(categories).fillna(0).sum())]
    chi2 = chi2_contingency(table + [other_row])
    return {
        "kimi_top": {str(k): dict(n=int(v), share=float(v / kimi_total)) for k, v in kimi_counts.head(top).items()},
        "rest_top": {str(k): dict(n=int(v), share=float(v / rest_total)) for k, v in rest_counts.head(top).items()},
        "tvd": tvd,
        "kimi_share_outside_listed_categories": float(
            kimi_total - kimi_counts.reindex(categories).fillna(0).sum()
        ) / kimi_total,
        "chi2": chi2,
        "n_kimi": kimi_total,
        "n_rest": rest_total,
    }


def _prompt_table(frame: pd.DataFrame, label: str) -> dict:
    per_class = []
    for cls, group in frame.groupby("model"):
        rows_per_prompt = group.groupby("prompt_id").size()
        per_class.append(
            dict(
                model=str(cls),
                is_focus=str(cls) == label,
                rows=int(len(group)),
                prompts=int(rows_per_prompt.size),
                rows_per_prompt_median=float(rows_per_prompt.median()),
                rows_per_prompt_p10=float(rows_per_prompt.quantile(0.10)),
                rows_per_prompt_p90=float(rows_per_prompt.quantile(0.90)),
                rows_per_prompt_max=int(rows_per_prompt.max()),
            )
        )
    per_class.sort(key=lambda r: (not r["is_focus"], -r["rows"]))
    return {"by_class": per_class, "prompts_total": int(frame["prompt_id"].nunique())}


def composition_diagnostics(pool: pd.DataFrame, official: pd.DataFrame, label: str) -> dict:
    """Where the kimi rows come from, and how many prompts each class covers."""
    is_kimi = (pool["model"] == label).to_numpy()
    blocks = {
        col: _composition_block(pool, is_kimi, col)
        for col in ("source_dataset", "origin", "source_pool", "generation_status", "provenance_tier")
    }
    kimi = pool[pool["model"] == label]
    rest = pool[pool["model"] != label]
    return {
        "pool_columns": blocks,
        "pool_prompts": _prompt_table(pool, label),
        "official_prompts": _prompt_table(official, label),
        "kimi_rows": int(len(kimi)),
        "pool_rows": int(len(pool)),
        "kimi_share": float(len(kimi) / max(len(pool), 1)),
        "kimi_prompts_in_pool": int(kimi["prompt_id"].nunique()),
        "kimi_prompts_share_of_pool_prompts": float(
            kimi["prompt_id"].nunique() / max(pool["prompt_id"].nunique(), 1)
        ),
        "rest_prompts_in_pool": int(rest["prompt_id"].nunique()),
    }


# --------------------------------------------------------------------------- #
# 2. near duplicates
# --------------------------------------------------------------------------- #


def _hashed_char_ngrams(texts: Sequence[str], n_features: int = 2 ** 20):
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(5, 5),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    return hv.transform(list(texts))


def _shingles(text: str, n: int = 5) -> set:
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 0))}


def _jaccard(a: str, b: str, n: int = 5) -> float:
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _sample_by_class(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    parts = []
    for cls, group in frame.groupby("model", sort=False):
        take = min(len(group), cap)
        parts.append(group.sample(take, random_state=seed) if take < len(group) else group)
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _knn_cosines(query_texts: Sequence[str], db_texts: Sequence[str], k: int, log=None, tag=""):
    from sklearn.neighbors import NearestNeighbors

    nq, nd = len(query_texts), len(db_texts)
    if nq == 0 or nd == 0:
        return np.zeros((nq, 0)), np.zeros((nq, 0), dtype=int)
    k = int(min(k, nd))
    if log:
        log(f"  hashing {tag}: {nq} queries vs {nd} db rows")
    xq = _hashed_char_ngrams(query_texts)
    xd = _hashed_char_ngrams(db_texts)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=1)
    nn.fit(xd)
    dist, idx = nn.kneighbors(xq, return_distance=True)
    return 1.0 - dist, idx


def near_duplicate_diagnostics(pool: pd.DataFrame, label: str, cfg: DiagnosticsConfig, log=print) -> dict:
    """Exact duplicates, cross-class near duplicates and the nearest class."""
    kimi = pool[pool["model"] == label]
    others = pool[pool["model"] != label]
    out: dict = {"n_kimi_full_pool": int(len(kimi)), "n_pool_full": int(len(pool))}

    # ---- exact duplicates inside kimi (full pool) ------------------------- #
    kimi_norm = [_norm_text(t) for t in kimi["text"]]
    kimi_hash = np.array([_digest(t) for t in kimi_norm], dtype=object)
    counts = pd.Series(kimi_hash).value_counts()
    out["exact_within_kimi"] = dict(
        rows=int(len(kimi)),
        distinct=int(counts.size),
        duplicate_rows=int(len(kimi) - counts.size),
        duplicate_rate=float(1.0 - counts.size / max(len(kimi), 1)),
        max_cluster=int(counts.max()) if counts.size else 0,
        clusters_gt1=int((counts > 1).sum()),
    )
    prefix_counts = pd.Series([t[:64] for t in kimi_norm]).value_counts()
    out["prefix64_within_kimi"] = dict(
        distinct=int(prefix_counts.size),
        duplicate_rate=float(1.0 - prefix_counts.size / max(len(kimi), 1)),
        max_cluster=int(prefix_counts.max()) if prefix_counts.size else 0,
    )

    # ---- exact duplicates against other classes (full pool) --------------- #
    kimi_set = set(kimi_hash)
    hit_by_class: dict[str, int] = {}
    hit_rows = 0
    hit_digests: set = set()
    for text, model in zip(others["text"], others["model"]):
        digest = _digest(_norm_text(text))
        if digest in kimi_set:
            hit_rows += 1
            hit_digests.add(digest)
            hit_by_class[str(model)] = hit_by_class.get(str(model), 0) + 1
    out["exact_cross_class"] = dict(
        matched_other_rows=hit_rows,
        kimi_rows_with_any_exact_twin=int(sum(1 for h in kimi_hash if h in hit_digests)),
        by_other_class=dict(sorted(hit_by_class.items(), key=lambda kv: -kv[1])),
        rate_of_kimi_rows=float(sum(1 for h in kimi_hash if h in hit_digests) / max(len(kimi), 1)),
        rate_of_other_rows=float(hit_rows / max(len(others), 1)),
        note="matched_other_rows = 非 kimi 行中与某条 kimi 文本完全相同的行数",
    )

    # ---- near duplicates: kimi queries vs sampled other classes ----------- #
    queries = kimi.sample(min(len(kimi), cfg.near_kimi_queries), random_state=cfg.seed)
    db = _sample_by_class(others, cfg.near_per_class_cap, cfg.seed + 1)
    cos, idx = _knn_cosines(queries["text"].tolist(), db["text"].tolist(), cfg.near_k, log=log, tag="kimi-vs-others")
    db_models = db["model"].to_numpy()
    db_texts = db["text"].tolist()
    top1 = db_models[idx[:, 0]]
    top1_cos = cos[:, 0]
    near_mask = cos >= 0.8
    has_cross = near_mask.any(1)
    best_cross = pd.Series(np.where(near_mask, cos, np.nan).max(1))
    neighbor_models = db_models[idx]
    mean_best_cross_by_class = {}
    for cls in pd.unique(db_models):
        sel = near_mask & (neighbor_models == cls)
        per_row = np.where(sel, cos, np.nan).max(1)
        if np.isfinite(per_row).any():
            mean_best_cross_by_class[str(cls)] = float(np.nanmean(per_row))

    out["near_kimi_vs_others"] = dict(
        n_queries=int(len(queries)),
        n_db=int(len(db)),
        db_class_cap=int(cfg.near_per_class_cap),
        share_with_cross_class_cos_0_8=float(has_cross.mean()),
        share_with_cross_class_cos_0_6=float((cos >= 0.6).any(1).mean()),
        mean_top1_cos=float(top1_cos.mean()),
        median_top1_cos=float(np.median(top1_cos)),
        max_top1_cos=float(top1_cos.max()),
        mean_best_cross_cos=float(best_cross.mean()),
        top1_class_counts={str(k): int(v) for k, v in pd.Series(top1).value_counts().items()},
        top1_class_share={str(k): float(v) for k, v in pd.Series(top1).value_counts(normalize=True).items()},
        mean_top1_cos_by_class={
            str(k): float(v) for k, v in pd.Series(top1_cos).groupby(pd.Series(top1)).mean().items()
        },
        mean_best_cross_cos_by_class=dict(
            sorted(mean_best_cross_by_class.items(), key=lambda kv: -kv[1])
        ),
    )
    if mean_best_cross_by_class:
        out["near_kimi_vs_others"]["closest_class_by_mean_best_cos"] = max(
            mean_best_cross_by_class, key=mean_best_cross_by_class.get
        )
    if len(out["near_kimi_vs_others"]["top1_class_share"]):
        out["near_kimi_vs_others"]["closest_class_by_top1_share"] = next(
            iter(out["near_kimi_vs_others"]["top1_class_share"])
        )

    # ---- near duplicates inside kimi -------------------------------------- #
    if len(kimi) >= 4:
        q_in = kimi.sample(min(len(kimi), cfg.near_within_kimi_queries), random_state=cfg.seed + 2)
        d_in = kimi.sample(min(len(kimi), cfg.near_within_kimi_db), random_state=cfg.seed + 3)
        cos_in, _ = _knn_cosines(q_in["text"].tolist(), d_in["text"].tolist(), cfg.near_k, log=log, tag="kimi-vs-kimi")
        if cos_in.size:
            # The same row can land on both sides; if the top hit is the row itself
            # (cos == 1) fall back to the second neighbour.
            first = cos_in[:, 0].copy()
            if cos_in.shape[1] > 1:
                identical = np.isclose(first, 1.0)
                first[identical] = cos_in[identical, 1]
            out["near_within_kimi"] = dict(
                n_queries=int(len(q_in)),
                n_db=int(len(d_in)),
                share_cos_0_8=float((cos_in >= 0.8).any(1).mean()),
                share_cos_0_95=float((cos_in >= 0.95).any(1).mean()),
                mean_top1_cos=float(first.mean()),
                median_top1_cos=float(np.median(first)),
            )
        else:
            out["near_within_kimi"] = dict(skipped="no cosine values")
    else:
        out["near_within_kimi"] = dict(skipped="too few kimi rows")

    # ---- exact Jaccard check on candidate pairs --------------------------- #
    # One pair per query (its best neighbour), spread evenly across queries, so
    # the estimate is not biased toward the globally most similar pairs.
    candidates = []
    for qi in range(len(queries)):
        row = cos[qi]
        best = int(np.argmax(row))
        if row[best] >= 0.5:
            candidates.append((qi, best, float(row[best])))
    if len(candidates) > cfg.jaccard_pairs:
        step = len(candidates) / float(cfg.jaccard_pairs)
        candidates = [candidates[min(int(i * step), len(candidates) - 1)] for i in range(cfg.jaccard_pairs)]
    jac = []
    for qi, ki, c in candidates:
        j = _jaccard(_norm_text(queries["text"].iloc[qi]), _norm_text(db_texts[ki]))
        jac.append(dict(cos=c, jaccard=j, other_model=str(db_models[ki])))
    if jac:
        cos_arr = np.array([p["cos"] for p in jac])
        jac_arr = np.array([p["jaccard"] for p in jac])
        high = jac_arr[cos_arr >= 0.8]
        out["jaccard_check"] = dict(
            pairs=len(jac),
            mean_jaccard=float(jac_arr.mean()),
            median_jaccard=float(np.median(jac_arr)),
            share_jaccard_gt_0_8=float((jac_arr > 0.8).mean()),
            pairs_with_cos_ge_0_8=int(len(high)),
            share_jaccard_gt_0_8_when_cos_ge_0_8=float((high > 0.8).mean()) if len(high) else None,
            mean_jaccard_when_cos_ge_0_8=float(high.mean()) if len(high) else None,
            cos_jaccard_corr=float(np.corrcoef(cos_arr, jac_arr)[0, 1]) if len(jac) > 2 else None,
            examples=jac[:3],
        )
    else:
        out["jaccard_check"] = dict(pairs=0)

    return out


# --------------------------------------------------------------------------- #
# 3. training coverage of the official set
# --------------------------------------------------------------------------- #


def coverage_diagnostics(pool: pd.DataFrame, official: pd.DataFrame, label: str, cfg: DiagnosticsConfig, log=print) -> dict:
    """Can the pool reproduce the official texts, class by class?

    ``exact`` uses the full pool (hash join).  ``near`` uses hashed char 5-grams
    and is computed per class so that the neighbour is guaranteed same-class;
    the pool side is capped at ``cover_pool_cap`` rows per class and the cap is
    reported with every number.
    """
    official = official.copy()
    official["norm"] = [_norm_text(t) for t in official["text"]]
    official["hash"] = [_digest(t) for t in official["norm"]]

    pool_hash: dict[str, set] = {}
    for cls, group in pool.groupby("model"):
        pool_hash[str(cls)] = {_digest(_norm_text(t)) for t in group["text"]}
    all_pool_hash = set().union(*pool_hash.values()) if pool_hash else set()

    per_class: dict[str, dict] = {}
    matched_pool_prompt: dict[str, dict[str, str]] = {}
    for cls, group in official.groupby("model"):
        cls = str(cls)
        exact_same = float(group["hash"].isin(pool_hash.get(cls, set())).mean())
        exact_any = float(group["hash"].isin(all_pool_hash).mean())
        pool_cls = pool[pool["model"] == cls]
        total_rows = int(len(pool_cls))
        used = min(total_rows, cfg.cover_pool_cap)
        if used == 0:
            per_class[cls] = dict(n=int(len(group)), exact_same_class=exact_same, exact_any_class=exact_any,
                                  near_0_8=None, near_0_6=None, median_max_cos=None, pool_rows_used=0)
            continue
        if used < total_rows:
            pool_cls = pool_cls.sample(used, random_state=cfg.seed + 7)
        cos, idx = _knn_cosines(group["text"].tolist(), pool_cls["text"].tolist(), 3, log=log, tag=f"coverage:{cls}")
        best = cos[:, 0]
        best_idx = idx[:, 0]
        per_class[cls] = dict(
            n=int(len(group)),
            exact_same_class=exact_same,
            exact_any_class=exact_any,
            near_0_8=float((best >= 0.8).mean()),
            near_0_6=float((best >= 0.6).mean()),
            median_max_cos=float(np.median(best)),
            p90_max_cos=float(np.percentile(best, 90)),
            pool_rows_used=int(used),
            pool_rows_total=total_rows,
            pool_rows_sampled=bool(used < total_rows),
        )
        matched_pool_prompt[cls] = {
            str(oid): str(pool_cls["prompt_id"].iloc[int(pid)])
            for oid, pid, c in zip(group["id"].astype(str), best_idx, best)
            if c >= 0.8
        }

    focus = per_class.get(label, {})
    others = [v for k, v in per_class.items() if k != label and v.get("near_0_8") is not None]
    out = {
        "per_class": dict(sorted(per_class.items(), key=lambda kv: -(kv[1].get("near_0_8") or -1))),
        "focus": label,
        "focus_near_0_8": focus.get("near_0_8"),
        "other_mean_near_0_8": float(np.mean([v["near_0_8"] for v in others])) if others else None,
        "focus_exact_same_class": focus.get("exact_same_class"),
        "other_mean_exact_same_class": float(np.mean([v["exact_same_class"] for v in others])) if others else None,
        "near_0_8_gap_vs_other_mean": (
            float(focus.get("near_0_8") - np.mean([v["near_0_8"] for v in others]))
            if others and focus.get("near_0_8") is not None
            else None
        ),
    }

    # ---- prompt-level agreement ------------------------------------------ #
    id_to_prompt = dict(zip(official["id"].astype(str), official["prompt_id"].astype(str)))
    matched_per_prompt: dict[str, list] = {}
    for cls, mapping in matched_pool_prompt.items():
        for oid, pool_prompt in mapping.items():
            pid = id_to_prompt.get(str(oid))
            if pid is None:
                continue
            matched_per_prompt.setdefault(str(pid), []).append((cls, pool_prompt))
    per_prompt_counts = {pid: len(v) for pid, v in matched_per_prompt.items()}
    votes = []
    for pid, pairs in matched_per_prompt.items():
        counter: dict[str, int] = {}
        for _, pool_prompt in pairs:
            counter[pool_prompt] = counter.get(pool_prompt, 0) + 1
        votes.append(max(counter.values()))
    out["prompt_level"] = dict(
        prompts_total=int(official["prompt_id"].nunique()),
        prompts_with_any_match=int(len(matched_per_prompt)),
        matched_classes_per_prompt={str(k): int(v) for k, v in
                                    pd.Series(list(per_prompt_counts.values())).value_counts().sort_index().items()},
        modal_pool_prompt_class_support={
            str(k): int(v) for k, v in pd.Series(votes).value_counts().sort_index().items()
        } if votes else {},
        note="matched_classes_per_prompt[k] = 官方 prompt 中有 k 个类别的文本在池里找到 cos>=0.8 的同 prompt 近邻",
    )
    return out


# --------------------------------------------------------------------------- #
# 4. separability: kimi vs rest, kimi vs every family
# --------------------------------------------------------------------------- #


def _vectorizer(cfg: DiagnosticsConfig):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )


def _logreg(seed: int = 17):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=300,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )


def _sample_pool(pool: pd.DataFrame, label: str, cfg: DiagnosticsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = pool[pool["model"] == label]
    if len(pos) > cfg.sep_pos_cap:
        pos = pos.sample(cfg.sep_pos_cap, random_state=cfg.seed)
    neg = _sample_by_class(pool[pool["model"] != label], cfg.sep_per_class_cap, cfg.seed + 1)
    return pos.reset_index(drop=True), neg.reset_index(drop=True)


def _group_splits(frame: pd.DataFrame, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = frame["prompt_id"].astype(str).to_numpy()
    n_unique = len(np.unique(groups))
    n_splits = int(min(n_folds, n_unique))
    if n_splits < 2:
        from sklearn.model_selection import KFold

        return list(KFold(n_splits=2, shuffle=True, random_state=seed).split(np.zeros(len(frame))))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(frame)), groups=groups))


def separability_diagnostics(pool: pd.DataFrame, label: str, cfg: DiagnosticsConfig, log=print) -> dict:
    """Prompt-grouped CV of kimi-vs-rest and kimi-vs-each-family logistic regression."""
    pos, neg = _sample_pool(pool, label, cfg)
    frame = pd.concat([pos, neg], ignore_index=True)
    frame["_y"] = (frame["model"] == label).astype(np.int8)
    frame["_fam"] = [family_of(m) for m in frame["model"]]
    y = frame["_y"].to_numpy()
    fam = frame["_fam"].to_numpy()
    splits = _group_splits(frame, cfg.n_folds, cfg.seed)

    families = [f for f in FAMILY_ORDER if f != family_of(label) and f in set(fam)]
    fam_scores = np.full((len(frame), len(families)), np.nan, dtype=float)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_rows = []
    sep_started = time.time()
    family_skips = 0

    for fold_i, (train_idx, valid_idx) in enumerate(splits):
        t0 = time.time()
        vec = _vectorizer(cfg)
        x_train = vec.fit_transform(frame["text"].iloc[train_idx].tolist())
        x_valid = vec.transform(frame["text"].iloc[valid_idx].tolist())
        clf = _logreg(cfg.seed + fold_i).fit(x_train, y[train_idx])
        oof[valid_idx] = clf.predict_proba(x_valid)[:, 1]
        fold_entry = {
            "fold": int(fold_i),
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "n_features": int(x_train.shape[1]),
            "kimi_vs_rest": binary_metrics(y[valid_idx], oof[valid_idx]),
            "families": {},
        }
        for fi, family in enumerate(families):
            if time.time() - sep_started > max(cfg.separability_minutes * 60.0 - 300.0, 0.0):
                family_skips += 1
                fold_entry["families_skipped"] = "time budget"
                break
            mask_train = (y[train_idx] == 1) | (fam[train_idx] == family)
            mask_valid = (y[valid_idx] == 1) | (fam[valid_idx] == family)
            if mask_train.sum() < 60 or mask_valid.sum() < 30 or len(np.unique(y[train_idx][mask_train])) < 2:
                continue
            sub = _logreg(cfg.seed + 100 + fold_i).fit(x_train[mask_train], y[train_idx][mask_train])
            scores = sub.predict_proba(x_valid[mask_valid])[:, 1]
            fam_scores[valid_idx[mask_valid], fi] = scores
            fold_entry["families"][family] = binary_metrics(y[valid_idx][mask_valid], scores)
        fold_entry["seconds"] = round(time.time() - t0, 1)
        fold_rows.append(fold_entry)
        auc_val = fold_entry["kimi_vs_rest"]["auc"]
        ba_val = fold_entry["kimi_vs_rest"]["balanced_accuracy"]
        log(
            f"  fold {fold_i}: auc={_fmt_metric(auc_val)} ba={_fmt_metric(ba_val)} "
            f"({fold_entry['seconds']:.0f}s)"
        )

    valid = np.isfinite(oof)
    overall = binary_metrics(y[valid], oof[valid])

    # ---- kimi vs every family, on the pooled OOF scores ------------------- #
    by_family = {}
    for fi, family in enumerate(families):
        sel = np.isfinite(fam_scores[:, fi])
        if sel.sum() < 30:
            continue
        by_family[family] = binary_metrics(y[sel], fam_scores[sel, fi])

    # ---- pairwise AUC from the kimi-vs-rest scores (no extra fit) -------- #
    pairwise_class = {}
    for cls in sorted(set(frame["model"]) - {label}):
        sel = valid & ((y == 1) | (frame["model"].to_numpy() == cls))
        if sel.sum() < 30:
            continue
        pairwise_class[cls] = auc_score(y[sel], oof[sel])
    pairwise_family = {}
    for family in families:
        sel = valid & ((y == 1) | (fam == family))
        if sel.sum() < 30:
            continue
        pairwise_family[family] = auc_score(y[sel], oof[sel])

    # ---- length-only control --------------------------------------------- #
    length_only = {
        "auc_n_words": auc_score(y[valid], frame["n_words"].to_numpy()[valid]),
        "kimi_mean_n_words": float(frame.loc[frame["_y"] == 1, "n_words"].mean()),
        "rest_mean_n_words": float(frame.loc[frame["_y"] == 0, "n_words"].mean()),
    }
    by_bin = {}
    for b in _bin_order(frame["word_bin"].dropna()):
        sel = valid & (frame["word_bin"].astype(str) == b).to_numpy()
        if sel.sum() < 40 or len(np.unique(y[sel])) < 2:
            continue
        by_bin[b] = dict(
            n=int(sel.sum()),
            n_kimi=int(y[sel].sum()),
            kimi_share=float(y[sel].mean()),
            auc=auc_score(y[sel], oof[sel]),
            ba=binary_metrics(y[sel], oof[sel])["balanced_accuracy"],
        )
    length_only["by_word_bin"] = by_bin
    known = [v["ba"] for v in by_bin.values() if v["ba"] is not None]
    length_only["mean_within_bin_ba"] = float(np.mean(known)) if known else None

    # ---- fold summary ----------------------------------------------------- #
    def _mean(key):
        vals = [f["kimi_vs_rest"][key] for f in fold_rows if f["kimi_vs_rest"][key] is not None]
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals))) if vals else None

    return {
        "sample": dict(
            n=int(len(frame)),
            n_kimi=int(y.sum()),
            n_rest=int(len(y) - y.sum()),
            kimi_share=float(y.mean()),
            prompts=int(frame["prompt_id"].nunique()),
            folds=[int(len(tr)) for tr, _ in splits],
            classes={str(k): int(v) for k, v in frame["model"].value_counts().items()},
            caps=dict(pos=cfg.sep_pos_cap, per_class=cfg.sep_per_class_cap),
        ),
        "folds": fold_rows,
        "oof_kimi_vs_rest": overall,
        "fold_auc": _mean("auc"),
        "fold_ba": _mean("balanced_accuracy"),
        "by_family": by_family,
        "pairwise_auc_class": pairwise_class,
        "pairwise_auc_family": pairwise_family,
        "length_only": length_only,
        "family_fits_skipped": family_skips,
        "_oof": (frame, oof),
    }


def official_domain_cv(official: pd.DataFrame, label: str, cfg: DiagnosticsConfig, log=print) -> dict:
    """In-domain control: can kimi be separated *inside* the official set?

    Same features and model as the transfer test, but trained on official rows
    with prompt-grouped folds.  Comparing this AUC with the pool->official AUC
    separates "the official texts are intrinsically overlapping" from "the pool
    is a different distribution".
    """
    y = (official["model"] == label).to_numpy().astype(int)
    splits = _group_splits(official, cfg.n_folds, cfg.seed)
    oof = np.full(len(official), np.nan, dtype=float)
    folds = []
    n_features = 0
    for fold_i, (train_idx, valid_idx) in enumerate(splits):
        vec = _vectorizer(cfg)
        x_train = vec.fit_transform(official["text"].iloc[train_idx].tolist())
        x_valid = vec.transform(official["text"].iloc[valid_idx].tolist())
        n_features = int(x_train.shape[1])
        clf = _logreg(cfg.seed + fold_i).fit(x_train, y[train_idx])
        oof[valid_idx] = clf.predict_proba(x_valid)[:, 1]
        folds.append(binary_metrics(y[valid_idx], oof[valid_idx]))
    metrics = binary_metrics(y, oof)
    metrics["n_features"] = n_features
    metrics["folds"] = folds
    metrics["note"] = "官方集内部 prompt 分组 CV；与 transfer 的差异即域偏移"
    log(f"  official-domain CV: auc={_fmt_metric(metrics['auc'])} ba={_fmt_metric(metrics['balanced_accuracy'])}")
    return metrics


def transfer_diagnostics(pool: pd.DataFrame, official: pd.DataFrame, label: str, cfg: DiagnosticsConfig, log=print) -> dict:
    """Train on the pool only, evaluate on the official 150-prompt set."""
    pos, neg = _sample_pool(pool, label, cfg)
    train = pd.concat([pos, neg], ignore_index=True)
    y_train = (train["model"] == label).to_numpy().astype(int)

    vec = _vectorizer(cfg)
    x_train = vec.fit_transform(train["text"].tolist())
    x_official = vec.transform(official["text"].tolist())
    clf = _logreg(cfg.seed).fit(x_train, y_train)
    scores = clf.predict_proba(x_official)[:, 1]
    train_scores = clf.predict_proba(x_train)[:, 1]

    y_official = (official["model"] == label).to_numpy().astype(int)
    official_prevalence = float(y_official.mean())
    # Threshold that would flag `official_prevalence` of the pool *training* rows.
    thr_prev = float(np.quantile(train_scores, 1.0 - official_prevalence))
    # Transductive quantile on the official scores (label-free, uses the known 1/18 design).
    thr_trans = float(np.quantile(scores, 1.0 - official_prevalence))

    out = {
        "train": dict(
            n=int(len(train)),
            n_kimi=int(y_train.sum()),
            kimi_share=float(y_train.mean()),
            n_features=int(x_train.shape[1]),
            caps=dict(pos=cfg.sep_pos_cap, per_class=cfg.sep_per_class_cap),
        ),
        "official": dict(
            n=int(len(official)),
            n_kimi=int(y_official.sum()),
            kimi_share=official_prevalence,
            prompts=int(official["prompt_id"].nunique()),
        ),
        "at_0.5": binary_metrics(y_official, scores, 0.5),
        "at_pool_prevalence_threshold": {
            **binary_metrics(y_official, scores, thr_prev),
            "threshold": thr_prev,
        },
        "at_official_quantile_threshold": {
            **binary_metrics(y_official, scores, thr_trans),
            "threshold": thr_trans,
        },
        "mean_score_by_true_class": {
            str(k): float(v) for k, v in pd.Series(scores).groupby(official["model"]).mean().items()
        },
        "median_score_by_true_class": {
            str(k): float(v) for k, v in pd.Series(scores).groupby(official["model"]).median().items()
        },
        "recall_by_word_bin": {},
    }
    frame = official.assign(_score=scores, _y=y_official)
    for b in _bin_order(frame["word_bin"].dropna()):
        sel = frame["word_bin"].astype(str) == b
        sub = frame[sel]
        if sub.empty:
            continue
        out["recall_by_word_bin"][b] = dict(
            n=int(len(sub)),
            n_kimi=int(sub["_y"].sum()),
            recall_at_0_5=float((sub.loc[sub["_y"] == 1, "_score"] >= 0.5).mean()) if sub["_y"].sum() else None,
            mean_score_kimi=float(sub.loc[sub["_y"] == 1, "_score"].mean()) if sub["_y"].sum() else None,
            mean_score_rest=float(sub.loc[sub["_y"] == 0, "_score"].mean()) if (~sub["_y"].astype(bool)).any() else None,
        )
    out["_scores"] = scores
    return out


# --------------------------------------------------------------------------- #
# 5. confusion structure of the saved OOF probabilities
# --------------------------------------------------------------------------- #


def _prob_matrix(oof: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    labels = [c[len("prob__"):] for c in oof.columns if c.startswith("prob__")]
    if not labels:
        raise ValueError("OOF 概率文件里没有 prob__<label> 列。")
    probs = oof[[f"prob__{l}" for l in labels]].to_numpy(dtype=np.float32)
    row_sum = probs.sum(1, keepdims=True)
    probs = probs / np.where(row_sum > 0, row_sum, 1.0)
    return labels, probs


def confusion_diagnostics(oof: pd.DataFrame, label: str) -> dict:
    labels, probs = _prob_matrix(oof)
    true = oof["true_model"].astype(str).to_numpy()
    li = {name: i for i, name in enumerate(labels)}
    order = np.argsort(-probs, axis=1)
    top1 = np.array(labels)[order[:, 0]]
    top2 = np.array(labels)[order[:, 1]] if probs.shape[1] > 1 else top1
    kimi_idx = li.get(label)
    if kimi_idx is None:
        raise ValueError(f"OOF 概率里没有 {label} 列。")
    kimi_rows = true == label
    n_kimi = int(kimi_rows.sum())
    rank_of_kimi = (probs > probs[:, [kimi_idx]]).sum(1) + 1

    top1_counts = pd.Series(top1[kimi_rows]).value_counts()
    top2_counts = pd.Series(top2[kimi_rows]).value_counts()
    pred_kimi = top1 == label
    true_of_pred_kimi = pd.Series(true[pred_kimi]).value_counts()

    # Full confusion matrix, rows = truth, cols = prediction.
    conf = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, top1):
        conf[li[t], li[p]] += 1
    per_class = {}
    for name, i in li.items():
        row = conf[i]
        per_class[name] = dict(
            support=int(row.sum()),
            recall=float(row[i] / row.sum()) if row.sum() else None,
            predicted_as_kimi=int(row[li[label]]) if label in li else 0,
            top_confusions={
                str(labels[j]): int(row[j]) for j in np.argsort(-row)[:4] if row[j] > 0
            },
        )

    kimi_pred = conf[li[label]]
    return {
        "n_rows": int(len(oof)),
        "n_kimi": n_kimi,
        "labels": labels,
        "confusion": conf.tolist(),
        "per_class": per_class,
        "kimi": dict(
            recall=float(kimi_pred[li[label]] / n_kimi) if n_kimi else None,
            top1_pred_counts={str(k): int(v) for k, v in top1_counts.items()},
            top1_pred_family_counts={
                str(k): int(v)
                for k, v in pd.Series([family_of(x) for x in top1[kimi_rows]]).value_counts().items()
            },
            top2_pred_counts={str(k): int(v) for k, v in top2_counts.items()},
            kimi_in_top2=float(((top1[kimi_rows] == label) | (top2[kimi_rows] == label)).mean()),
            mean_prob_kimi_on_kimi_rows=float(probs[kimi_rows, kimi_idx].mean()),
            median_prob_kimi_on_kimi_rows=float(np.median(probs[kimi_rows, kimi_idx])),
            mean_rank_of_kimi=float(rank_of_kimi[kimi_rows].mean()),
            median_rank_of_kimi=float(np.median(rank_of_kimi[kimi_rows])),
            mean_prob_kimi_on_non_kimi_rows=float(probs[~kimi_rows, kimi_idx].mean()),
            mean_top1_margin=float(
                (probs[kimi_rows, order[kimi_rows, 0]] - probs[kimi_rows, kimi_idx]).mean()
            ),
        ),
        "kimi_precision": dict(
            predicted_kimi_rows=int(pred_kimi.sum()),
            true_kimi_among_predicted=int((true[pred_kimi] == label).sum()),
            precision=float((true[pred_kimi] == label).mean()) if pred_kimi.any() else None,
            true_class_of_predicted_kimi={str(k): int(v) for k, v in true_of_pred_kimi.items()},
            predicted_kimi_by_true_class={
                str(k): int(v) for k, v in pd.Series(top1 == label).groupby(true).sum().items()
            },
        ),
    }


# --------------------------------------------------------------------------- #
# 6. length-controlled recall
# --------------------------------------------------------------------------- #


def length_conditioned_recall(oof: pd.DataFrame, label: str) -> dict:
    """Does kimi's recall gap survive inside word bins?

    If kimi recall is below the other-class mean in *every* bin, length does not
    explain the gap.  The standardised numbers reweight the per-bin recalls to a
    common bin distribution (all rows, or uniform), which is the counterfactual
    "same length mix as everyone else".
    """
    labels, probs = _prob_matrix(oof)
    true = oof["true_model"].astype(str).to_numpy()
    pred = np.array(labels)[probs.argmax(1)]
    bins = oof["word_bin"].astype(str).to_numpy()
    orders = _bin_order(set(bins))
    focus = label

    bin_stats = {}
    for b in orders:
        sel = bins == b
        if sel.sum() == 0:
            continue
        sub_true, sub_pred = true[sel], pred[sel]
        kimi_sel = sub_true == focus
        others = {}
        for cls in set(sub_true):
            if cls == focus:
                continue
            m = sub_true == cls
            others[cls] = float((sub_pred[m] == cls).mean())
        bin_stats[b] = dict(
            n=int(sel.sum()),
            n_kimi=int(kimi_sel.sum()),
            kimi_recall=float((sub_pred[kimi_sel] == focus).mean()) if kimi_sel.any() else None,
            other_mean_recall=float(np.mean(list(others.values()))) if others else None,
            gap=(
                float((sub_pred[kimi_sel] == focus).mean() - np.mean(list(others.values())))
                if kimi_sel.any() and others
                else None
            ),
        )

    overall_bins = pd.Series(bins).value_counts(normalize=True).to_dict()
    kimi_bins = pd.Series(bins[true == focus]).value_counts(normalize=True).to_dict() if (true == focus).any() else {}

    def _standardise(weights: Mapping[str, float]) -> float | None:
        vals, ws = [], []
        for b, w in weights.items():
            rec = bin_stats.get(b, {}).get("kimi_recall")
            if rec is not None and w > 0:
                vals.append(rec)
                ws.append(w)
        if not vals:
            return None
        ws = np.array(ws, dtype=float)
        return float(np.dot(np.array(vals), ws / ws.sum()))

    uniform = {b: 1.0 for b in bin_stats}
    return {
        "kimi_rows": int((true == focus).sum()),
        "bins": bin_stats,
        "bin_distribution_kimi": {str(k): float(v) for k, v in kimi_bins.items()},
        "bin_distribution_all": {str(k): float(v) for k, v in overall_bins.items()},
        "kimi_recall_standardised_to_all": _standardise(overall_bins),
        "kimi_recall_standardised_uniform": _standardise(uniform),
        "kimi_recall_raw": float((pred[true == focus] == focus).mean()) if (true == focus).any() else None,
        "mean_gap_within_bin": float(
            np.mean([v["gap"] for v in bin_stats.values() if v["gap"] is not None])
        ) if any(v["gap"] is not None for v in bin_stats.values()) else None,
        "bins_with_negative_gap": int(sum(1 for v in bin_stats.values() if (v["gap"] or 0) < 0)),
        "bins_evaluated": len(bin_stats),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def run_all(
    external_path: str | Path,
    official_path: str | Path,
    oof_dir: str | Path,
    *,
    label: str = DEFAULT_LABEL,
    config: DiagnosticsConfig | None = None,
    artifact_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every diagnostic and return one JSON-ready dict."""
    cfg = config or DiagnosticsConfig(label=label)
    cfg.label = label
    started = time.time()
    paths = {
        "external": Path(external_path),
        "official": Path(official_path),
        "oof_dir": Path(oof_dir),
        "folds": sorted(p.name for p in Path(oof_dir).glob("fold*") if p.is_dir()),
    }
    log(f"[load] pool={paths['external']}")
    log(f"[load] official={paths['official']}")
    log(f"[load] oof={paths['oof_dir']} folds={paths['folds']}")
    tables = load_tables(paths)
    pool, official, oof = tables["pool"], tables["official"], tables["oof"]
    log(f"[load] rows: pool={len(pool):,} official={len(official):,} oof={len(oof):,}")
    for name, frame in (("pool", pool), ("official", official)):
        if label not in set(frame["model"]):
            raise ValueError(f"{label} 不在 {name} 的 model 列里；现有: {sorted(set(frame['model']))[:20]}")
    if label not in set(oof["true_model"].astype(str)):
        raise ValueError(f"OOF 概率的 true_model 里没有 {label}。")

    results: dict = {
        "meta": {
            "label": label,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "paths": {
                "external": str(paths["external"]),
                "official": str(paths["official"]),
                "oof_dir": str(paths["oof_dir"]),
                "folds": paths["folds"],
            },
            "rows": dict(pool=len(pool), official=len(official), oof=len(oof)),
            "official_prompts": int(official["prompt_id"].nunique()),
            "official_classes": sorted(official["model"].unique().tolist()),
            "config": asdict(cfg),
            "missing_columns": {
                "pool": pool.attrs.get("missing_columns", []),
                "official": official.attrs.get("missing_columns", []),
            },
        },
        "stage_seconds": {},
    }

    def stage(name: str, fn, *args, **kwargs):
        t0 = time.time()
        log(f"[{name}] start")
        value = fn(*args, **kwargs)
        elapsed = time.time() - t0
        results["stage_seconds"][name] = round(elapsed, 1)
        _log_duration(log, name, elapsed)
        return value

    results["length"] = stage("length", length_diagnostics, pool, official, label)
    results["composition"] = stage("composition", composition_diagnostics, pool, official, label)
    results["near_duplicate"] = stage("near_duplicate", near_duplicate_diagnostics, pool, label, cfg, log=log)
    results["coverage"] = stage("coverage", coverage_diagnostics, pool, official, label, cfg, log=log)
    sep = stage("separability", separability_diagnostics, pool, label, cfg, log=log)
    oof_frame, oof_scores = sep.pop("_oof")
    results["separability"] = sep

    elapsed_min = (time.time() - started) / 60.0
    if elapsed_min < cfg.budget_minutes:
        try:
            results["official_cv"] = stage("official_cv", official_domain_cv, official, label, cfg, log=log)
        except Exception as exc:
            results["official_cv"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        try:
            results["transfer"] = stage("transfer", transfer_diagnostics, pool, official, label, cfg, log=log)
            results["transfer"].pop("_scores", None)
        except Exception as exc:  # keep the rest of the report if transfer dies
            results["transfer"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    else:
        results["official_cv"] = {"skipped": f"budget: {elapsed_min:.1f} min > {cfg.budget_minutes} min"}
        results["transfer"] = {"skipped": f"budget: {elapsed_min:.1f} min > {cfg.budget_minutes} min"}

    results["confusion"] = stage("confusion", confusion_diagnostics, oof, label)
    results["length_conditioned"] = stage("length_conditioned", length_conditioned_recall, oof, label)

    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = ["id", "prompt_id", "model", "n_words", "word_bin"]
        keep = [c for c in keep if c in oof_frame.columns]
        scores_frame = oof_frame[keep].copy()
        scores_frame["p_kimi"] = oof_scores
        scores_frame.to_parquet(out_dir / "sep_oof_scores.parquet", index=False)
        results["meta"]["artifacts"] = [str(out_dir / "sep_oof_scores.parquet")]

    results["meta"]["runtime_seconds"] = round(time.time() - started, 1)
    log(f"[done] {results['meta']['runtime_seconds']:.1f}s total")
    return _jsonable(results)


# --------------------------------------------------------------------------- #
# readable summary
# --------------------------------------------------------------------------- #


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def summarize(results: Mapping) -> str:
    """Human-readable readout; every line is a number from ``results``."""
    r = results
    meta = r.get("meta", {})
    label = meta.get("label", DEFAULT_LABEL)
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"Direction C — {label} 跨族塌陷诊断")
    add("=" * 72)
    add(f"数据: pool={meta.get('rows', {}).get('pool')} 行, official={meta.get('rows', {}).get('official')} 行, "
        f"oof={meta.get('rows', {}).get('oof')} 行, 官方 prompt={meta.get('official_prompts')}")
    add(f"运行: {meta.get('runtime_seconds')}s, 各阶段: {r.get('stage_seconds')}")
    add("")

    # 1. sample + length
    length = r.get("length", {})
    comp = r.get("composition", {})
    add("【1】样本量与长度")
    add(f"  pool 中 {label} 占比 {_fmt(comp.get('kimi_share'))} "
        f"({comp.get('kimi_rows')} 行 / {comp.get('pool_rows')} 行), "
        f"覆盖 {comp.get('kimi_prompts_in_pool')} 个 prompt "
        f"({_fmt(comp.get('kimi_prompts_share_of_pool_prompts'))} of pool prompts)")
    for name in ("pool", "official"):
        block = length.get(name, {})
        if not block or block.get("empty"):
            continue
        k, rest = block.get("kimi", {}), block.get("rest", {})
        ks = block.get("ks_kimi_vs_rest", {})
        add(f"  {name}: n_words kimi 均值 {_fmt(k.get('mean'), 1)} (p10 {_fmt(k.get('p10'), 1)}, "
            f"p90 {_fmt(k.get('p90'), 1)}) vs 其余 {_fmt(rest.get('mean'), 1)} "
            f"(p10 {_fmt(rest.get('p10'), 1)}, p90 {_fmt(rest.get('p90'), 1)}); "
            f"KS D={_fmt(ks.get('D'))} p={_fmt(ks.get('p'))}")
        add(f"  {name} word_bin 卡方: {block.get('word_bin_chi2')}")
    closest = list(length.get("pool", {}).get("ks_kimi_vs_class", {}).items())[:3]
    if closest:
        add("  与 kimi 长度分布最接近的类: "
            + ", ".join(f"{c} (D={_fmt(v.get('D'))})" for c, v in closest))
    add("  读法: KS D 小 + word_bin 卡方不显著 → 长度不是区分 kimi 的抓手；"
        "若 kimi 显著更短，后续所有召回都要看长度分层。")
    add("")

    # 2. composition
    add("【2】来源组成 (pool)")
    for col, block in (comp.get("pool_columns") or {}).items():
        if not isinstance(block, dict) or block.get("missing"):
            continue
        top = list(block.get("kimi_top", {}).items())[:2]
        add(f"  {col}: TVD={_fmt(block.get('tvd'))}, "
            f"kimi 主要取值 " + ", ".join(f"{k}={_fmt(v.get('share'))}" for k, v in top))
    add("  读法: TVD 高 → kimi 行来自不同的采集来源/生成协议，"
        "可能带来与官方集不同的风格或质量。")
    add("")

    # 3. near duplicates
    nd = r.get("near_duplicate", {})
    add("【3】近重复")
    add(f"  kimi 精确重复率(全池)={_fmt(nd.get('exact_within_kimi', {}).get('duplicate_rate'))}, "
        f"最大簇={nd.get('exact_within_kimi', {}).get('max_cluster')}, "
        f"64 字符前缀重复率={_fmt(nd.get('prefix64_within_kimi', {}).get('duplicate_rate'))}")
    near = nd.get("near_kimi_vs_others", {})
    add(f"  kimi 行有跨类 cos>=0.8 近邻的比例={_fmt(near.get('share_with_cross_class_cos_0_8'))} "
        f"(cos>=0.6: {_fmt(near.get('share_with_cross_class_cos_0_6'))}), "
        f"top1 cos 中位数={_fmt(near.get('median_top1_cos'))}")
    jc = nd.get("jaccard_check", {})
    add(f"  Jaccard 校验: {jc.get('pairs')} 对, 中位 J={_fmt(jc.get('median_jaccard'))}, "
        f"J>0.8 比例={_fmt(jc.get('share_jaccard_gt_0_8'))}, "
        f"cos>=0.8 时 J>0.8 比例={_fmt(jc.get('share_jaccard_gt_0_8_when_cos_ge_0_8'))}, "
        f"cos-J 相关={_fmt(jc.get('cos_jaccard_corr'))}")
    top1_classes = list((near.get("top1_class_share") or {}).items())[:3]
    if top1_classes:
        add("  kimi 最近邻最常落在: " + ", ".join(f"{c} {_fmt(s)}" for c, s in top1_classes))
    add("  读法: 跨类近重复率低 → 不是同一批文本被贴了两个标签；"
        "近重复高则提示标签污染或多模型同源。")
    add("")

    # 4. coverage
    cov = r.get("coverage", {})
    add("【4】官方集在 pool 中的覆盖")
    add(f"  {label}: 精确同(=同文本)={_fmt(cov.get('focus_exact_same_class'))}, "
        f"cos>=0.8 近邻={_fmt(cov.get('focus_near_0_8'))}; "
        f"其余类平均 exact={_fmt(cov.get('other_mean_exact_same_class'))}, "
        f"near={_fmt(cov.get('other_mean_near_0_8'))}, 差距={_fmt(cov.get('near_0_8_gap_vs_other_mean'))}")
    pl = cov.get("prompt_level", {})
    add(f"  官方 {pl.get('prompts_total')} 个 prompt 中 {pl.get('prompts_with_any_match')} 个至少一类能在池中找到同 prompt 近邻; "
        f"每 prompt 匹配类数分布={pl.get('matched_classes_per_prompt')}")
    per_class = cov.get("per_class", {})
    if per_class:
        ranked = sorted(per_class.items(), key=lambda kv: -(kv[1].get("near_0_8") or -1))
        add("  各类覆盖率 (near>=0.8): " + ", ".join(f"{c}={_fmt(v.get('near_0_8'))}" for c, v in ranked))
    add("  读法: kimi 覆盖率显著低于其余类 → 训练数据不足/采样偏移；"
        "覆盖率相当却仍然跨族错 → 决策边界问题。")
    add("")

    # 5. separability
    sep = r.get("separability", {})
    oofm = sep.get("oof_kimi_vs_rest", {})
    sample = sep.get("sample", {})
    add("【5】可分离性 (pool 内, prompt 分组 5 折)")
    add(f"  样本 {sample.get('n')} 行 (kimi {sample.get('n_kimi')}, 其余 {sample.get('n_rest')}), "
        f"prompt {sample.get('prompts')} 个")
    add(f"  kimi-vs-rest: AUC={_fmt(oofm.get('auc'))}, BA@0.5={_fmt(oofm.get('balanced_accuracy'))}, "
        f"recall={_fmt(oofm.get('recall'))}, precision={_fmt(oofm.get('precision'))}, "
        f"AP={_fmt(oofm.get('average_precision'))}")
    add(f"  折间 AUC {sep.get('fold_auc')}, BA {sep.get('fold_ba')}")
    fam = sep.get("pairwise_auc_family", {})
    if fam:
        add("  与各族二分类 AUC: " + ", ".join(f"{k}={_fmt(v)}" for k, v in fam.items()))
    lo = sep.get("length_only", {})
    add(f"  仅长度基线: AUC(n_words)={_fmt(lo.get('auc_n_words'))}, "
        f"kimi 均值 {_fmt(lo.get('kimi_mean_n_words'), 1)} vs 其余 {_fmt(lo.get('rest_mean_n_words'), 1)}; "
        f"各 bin 内平均 BA={_fmt(lo.get('mean_within_bin_ba'))}")
    add("  读法: AUC 高 (>=0.9) 而官方迁移 recall 低 → 分布偏移；AUC 本身低 → 特征层面就无法区分，"
        "属数据/标签问题。")
    add("")

    # 6. transfer
    tr = r.get("transfer", {})
    ocv = r.get("official_cv", {})
    if tr.get("skipped"):
        add(f"【6】迁移: 跳过 ({tr['skipped']})")
    else:
        add("【6】pool → official 迁移 (只用外部池训练)")
        a = tr.get("at_0.5", {})
        b = tr.get("at_pool_prevalence_threshold", {})
        c = tr.get("at_official_quantile_threshold", {})
        add(f"  域内对照 (官方集内部 prompt 分组 CV): AUC={_fmt(ocv.get('auc'))}, "
            f"BA={_fmt(ocv.get('balanced_accuracy'))}, recall={_fmt(ocv.get('recall'))}, "
            f"precision={_fmt(ocv.get('precision'))}")
        add(f"  迁移后: AUC={_fmt(a.get('auc'))}, AP={_fmt(a.get('average_precision'))}")
        add(f"  0.5 阈值: recall={_fmt(a.get('recall'))}, precision={_fmt(a.get('precision'))}, "
            f"BA={_fmt(a.get('balanced_accuracy'))}")
        add(f"  池先验阈值 {_fmt(b.get('threshold'))}: recall={_fmt(b.get('recall'))}, precision={_fmt(b.get('precision'))}")
        add(f"  官方分位阈值 {_fmt(c.get('threshold'))}: recall={_fmt(c.get('recall'))}, precision={_fmt(c.get('precision'))}")
        add(f"  按真实类别的 kimi 分数均值: "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in (tr.get("mean_score_by_true_class") or {}).items()))
        add("  读法: 域内 AUC 高 + 迁移 AUC 低 → 分布偏移；两者都低 → 文本本身不可分（重叠/标签噪声）；"
            "迁移 AUC 高而 BA 差 → 只是阈值/先验问题。")
    add("")

    # 7. confusion
    cf = r.get("confusion", {})
    k = cf.get("kimi", {})
    add("【7】OOF 混淆结构 (plain_centered_gaussian)")
    add(f"  kimi 召回={_fmt(k.get('recall'))}, top-2 命中率={_fmt(k.get('kimi_in_top2'))}, "
        f"kimi 概率均值(真 kimi)={_fmt(k.get('mean_prob_kimi_on_kimi_rows'))}, "
        f"kimi 排名中位数={_fmt(k.get('median_rank_of_kimi'), 1)}")
    add(f"  kimi 真实行 top-1 落点: {k.get('top1_pred_counts')}")
    add(f"  kimi 真实行 top-1 家族: {k.get('top1_pred_family_counts')}")
    add(f"  kimi 真实行 top-2 落点: {k.get('top2_pred_counts')}")
    prec = cf.get("kimi_precision", {})
    add(f"  预测为 kimi 的行 {prec.get('predicted_kimi_rows')} 个, 其中真 kimi "
        f"{prec.get('true_kimi_among_predicted')} 个, precision={_fmt(prec.get('precision'))}; "
        f"这些行的真实类别={prec.get('true_class_of_predicted_kimi')}")
    add("  读法: 若 kimi 行 top-1 集中在单一族 → 与那一族的边界重叠；"
        "top-1/top-2 分散在多族 → 更像整体表征不足或分布外。")
    add("")

    # 8. length controlled
    lc = r.get("length_conditioned", {})
    add("【8】长度控制下的条件召回 (OOF)")
    add(f"  kimi 原始召回={_fmt(lc.get('kimi_recall_raw'))}, "
        f"按全体长度分布标准化={_fmt(lc.get('kimi_recall_standardised_to_all'))}, "
        f"均匀重加权={_fmt(lc.get('kimi_recall_standardised_uniform'))}")
    add(f"  各 bin 内 kimi 与其余类平均召回差: 均值={_fmt(lc.get('mean_gap_within_bin'))}, "
        f"负差距 bin 数={lc.get('bins_with_negative_gap')}/{lc.get('bins_evaluated')}")
    for b, v in (lc.get("bins") or {}).items():
        add(f"    {b}: n={v.get('n')} kimi={v.get('n_kimi')} "
            f"recall={_fmt(v.get('kimi_recall'))} 其余均值={_fmt(v.get('other_mean_recall'))}")
    add("  读法: 标准化后召回基本不变 → 不是长度效应；标准化后明显回升 → 长度/风格短文本拉低召回。")
    add("")
    add("结果 JSON 字段与以上小节一一对应；meta.stage_seconds 记录各阶段耗时。")
    return "\n".join(lines)


def write_results(results: Mapping, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(results), ensure_ascii=False, indent=1, allow_nan=False)
    path.write_text(payload, encoding="utf-8")
    return path
