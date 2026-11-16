"""Family-restricted discriminators and conditional reranking for Q8 readouts.

CPU-only helpers: two independent text feature spaces (character 3-5 gram TF-IDF and
word 1-2 gram TF-IDF reduced by TruncatedSVD), one multinomial logistic regression per
family (or per hard label group), and a blender that mixes the restricted log-probability
into an existing 18-class posterior without touching family-level decisions.

The blend is mass preserving inside the reranked label subset, so each family's total mass is
unchanged; rows whose decision would leave the triggered family are reverted to the input
posterior, which keeps every inter-family decision identical to the input.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FAMILIES = {
    "claude": ("claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
               "claude-sonnet-5"),
    "gemini": ("gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
               "gemini-3.8-flash"),
    "glm": ("glm-5.3", "glm-5.3-flash"),
    "gpt": ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"),
    "kimi": ("kimi-k3",),
    "qwen": ("qwen3.8-max",),
}

HARD_GROUPS = {
    "glm_pair": ("glm-5.3", "glm-5.3-flash"),
    "gemini_37_38": ("gemini-3.7-flash", "gemini-3.8-flash"),
    "gpt_56": ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"),
}

LABELS = tuple(label for family in FAMILIES.values() for label in family)
LABEL_FAMILY = {label: family for family, labels in FAMILIES.items() for label in labels}
HARD_OF_LABEL = {label: name for name, labels in HARD_GROUPS.items() for label in labels}


def family_of(label):
    """Family of a model label; unknown labels fall back to their dash prefix."""
    return LABEL_FAMILY.get(str(label), str(label).split("-")[0])


def _texts(texts):
    return ["" if text is None else str(text) for text in texts]


class CharNgramSpace:
    """char_wb 3-5 gram sublinear TF-IDF."""

    name = "char_wb"

    def __init__(self, min_df=3, max_features=300_000, ngram_range=(3, 5), dtype=np.float32):
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                                          min_df=min_df, max_features=max_features,
                                          sublinear_tf=True, dtype=dtype)

    def fit(self, texts):
        self.vectorizer.fit(_texts(texts))
        return self

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(_texts(texts))

    def transform(self, texts):
        return self.vectorizer.transform(_texts(texts))

    def describe(self):
        vocabulary = getattr(self.vectorizer, "vocabulary_", None)
        return dict(kind=self.name, analyzer="char_wb", ngram_range=[3, 5], min_df=self.min_df,
                    max_features=self.vectorizer.max_features,
                    vocabulary_size=len(vocabulary) if vocabulary is not None else 0)


class WordSvdSpace:
    """word 1-2 gram sublinear TF-IDF -> TruncatedSVD(256) -> standardized dense."""

    name = "word_svd"

    def __init__(self, n_components=256, min_df=3, max_features=200_000, seed=0,
                 dtype=np.float32):
        self.seed, self.dtype, self.n_components = int(seed), dtype, int(n_components)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                          max_features=max_features, sublinear_tf=True,
                                          dtype=dtype)
        self.svd, self.scaler = None, StandardScaler()

    def fit(self, texts):
        self.fit_transform(texts)
        return self

    def fit_transform(self, texts):
        sparse = self.vectorizer.fit_transform(_texts(texts))
        components = max(2, min(self.n_components, sparse.shape[1] - 1, sparse.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed, n_iter=7)
        self.scaler = StandardScaler()
        dense = self.svd.fit_transform(sparse)
        return self.scaler.fit_transform(dense).astype(self.dtype, copy=False)

    def transform(self, texts):
        dense = self.svd.transform(self.vectorizer.transform(_texts(texts)))
        return self.scaler.transform(dense).astype(self.dtype, copy=False)

    def describe(self):
        vocabulary = getattr(self.vectorizer, "vocabulary_", None)
        return dict(kind=self.name, analyzer="word", ngram_range=[1, 2], min_df=3,
                    max_features=self.vectorizer.max_features,
                    vocabulary_size=len(vocabulary) if vocabulary is not None else 0,
                    n_components=None if self.svd is None else int(self.svd.n_components),
                    explained_variance=None if self.svd is None
                    else float(self.svd.explained_variance_ratio_.sum()))


def make_space(kind, seed=0):
    """Build an unfitted feature space by name."""
    if kind == "char_wb":
        return CharNgramSpace()
    if kind == "word_svd":
        return WordSvdSpace(seed=seed)
    raise ValueError(f"unknown feature space {kind!r}; expected 'char_wb' or 'word_svd'")


def _logreg(max_iter=2000, C=1.0):
    """The published family classifier; n_jobs is optional across sklearn versions."""
    try:
        return LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced", n_jobs=-1)
    except TypeError:
        return LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")


def fit_restricted_models(X, labels, groups, *, kind="family", C=1.0, max_iter=2000,
                          min_rows=40):
    """Train one multinomial LR per label group; groups with <2 labels are recorded, not fit."""
    labels = np.asarray([str(label) for label in labels])
    present = set(labels.tolist())
    models = {}
    for name, group in groups.items():
        group_labels = tuple(label for label in group if label in present)
        mask = np.isin(labels, group_labels)
        counts = {label: int(np.sum(labels == label)) for label in group_labels}
        record = dict(name=name, kind=kind, labels=group_labels, n_rows=int(mask.sum()),
                      counts=counts, classifier=None, skipped=None, train_accuracy=None)
        if len(group_labels) < 2:
            record["skipped"] = "single_label"
        elif int(mask.sum()) < min_rows:
            record["skipped"] = f"rows<{min_rows}"
        elif min(counts.values()) < 2:
            record["skipped"] = "label_with_<2_rows"
        else:
            order = {label: i for i, label in enumerate(group_labels)}
            target = np.array([order[label] for label in labels[mask]], dtype=np.int64)
            classifier = _logreg(max_iter=max_iter, C=C)
            classifier.fit(X[mask], target)
            if not np.array_equal(np.asarray(classifier.classes_, dtype=np.int64),
                                  np.arange(len(group_labels))):
                record["skipped"] = "class_order_mismatch"
            else:
                record.update(classifier=classifier,
                              train_accuracy=float(np.mean(classifier.predict(X[mask]) == target)))
        models[name] = record
    return models


def target_log_proba(model, X):
    """Log-probability over the model's own label order; a single-label model is log 1.0."""
    if model.get("classifier") is None:
        return np.zeros((X.shape[0], len(model["labels"])), dtype=np.float64)
    return np.asarray(model["classifier"].predict_log_proba(X), dtype=np.float64)


def make_bank(models, X_query, labels):
    """Pre-score every restricted model on the query rows once, in the query label order."""
    labels = [str(label) for label in labels]
    known = set(labels)
    targets = {}
    for name, model in models.items():
        if model.get("classifier") is None:
            continue
        keep = [i for i, label in enumerate(model["labels"]) if label in known]
        if len(keep) < 2:
            continue
        targets[name] = dict(name=name, kind=model["kind"],
                             labels=[model["labels"][i] for i in keep],
                             logq=target_log_proba(model, X_query)[:, keep])
    return dict(labels=labels, targets=targets,
                groups={name: list(target["labels"]) for name, target in targets.items()})


def _logsumexp(a):
    peak = a.max(1, keepdims=True)
    return (peak + np.log(np.exp(a - peak).sum(1, keepdims=True))).ravel()


def _order(P):
    return np.argsort(-np.asarray(P, dtype=np.float64), axis=1, kind="stable")


def same_family_top_mask(P, labels, topk=2):
    """Rows whose best `topk` labels all belong to one family."""
    labels = list(labels)
    order = _order(P)
    k = max(1, min(int(topk), len(labels)))
    mask = np.ones(len(order), dtype=bool)
    family = np.array([family_of(label) for label in labels], dtype=object)
    for column in range(1, k):
        mask &= family[order[:, 0]] == family[order[:, column]]
    return mask


def trigger_stats(P, y=None, labels=None, topk=2, bank=None):
    """How often the top-k labels agree on one family, and how often the decision is wrong."""
    if labels is None:
        if bank is None:
            raise ValueError("trigger_stats needs labels or a bank")
        labels = bank["labels"]
    labels = [str(label) for label in labels]
    P = np.asarray(P, dtype=np.float64)
    mask = same_family_top_mask(P, labels, topk=topk)
    order = _order(P)
    top1 = np.asarray(labels, dtype=object)[order[:, 0]]
    second = order[:, 1] if order.shape[1] > 1 else order[:, 0]
    hard = np.array([HARD_OF_LABEL.get(labels[a]) is not None
                     or HARD_OF_LABEL.get(labels[b]) is not None
                     for a, b in zip(order[:, 0], second)], dtype=bool)
    stats = dict(rows=int(len(P)), topk=int(topk), topk_same_family=int(mask.sum()),
                 topk_same_family_rate=float(mask.mean()) if len(P) else None,
                 top1_wrong=None, top1_wrong_same_family=None,
                 same_family_without_model=None,
                 hard_group_in_top2=int(hard.sum()),
                 hard_applicable=int((mask & hard).sum()))
    if y is not None:
        truth = np.asarray([str(label) for label in y], dtype=object)
        correct = top1 == truth
        stats["top1_wrong"] = int((~correct).sum())
        stats["top1_wrong_same_family"] = int((mask & ~correct).sum())
    if bank is not None:
        covered = np.array([family_of(label) in bank["targets"] for label in top1], dtype=bool)
        stats["same_family_without_model"] = int((mask & ~covered).sum())
    stats["by_family"] = {}
    family_top1 = np.array([family_of(label) for label in top1], dtype=object)
    for family in sorted(set(family_top1.tolist())):
        local = family_top1 == family
        entry = dict(rows=int(local.sum()), same_family=int((mask & local).sum()),
                     share_of_same_family=float((mask & local).sum() / max(1, mask.sum())))
        if y is not None:
            entry["top1_wrong_same_family"] = int((mask & ~correct & local).sum())
        stats["by_family"][family] = entry
    return stats


def _hit_rows(mask, order, labels, name, target, mode):
    """Absolute row indices where one target applies, given the same-family trigger."""
    rows = []
    for row in np.flatnonzero(mask):
        first, second = labels[order[row, 0]], labels[order[row, 1]]
        if mode == "family":
            hit = (family_of(target["labels"][0]) == family_of(first)
                   and family_of(second) == family_of(first))
        else:
            hit = name == HARD_OF_LABEL.get(first) or name == HARD_OF_LABEL.get(second)
        if hit:
            rows.append(row)
    return np.asarray(rows, dtype=int)


def conditional_rerank(P, family_model, topk=2, blend_w=1.0, mode="family", labels=None,
                       keep_family=True):
    """Blend family-restricted log-probability into P where the top-k labels share a family.

    ``log P' = log P + blend_w * (log Q_group - log P_group_restricted)`` with both terms
    normalized inside the reranked label subset, plus a constant that keeps the subset's
    input mass, so the family posterior is unchanged. ``blend_w=0`` returns the input
    unchanged and ``blend_w=1`` reduces exactly to the restricted model's decision inside
    the family.

    With ``keep_family=True`` (default) any row whose decision would move out of the
    triggered family is reverted to the input posterior, which keeps every inter-family
    decision identical to the baseline; the count is reported as ``reverted``.

    mode="family" reranks every triggered row inside its family, mode="hard" only reranks
    the configured hard groups (glm pair, gemini 3.7/3.8, gpt-5.6 triple).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("P must be a 2-D (rows, labels) array")
    labels = [str(label) for label in (family_model["labels"] if labels is None else labels)]
    index = {label: i for i, label in enumerate(labels)}
    logp = np.log(np.maximum(P, 1e-12))
    info = dict(mode=mode, topk=int(topk), blend_w=float(blend_w), rows=int(P.shape[0]),
                topk_same_family=0, applied=0, changed=0, reverted=0,
                family_mass_shift=None)
    if float(blend_w) == 0.0:
        return logp, info
    if mode not in ("family", "hard"):
        raise ValueError(f"unknown mode {mode!r}; expected 'family' or 'hard'")

    order = _order(P)
    mask = same_family_top_mask(P, labels, topk=topk)
    info["topk_same_family"] = int(mask.sum())
    out = logp.copy()
    applied = np.zeros(P.shape[0], dtype=bool)

    for name, target in family_model["targets"].items():
        if target["kind"] != mode:
            continue
        if not all(label in index for label in target["labels"]):
            continue
        columns = np.asarray([index[label] for label in target["labels"]], dtype=int)
        if len(columns) < 2:
            continue
        rows = _hit_rows(mask, order, labels, name, target, mode)
        if not rows.size:
            continue
        block = P[np.ix_(rows, columns)]
        total = np.maximum(block.sum(1, keepdims=True), 1e-12)
        restricted = np.log(np.maximum(block / total, 1e-12))
        blended = np.log(np.maximum(block, 1e-12)) + float(blend_w) * (target["logq"][rows]
                                                                     - restricted)
        # Row-wise constant so the reranked subset keeps exactly the input mass; a constant
        # shift inside the subset cannot change which label wins.
        shift = np.log(total.ravel()) - _logsumexp(blended)
        out[np.ix_(rows, columns)] = blended + shift[:, None]
        applied[rows] = True

    info["applied"] = int(applied.sum())
    if keep_family and applied.any():
        family_array = np.array([family_of(label) for label in labels], dtype=object)
        leave = applied & (family_array[out.argmax(1)] != family_array[logp.argmax(1)])
        if leave.any():
            out[leave] = logp[leave]
            info["reverted"] = int(leave.sum())
    info["changed"] = int((out.argmax(1) != logp.argmax(1)).sum())
    if float(blend_w) <= 10.0 and applied.any():
        families = sorted({family_of(label) for label in labels})
        indicator = np.array([[1.0 if family_of(label) == fam else 0.0 for fam in families]
                              for label in labels])
        before, after = P[applied] @ indicator, np.exp(out[applied]) @ indicator
        info["family_mass_shift"] = float(np.abs(before - after).max())
    return out, info


def replace_family_decisions(P, family_model, topk=2, mode="family", labels=None):
    """Hard override: triggered rows take the restricted model's argmax, everything else keeps P.

    Only rows whose current top-1 already lies inside the reranked group are overridden, which
    is exactly what ``blend_w=1`` does through the mass-preserving blend; the runner checks the
    two paths against each other (``matches_blend_one``).
    """
    P = np.asarray(P, dtype=np.float64)
    labels = [str(label) for label in (family_model["labels"] if labels is None else labels)]
    index = {label: i for i, label in enumerate(labels)}
    order = _order(P)
    mask = same_family_top_mask(P, labels, topk=topk)
    guess = np.asarray(labels, dtype=object)[order[:, 0]].copy()
    applied = np.zeros(P.shape[0], dtype=bool)
    for name, target in family_model["targets"].items():
        if target["kind"] != mode or not all(label in index for label in target["labels"]):
            continue
        rows = _hit_rows(mask, order, labels, name, target, mode)
        if rows.size:
            keep = [row for row in rows if labels[order[row, 0]] in target["labels"]]
            rows = np.asarray(keep, dtype=int)
        if rows.size:
            guess[rows] = np.asarray(target["labels"], dtype=object)[target["logq"][rows].argmax(1)]
            applied[rows] = True
    return guess, dict(mode=mode, topk=int(topk), rows=int(P.shape[0]),
                       topk_same_family=int(mask.sum()), applied=int(applied.sum()),
                       changed=int(np.sum(guess != np.asarray(labels, dtype=object)[order[:, 0]])))


def per_class_recall(y, pred, labels=None):
    """Recall per class over the rows whose truth is that class."""
    truth = np.asarray([str(value) for value in y], dtype=object)
    guess = np.asarray([str(value) for value in pred], dtype=object)
    names = sorted(set(truth.tolist())) if labels is None else [str(l) for l in labels]
    recalls = {}
    for label in names:
        mask = truth == label
        if not mask.any():
            continue
        recalls[label] = float((guess[mask] == label).mean())
    return recalls


def balanced_accuracy(y, pred, labels=None):
    """Macro recall over the classes present in y."""
    recalls = per_class_recall(y, pred, labels=labels)
    return float(np.mean(list(recalls.values()))) if recalls else float("nan")


def family_accuracy(y, pred):
    truth = [family_of(label) for label in y]
    guess = [family_of(label) for label in pred]
    return float(np.mean([a == b for a, b in zip(truth, guess)]))


def family_conditional_accuracy(y, pred):
    """Family accuracy plus accuracy restricted to rows where truth and guess share a family."""
    truth = np.asarray([str(value) for value in y], dtype=object)
    guess = np.asarray([str(value) for value in pred], dtype=object)
    same = np.array([family_of(a) == family_of(b) for a, b in zip(truth, guess)])
    return dict(family_accuracy=float(same.mean()),
                conditional_accuracy=float((truth[same] == guess[same]).mean())
                if same.any() else None,
                n_conditional=int(same.sum()))


def score_predictions(y, pred, labels=None):
    """The metric bundle reported for every rerank setting."""
    scores = family_conditional_accuracy(y, pred)
    return dict(balanced_accuracy=balanced_accuracy(y, pred, labels=labels),
                family_accuracy=scores["family_accuracy"],
                conditional_accuracy=scores["conditional_accuracy"],
                n_conditional=scores["n_conditional"],
                per_class_recall=per_class_recall(y, pred, labels=labels))


def predict_from_log_prob(log_prob, labels):
    labels = np.asarray([str(label) for label in labels], dtype=object)
    return labels[np.asarray(log_prob).argmax(1)]
