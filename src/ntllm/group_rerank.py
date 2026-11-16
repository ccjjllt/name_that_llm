"""Cross-family hard-group reranking and single-label log-odds gates for the Q8 readout.

Why this module exists
----------------------
``family_rerank.conditional_rerank`` can only move a decision *inside* one provider
family, and only when the top-2 labels already share a family.  The 26,484-update
five-fold OOF confusion matrix says the largest remaining error blocks are not of that
shape: ``glm-5.3 <-> claude-fable-5`` (65 rows) and ``glm-5.3-flash <-> claude-fable-5``
(52 rows) cross a family boundary, and ``kimi-k3`` / ``qwen3.8-max`` are single-label
families whose in-family rerank can never fire.  This module generalises the reranker
from "family" to "any named set of labels" and adds an additive log-odds gate for
single-label classes.

The blend, and what each mass policy can do
------------------------------------------
For one group ``G`` (``|G| = k >= 2``) with ``q`` the group-restricted discriminator's
log-probability over ``G`` and ``s = sum_{j in G} P_j`` the input mass of the group::

    r    = log P[:, G] - log s              (baseline conditional inside G)
    b(w) = log P[:, G] + w * (q - r)
         = (1 - w) * log P[:, G] + w * q + w * log s        (same thing, expanded)

so the blend is a log-linear interpolation between the baseline *conditional*
distribution inside ``G`` and the discriminator, plus a group-mass term.  Two mass
policies follow from that identity:

``restore`` (default, ``allow_cross_group=False``)
    Subtract a per-row constant so the group keeps exactly ``s``.  ``w=0`` is the
    identity and ``w=1`` reduces to ``s * softmax(q)``.  A per-row constant cannot
    change the argmax *inside* ``G``, so the entire effect is the reordering of ``G``;
    the quantity the outside labels compete with (the group's total mass) is untouched.

``free`` (``allow_cross_group=True``)
    No constant.  Both ``r`` and ``q`` are normalised distributions, so by Hölder
    ``sum_j exp b_j = s * sum_j r_j^(1-w) q_j^w <= s`` for ``0 < w < 1``: the group mass
    *contracts* at intermediate ``w`` and returns to ``s`` at ``w=1``.  It never grows
    (``w > 1`` reverses the inequality, which is why the experiment grid stops at 1.0),
    so this variant alone cannot pull in a row whose top-1 sits outside ``G``.  ``w=1``
    makes both policies agree on the group block.

Neither policy can transfer mass across the group boundary, which is why the function
also accepts ``lift``: a per-row additive log-odds on the whole group.  Under
``restore`` a per-row constant is absorbed by the mass restoration (mathematically a
no-op for predictions), so ``lift`` only acts with ``allow_cross_group=True``, where it
is the missing ingredient -- a group-membership score such as a group-vs-rest logistic
regression, exactly analogous to ``singleton_gate`` below.  ``fit_membership_models``
trains those scores on the external pool.

Group-internal cross-family errors (e.g. ``glm-5.3 -> claude-fable-5``, both inside the
``{glm-5.3, glm-5.3-flash, claude-fable-5}`` triple) are reachable under either policy,
because both labels live inside the reranked block.  Errors whose top-1 is outside the
group need ``lift``.

The module is import-safe on a stripped image: it only needs numpy, sklearn and the
sibling ``family_rerank`` module.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

try:  # package import; the single-file Kaggle bundle already defines these above
    from .family_rerank import (
        _logreg, _logsumexp, _order, family_of, fit_restricted_models, make_bank)
except ImportError:  # pragma: no cover - only the concatenated kernel takes this path
    pass

# The error blocks of the five-fold OOF confusion matrix, largest first.  A label may
# belong to more than one group (glm-5.3 is in glm_pair *and* in the triple); a rerank
# call takes one group at a time, and multi-group calls defer a row to the first group
# that claims it, so the caller controls the overlap by the dict order it passes.
HARD_GROUPS = {
    "glm_claude_triple": ("glm-5.3", "glm-5.3-flash", "claude-fable-5"),
    "glm_pair": ("glm-5.3", "glm-5.3-flash"),
    "gemini_quad": ("gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                    "gemini-3.8-flash"),
    "gpt_quint": ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"),
    "claude_quint": ("claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
                     "claude-sonnet-5"),
    "kimi_gpt55": ("kimi-k3", "gpt-5.5"),
}

# Classes whose family has a single label: in-family reranking is structurally impossible.
SINGLETON_LABELS = ("kimi-k3", "qwen3.8-max")

TRIGGERS = ("top1_in_group", "top2_in_group", "group_mass")

# Tightest first: with several overlapping groups the first one to claim a row reranks it.
ALL_GROUPS_ORDER = ("glm_claude_triple", "kimi_gpt55", "glm_pair", "gemini_quad", "gpt_quint",
                    "claude_quint")


def hard_groups(label_set=None):
    """The named hard groups, optionally restricted to the labels a dataset actually has."""
    if label_set is None:
        return {name: tuple(group) for name, group in HARD_GROUPS.items()}
    known = {str(label) for label in label_set}
    return {name: tuple(label for label in group if label in known)
            for name, group in HARD_GROUPS.items()}


def _group_items(groups):
    """Normalise a group spec into ordered ``(name, labels)`` pairs."""
    if hasattr(groups, "items"):
        return [(str(name), [str(label) for label in group]) for name, group in groups.items()]
    return [("group", [str(label) for label in groups])]


def _present_columns(labels, group):
    index = {label: i for i, label in enumerate(labels)}
    keep = [label for label in group if label in index]
    return np.asarray([index[label] for label in keep], dtype=int), keep


def match_target(bank, group_labels):
    """The bank target that scores this exact label set, else the smallest superset.

    A superset match is re-sliced to the group's columns without renormalising, so its
    scores no longer sum to one over the group; the blend then loses the clean
    interpolation identity.  ``fit_group_models`` makes exact matches the normal case.
    """
    wanted = set(group_labels)
    exact = [target for target in bank["targets"].values() if set(target["labels"]) == wanted]
    if exact:
        return exact[0]
    supersets = [(len(target["labels"]), target) for target in bank["targets"].values()
                 if wanted <= set(target["labels"])]
    if not supersets:
        return None
    return min(supersets, key=lambda item: item[0])[1]


def trigger_mask(P, columns, trigger="top1_in_group", threshold=0.5):
    """Rows a group rerank would act on.

    ``top1_in_group``: the global argmax is inside the group.
    ``top2_in_group``: the argmax *or* the runner-up is inside the group, i.e. a superset
        of the previous mask; it is the only trigger that can reach a row whose top-1 is
        outside the group, which is why it matters with ``allow_cross_group`` + ``lift``.
    ``group_mass``: the group's summed probability is at least ``threshold`` *and* at
        least every individual outside label, i.e. the group owns the row as a block.
    """
    P = np.asarray(P, dtype=np.float64)
    columns = np.asarray(columns, dtype=int)
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown trigger {trigger!r}; expected one of {TRIGGERS}")
    if columns.size < 1:
        return np.zeros(P.shape[0], dtype=bool)
    order = _order(P)
    inside_top1 = np.isin(order[:, 0], columns)
    if trigger == "top1_in_group":
        return inside_top1
    if trigger == "top2_in_group":
        second = order[:, 1] if order.shape[1] > 1 else order[:, 0]
        return inside_top1 | np.isin(second, columns)
    mass = P[:, columns].sum(1)
    outside = np.ones(P.shape[1], dtype=bool)
    outside[columns] = False
    best_outside = P[:, outside].max(1) if outside.any() else np.zeros(P.shape[0])
    return (mass >= float(threshold)) & (mass >= best_outside)


def group_conditional_rerank(P, bank, groups, blend_w=1.0, trigger="top1_in_group", *,
                             labels=None, allow_cross_group=False, threshold=0.5,
                             keep_group=None, lift=None, min_labels=2):
    """Blend a group-restricted discriminator into P without touching the other labels.

    ``log P'[:, G] = log P[:, G] + blend_w * (logQ - log P|G)`` (+ row constant when the
    group mass is restored, + ``lift`` when given).  ``blend_w=0`` returns the input
    unchanged, byte for byte.

    ``groups`` is either ``{name: labels}`` or a single iterable of labels.  With several
    groups they are applied in iteration order and a row is reranked by the first group
    that triggers on it; the groups' column blocks are written independently, so a later
    group never re-derives columns an earlier one already wrote for the same row.  A row
    that a group reverts is released, so a later group may still claim it.

    ``keep_group`` (default ``True`` unless ``allow_cross_group``) reverts a row whose
    decision would leave the group, which keeps every out-of-group decision identical to
    the input; the count is reported as ``reverted``.  With ``allow_cross_group=True``
    the same rows are only counted (``left_group``) so the free-blend diagnostics stay
    visible.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("P must be a 2-D (rows, labels) array")
    labels = [str(label) for label in (bank["labels"] if labels is None else labels)]
    logp = np.log(np.maximum(P, 1e-12))
    if keep_group is None:
        keep_group = not allow_cross_group
    info = dict(trigger=str(trigger), blend_w=float(blend_w),
                mass_mode="free" if allow_cross_group else "restore",
                rows=int(P.shape[0]), triggered=0, trigger_rate=None, applied=0, changed=0,
                reverted=0, left_group=0, deferred=0, group_mass_shift=None, per_group={})
    if lift is not None:
        lift = np.asarray(lift, dtype=np.float64).ravel()
        if lift.size != P.shape[0]:
            raise ValueError(f"lift must have {P.shape[0]} rows, got {lift.size}")
        info["lift_mean"] = float(lift.mean())
    if float(blend_w) == 0.0:
        return logp, info
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown trigger {trigger!r}; expected one of {TRIGGERS}")

    out = logp.copy()
    claimed = np.zeros(P.shape[0], dtype=bool)
    triggered = np.zeros(P.shape[0], dtype=bool)
    applied = np.zeros(P.shape[0], dtype=bool)
    mass_shift = 0.0
    for name, group in _group_items(groups):
        columns, group_labels = _present_columns(labels, group)
        entry = dict(labels=group_labels, triggered=0, applied=0, changed=0, reverted=0,
                     left_group=0, deferred=0)
        info["per_group"][name] = entry
        if len(columns) < max(2, int(min_labels)):
            entry["skipped"] = f"labels<{max(2, int(min_labels))}"
            continue
        target = match_target(bank, group_labels)
        if target is None:
            entry["skipped"] = "no_discriminator"
            continue
        logq = np.asarray(target["logq"], dtype=np.float64)[
            :, [target["labels"].index(label) for label in group_labels]]
        mask = trigger_mask(P, columns, trigger=trigger, threshold=threshold)
        triggered |= mask
        entry["triggered"] = int(mask.sum())
        rows = np.flatnonzero(mask & ~claimed)
        entry["deferred"] = int(mask.sum() - rows.size)
        if not rows.size:
            continue
        block = P[np.ix_(rows, columns)]
        total = np.maximum(block.sum(1, keepdims=True), 1e-12)
        restricted = np.log(np.maximum(block / total, 1e-12))
        blended = np.log(np.maximum(block, 1e-12)) + float(blend_w) * (logq[rows] - restricted)
        if lift is not None:
            blended = blended + lift[rows][:, None]
        if not allow_cross_group:
            # Row-wise constant: the group keeps exactly its input mass and, because a
            # constant cannot reorder a row, the group's argmax is set by `blended` alone.
            blended = blended + (np.log(total.ravel()) - _logsumexp(blended))[:, None]
        out[np.ix_(rows, columns)] = blended
        claimed[rows] = True
        applied[rows] = True
        entry["applied"] = int(rows.size)
        if float(blend_w) <= 10.0:
            before = block.sum(1)
            after = np.exp(out[np.ix_(rows, columns)]).sum(1)
            mass_shift = max(mass_shift, float(np.abs(before - after).max()))
        moved = out[rows].argmax(1)
        outside = ~np.isin(moved, columns)
        entry["left_group"] = int(outside.sum())
        info["left_group"] += int(outside.sum())
        if keep_group and outside.any():
            back = rows[outside]
            out[np.ix_(back, columns)] = logp[np.ix_(back, columns)]
            claimed[back] = False
            applied[back] = False
            entry["reverted"] = int(back.size)
            entry["applied"] = int(rows.size - back.size)
            entry["left_group"] = 0
        entry["changed"] = int((out[rows].argmax(1) != logp[rows].argmax(1)).sum())

    info["triggered"] = int(triggered.sum())
    info["trigger_rate"] = float(triggered.mean()) if triggered.size else None
    info["applied"] = int(applied.sum())
    info["reverted"] = int(sum(entry.get("reverted", 0) for entry in info["per_group"].values()))
    info["deferred"] = int(sum(entry.get("deferred", 0) for entry in info["per_group"].values()))
    info["changed"] = int((out.argmax(1) != logp.argmax(1)).sum())
    info["group_mass_shift"] = mass_shift if applied.any() else None
    if len(info["per_group"]) <= 1:
        info.pop("per_group")
    return out, info


def singleton_gate(P, label, scores, blend_w=0.0, *, labels=None, prior=0.5,
                   normalize=True, clip=1e-6):
    """Additive log-odds gate for a single-label class.

    ``log P'[label] += blend_w * (logit(score) - logit(prior))``; every other label keeps
    its value, so the gate moves only the threshold of this one class.  ``prior`` is the
    prevalence the score is calibrated to: 0.5 for a ``class_weight="balanced"`` fit, or
    the training prevalence to additionally re-calibrate toward a different base rate.
    ``blend_w=0`` returns the input unchanged.
    """
    P = np.asarray(P, dtype=np.float64)
    labels = [str(value) for value in (range(P.shape[1]) if labels is None else labels)]
    label = str(label)
    if label not in labels:
        raise ValueError(f"label {label!r} not in the probability columns")
    column = labels.index(label)
    logp = np.log(np.maximum(P, 1e-12))
    score = np.clip(np.asarray(scores, dtype=np.float64).ravel(), clip, 1.0 - clip)
    prior = float(np.clip(float(prior), clip, 1.0 - clip))
    info = dict(label=label, blend_w=float(blend_w), prior=prior, rows=int(P.shape[0]),
                applied=0, changed=0, flipped_in=0, flipped_out=0, delta_mean=None,
                delta_min=None, delta_max=None, trigger_rate=None)
    if float(blend_w) == 0.0:
        return logp, info
    if score.size != P.shape[0]:
        raise ValueError(f"scores must have {P.shape[0]} rows, got {score.size}")
    delta = float(blend_w) * (np.log(score / (1.0 - score)) - np.log(prior / (1.0 - prior)))
    out = logp.copy()
    out[:, column] += delta
    if normalize:
        out -= _logsumexp(out)[:, None]
    before = logp.argmax(1)
    after = out.argmax(1)
    info.update(applied=int(P.shape[0]), changed=int((after != before).sum()),
                flipped_in=int(((after == column) & (before != column)).sum()),
                flipped_out=int(((after != column) & (before == column)).sum()),
                delta_mean=float(delta.mean()), delta_min=float(delta.min()),
                delta_max=float(delta.max()),
                trigger_rate=float((score >= prior).mean()) if score.size else None)
    return out, info


def fit_group_models(X, labels, named_groups, *, kind="shared", min_labels=2, **kwargs):
    """One multinomial LR per *distinct* label set, aliased to every name that uses it.

    The family fit of ``claude`` and the hard group ``claude_quint`` are the same label
    set, so fitting them twice would only burn CPU.  ``alias`` maps each name to the key
    of the shared fit; ``bank_from`` turns that into a bank with one target per name.
    """
    fits, alias = {}, {}
    for name, group in named_groups.items():
        group = [str(label) for label in group]
        if len(group) < int(min_labels):
            continue
        key = "|".join(sorted(group))
        fits.setdefault(key, tuple(group))
        alias[str(name)] = key
    models = fit_restricted_models(X, labels, fits, kind=kind, **kwargs) if fits else {}
    return dict(models=models, alias=alias, fits=fits)


def bank_from(shared, X_query, labels, kinds=None):
    """Score every aliased group on the query rows once, keeping one target per name."""
    kinds = {} if kinds is None else dict(kinds)
    aliased = {}
    for name, key in shared["alias"].items():
        model = shared["models"].get(key)
        if model is None:
            continue
        aliased[name] = dict(model, kind=kinds.get(name, model.get("kind", "shared")))
    return make_bank(aliased, X_query, labels)


def fit_membership_models(X, labels, targets, *, C=1.0, max_iter=2000, min_rows=20,
                          kind="membership"):
    """Binary ``inside target vs rest`` logistic regressions (groups or single labels).

    These are the group-level scores the within-group discriminator cannot provide: a
    multinomial fit over a label subset is blind to "should this row be in the group at
    all", which is exactly the decision that separates ``glm-5.3`` from ``kimi-k3``.
    """
    labels = np.asarray([str(value) for value in labels])
    out = {}
    for name, group in _group_items(targets):
        inside = np.isin(labels, group)
        n_pos, n_neg = int(inside.sum()), int(len(labels) - inside.sum())
        record = dict(name=name, kind=kind, labels=group, positives=n_pos, negatives=n_neg,
                      prevalence=n_pos / max(1, len(labels)), classifier=None,
                      skipped=None, train_auc=None)
        if n_pos < int(min_rows) or n_neg < int(min_rows):
            record["skipped"] = f"class<{int(min_rows)}_rows"
        else:
            binary = inside.astype(np.int64)
            classifier = _logreg(max_iter=max_iter, C=C)
            classifier.fit(X, binary)
            if not np.array_equal(np.asarray(classifier.classes_, dtype=np.int64),
                                  np.array([0, 1])):
                record["skipped"] = "class_order"
            else:
                record["classifier"] = classifier
                record["train_auc"] = float(roc_auc_score(binary,
                                                          classifier.predict_proba(X)[:, 1]))
        out[name] = record
    return out


def membership_scores(models, X):
    """``{name: P(inside)}`` for every membership model that was actually fitted."""
    out = {}
    for name, record in models.items():
        if record.get("classifier") is None:
            continue
        out[name] = np.asarray(record["classifier"].predict_proba(X)[:, 1], dtype=np.float64)
    return out


def group_error_report(y, pred, labels=None, groups=None):
    """Rows whose truth *and* prediction sit inside one group, and the BA ceiling they cap.

    ``gain_if_fixed`` is the pooled balanced-accuracy gain obtained by fixing every
    group-internal error, i.e. ``sum 1 / (n_classes * n_truth)`` over those rows; it is
    the ceiling a perfect within-group discriminator could reach, not a prediction.
    """
    truth = np.asarray([str(value) for value in y], dtype=object)
    guess = np.asarray([str(value) for value in pred], dtype=object)
    names = sorted(set(truth.tolist())) if labels is None else [str(l) for l in labels]
    counts = {label: int((truth == label).sum()) for label in names}
    n_classes = max(1, len([label for label, count in counts.items() if count]))
    groups = HARD_GROUPS if groups is None else groups
    out = {}
    for name, group in groups.items():
        inside = set(group)
        in_group = np.array([a in inside and b in inside for a, b in zip(truth, guess)])
        wrong = in_group & (truth != guess)
        pairs, gain = {}, 0.0
        for t, p in zip(truth[wrong], guess[wrong]):
            pairs[f"{t} -> {p}"] = pairs.get(f"{t} -> {p}", 0) + 1
            gain += 1.0 / (n_classes * max(1, counts.get(t, 0)))
        out[name] = dict(
            labels=list(group), rows_in_group=int(in_group.sum()), wrong=int(wrong.sum()),
            cross_family_wrong=int(sum(1 for t, p in zip(truth[wrong], guess[wrong])
                                       if family_of(t) != family_of(p))),
            gain_if_fixed=float(gain), gain_if_fixed_pp=100.0 * float(gain),
            pairs=dict(sorted(pairs.items(), key=lambda item: -item[1])[:12]))
    return out
