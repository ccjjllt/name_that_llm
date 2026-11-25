"""Reconstruct the frozen Sibling Odds front end and add Comp at its original slot."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp


COMP_WEIGHT = .15
GSQ_WEIGHT = .03
GSQ_MIN_AGREEMENT = .30
NATIVE_WEIGHT = .40
NATIVE_MIN_AGREEMENT = .35
NATIVE_PREFIX = 'native__fold_'
# Captured member arrays are float32, while the archived stack is float64.
REPLAY_TOLERANCE = 5e-5


def rowsoftmax(values):
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def decide_scores(probabilities, counts):
    refitted = probabilities.copy()
    marginal = np.clip(probabilities.mean(axis=0), 1e-6, None)
    marginal /= marginal.sum()
    prior = counts.astype(np.float64) + 1.
    prior /= prior.sum()
    target = np.clip(.65 * marginal + .35 * prior, 1e-6, None)
    target /= target.sum()
    for _ in range(100):
        refitted *= (len(refitted) * target[None, :]) / np.clip(refitted.sum(axis=0, keepdims=True), 1e-12, None)
        refitted /= np.clip(refitted.sum(axis=1, keepdims=True), 1e-12, None)
    return .70 * np.log(probabilities + 1e-12) + .30 * np.log(refitted + 1e-12)


def read_comp(path, ids, labels):
    receipt = json.loads((path / 'report.json').read_text())
    if not receipt['complete'] or receipt['returncode'] != 0:
        raise ValueError('Only a completed checkpoint-repaired Comp can be replayed')
    with np.load(path / 'scores.npz', allow_pickle=False) as archive:
        keys, classes, logits = (archive[key] for key in ('ids', 'labels', 'raw_logits'))
    order = pd.Index(keys).get_indexer(ids)
    columns = pd.Index(classes).get_indexer(labels)
    digest = hashlib.sha256(('\n'.join(keys) + '\n').encode()).hexdigest()
    if (len(keys) != receipt['rows'] or digest != receipt['ordered_ids_sha256']
            or len(classes) != 18 or logits.shape != (len(keys), len(classes))
            or min(order) < 0 or min(columns) < 0 or not np.isfinite(logits).all()):
        raise ValueError('Comp scores cannot be aligned to the completed diagnostic receipt')
    return logits[np.ix_(order, columns)], receipt


def _matches(actual, reference, tolerance):
    actual = actual - logsumexp(actual, axis=1, keepdims=True)
    reference = reference - logsumexp(reference, axis=1, keepdims=True)
    difference = np.abs(actual - reference)
    same = actual.argmax(1) == reference.argmax(1)
    return dict(same_predictions=int(same.sum()), rows=len(same),
                max_abs_logp_delta=float(difference.max()), mean_abs_logp_delta=float(difference.mean()),
                matches=bool(same.all() and difference.max() <= tolerance))


def _later_stages(probabilities, counts, members, masks, folds, *,
                  allow_gsq, allow_native, enforce_agreement, gsq_weight=GSQ_WEIGHT):
    report = dict(gsq_included=False, native_included=False)
    if allow_gsq:
        gsq = rowsoftmax(members['infer__gsq'].astype(np.float64))
        agreement = float((gsq.argmax(1) == probabilities.argmax(1)).mean())
        report['gsq_agreement'] = agreement
        if not enforce_agreement or agreement >= GSQ_MIN_AGREEMENT:
            probabilities = (1. - gsq_weight) * probabilities + gsq_weight * gsq
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            report['gsq_included'] = True
    if not allow_native:
        return decide_scores(probabilities, counts), report

    finite = masks[folds[0]].all(1)
    if any(not np.array_equal(masks[key], np.broadcast_to(finite[:, None], probabilities.shape)) for key in folds):
        raise ValueError('The original Native worker used one common finite mask across its completed folds')
    matches = [(members[key].argmax(1) == probabilities.argmax(1)) & finite for key in folds]
    lower = min(float(match.mean()) for match in matches)
    upper = lower + float((~finite).mean())
    report.update(native_folds=folds, native_rows=int(finite.sum()),
                  native_min_agreement_bounds=[lower, upper])
    if enforce_agreement and upper < NATIVE_MIN_AGREEMENT:
        return decide_scores(probabilities, counts), report
    if enforce_agreement and lower < NATIVE_MIN_AGREEMENT <= upper:
        report['native_agreement_uncertain'] = True
    logp = np.log(np.maximum(probabilities, 1e-15))
    share = (NATIVE_WEIGHT * finite)[:, None]
    mixture = np.zeros_like(probabilities)
    for key in folds:
        native = rowsoftmax((1. - share) * logp + share * members[key])
        mixture += rowsoftmax(decide_scores(native, counts)) / len(folds)
    report['native_included'] = True
    return np.log(np.maximum(mixture, 1e-15)), report


def _original_frontend(trace, members, masks, kinds, train_counts, tolerance):
    reference = trace['stack_logp']
    shape = reference.shape
    counts = np.asarray(train_counts)
    if counts.shape != (shape[1],) or np.any(counts <= 0) or not np.isfinite(counts).all() or tolerance <= 0:
        raise ValueError('Training counts or replay tolerance are invalid')
    if 'infer__comp_blend' in members:
        raise ValueError('This diagnostic requires a historical trace with Comp absent')
    terms = [key for key, array in members.items() if key.startswith('infer__')
             and kinds[key]['kind'] == 'weighted_logp' and np.any(array != 0.)]
    if not terms:
        raise ValueError('No weighted pre-blend terms were captured')
    if any(np.any(members[key][~masks[key]] != 0.) for key in terms):
        raise ValueError('An unavailable weighted term contains a nonzero contribution')
    weighted = sum(members[key].astype(np.float64) for key in terms) / len(terms)
    base = rowsoftmax(weighted)
    gsq = 'infer__gsq' in members
    if gsq and not masks['infer__gsq'].all():
        raise ValueError('Original accepted GSQ must cover the full batch')
    folds = sorted([key for key in members if key.startswith(NATIVE_PREFIX)],
                   key=lambda key: int(key.removeprefix(NATIVE_PREFIX)))
    restored, original_delivery = _later_stages(base, counts, members, masks, folds,
                                               allow_gsq=gsq, allow_native=bool(folds), enforce_agreement=False)
    identity = _matches(restored, reference, tolerance)
    assumed_native = bool(folds)
    if folds and not identity['matches']:
        other, delivery = _later_stages(base, counts, members, masks, folds,
                                       allow_gsq=gsq, allow_native=False, enforce_agreement=False)
        no_native = _matches(other, reference, tolerance)
        if no_native['matches']:
            restored, original_delivery, identity, assumed_native = other, delivery, no_native, False
    if not identity['matches'] or not np.array_equal(trace['labels'][restored.argmax(1)], trace['stack']):
        raise ValueError(f'Comp-free front end does not reproduce the archived word-level stack: {identity}')
    audit = dict(identity=identity, matched_tolerance=tolerance, active_weighted_terms=terms,
                 original=original_delivery, native_captured_but_not_used=bool(folds) and not assumed_native)
    return base, counts, gsq, folds, assumed_native, restored, audit


def replay_frontend(trace, members, masks, kinds, train_counts, comp_logits, *, tolerance=REPLAY_TOLERANCE):
    """Return the archived reconstruction and restored Comp counterfactual."""
    if comp_logits.shape != trace['stack_logp'].shape or not np.isfinite(comp_logits).all():
        raise ValueError('Repaired Comp scores are invalid')
    base, counts, gsq, folds, assumed_native, restored, report = _original_frontend(
        trace, members, masks, kinds, train_counts, tolerance)

    comp_probabilities = rowsoftmax(comp_logits)
    blended = (1. - COMP_WEIGHT) * base + COMP_WEIGHT * comp_probabilities
    blended /= blended.sum(axis=1, keepdims=True)
    candidate, delivery = _later_stages(blended, counts, members, masks, folds, allow_gsq=gsq,
                                      allow_native=assumed_native, enforce_agreement=True)
    report.update(repaired=delivery, comp_weight=COMP_WEIGHT,
                  limitation='Only captured optional GSQ and Native branches can be evaluated')
    return restored, candidate, report


def replay_gsq_weights(trace, members, masks, kinds, train_counts, weights, *, tolerance=REPLAY_TOLERANCE):
    """Change GSQ blending while preserving the observed Native delivery."""
    base, counts, gsq, folds, assumed_native, restored, audit = _original_frontend(
        trace, members, masks, kinds, train_counts, tolerance)
    if not gsq:
        raise ValueError('Expected the observed full-batch GSQ scores')
    reference = trace['stack_logp']
    restored_logp = restored - logsumexp(restored, axis=1, keepdims=True)
    results = {}
    for weight in weights:
        if not np.isfinite(weight) or not 0. <= weight <= 1.:
            raise ValueError('GSQ weights must be within [0, 1]')
        if weight == GSQ_WEIGHT:
            results[weight] = reference.copy()
        else:
            alternative, _ = _later_stages(base, counts, members, masks, folds,
                                           allow_gsq=True, allow_native=assumed_native,
                                           enforce_agreement=False, gsq_weight=weight)
            alternative_logp = alternative - logsumexp(alternative, axis=1, keepdims=True)
            results[weight] = reference + (alternative_logp - restored_logp)
    return results, audit
