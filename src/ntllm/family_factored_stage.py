"""Fit small portable logarithmic pools over the three frozen score stages."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp


STAGES = ('stack_logp', 'glm_logp', 'sibling_logp')
FAMILIES = ('claude', 'gemini', 'glm', 'gpt', 'kimi', 'qwen')
REGULARIZATION = .05
FORMAT = 'ntl-stage-opinion-pool-v1'


def inputs(stage_logp, labels):
    labels = list(labels)
    if len(labels) != 18 or labels != sorted(set(labels)):
        raise ValueError('Expected the original sorted 18-class order')
    # Same provider-family map as sibling_specialist.family.
    families = ['qwen' if name.startswith('qwen') else name.split('-', 1)[0] for name in labels]
    if set(families) != set(FAMILIES):
        raise ValueError('Expected the six original model families')
    family_index = np.array([FAMILIES.index(name) for name in families])
    scores = np.asarray(stage_logp, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1:] != (len(STAGES), len(labels)) or not np.isfinite(scores).all():
        raise ValueError('Expected finite score matrices in (passage, stage, class) order')
    return log_softmax(scores, axis=2), labels, family_index


def decompose(scores, family_index):
    mass = np.stack([logsumexp(scores[:, :, family_index == i], axis=2) for i in range(len(FAMILIES))], axis=2)
    return mass, scores - mass[:, :, family_index]


def mix(values, weights):
    return np.einsum('nsc,s->nc', values, weights)


def fit_weights(features, target, row_weights, allowed=None):
    row = np.arange(len(target))
    uniform = np.full(len(STAGES), 1. / len(STAGES))
    gold_features = features[row, :, target]
    gold_mean = row_weights @ gold_features

    def objective(weights):
        scores = mix(features, weights)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        normalizer = logsumexp(scores, axis=1)
        posterior = np.exp(scores - normalizer[:, None])
        displacement = weights - uniform
        loss = row_weights @ (normalizer - scores[row, target])
        gradient = np.einsum('n,nc,nsc->s', row_weights, posterior, features) - gold_mean
        return (float(loss + .5 * REGULARIZATION * (displacement @ displacement)),
                gradient + REGULARIZATION * displacement)

    simplex = dict(type='eq', fun=lambda weights: weights.sum() - 1., jac=lambda weights: np.ones_like(weights))
    result = minimize(objective, uniform, jac=True, method='SLSQP', constraints=[simplex],
                      bounds=[(0., 1.)] * len(STAGES), options={'maxiter': 300, 'ftol': 1e-11})
    if not result.success or not np.isfinite(result.x).all() or not np.isclose(result.x.sum(), 1., atol=1e-8):
        raise RuntimeError(f'Stage reliability fit did not converge: {result.message}')
    weights = result.x.clip(0., 1.)
    return (weights / weights.sum()).tolist()


def fit(stage_logp, labels, truth_labels, row_weights, *, factored):
    """Fit on aligned rows and return JSON-safe stage reliabilities."""
    scores, labels, family_index = inputs(stage_logp, labels)
    lookup = {name: i for i, name in enumerate(labels)}
    gold = np.array([lookup.get(name, -1) for name in truth_labels], dtype=int)
    weights = np.asarray(row_weights, dtype=np.float64)
    if (gold.shape != (len(scores),) or (gold < 0).any() or weights.shape != gold.shape
            or not np.isfinite(weights).all() or (weights < 0.).any() or weights.sum() <= 0.):
        raise ValueError('Fit needs aligned known labels and nonnegative row weights')
    weights = weights / weights.sum()
    params = dict(format=FORMAT, stage_order=list(STAGES), label_order=labels, factored=bool(factored),
                  regularization=REGULARIZATION)
    if factored:
        mass, variant = decompose(scores, family_index)
        gold_family = family_index[gold]
        same_family = family_index[None, :] == gold_family[:, None]
        params.update(family_weights=fit_weights(mass, gold_family, weights),
                      variant_weights=fit_weights(variant, gold, weights, allowed=same_family))
    else:
        params['flat_weights'] = fit_weights(scores, gold, weights)
    return params


def portable_weights(params, name):
    weights = np.asarray(params[name], dtype=np.float64)
    if (weights.shape != (len(STAGES),) or not np.isfinite(weights).all() or (weights < 0.).any()
            or not np.isclose(weights.sum(), 1., atol=1e-8)):
        raise ValueError(f'Invalid portable stage weights: {name}')
    return weights


def predict(stage_logp, labels, params):
    """Return normalized log probabilities over the original 18 classes."""
    scores, labels, family_index = inputs(stage_logp, labels)
    if (params.get('format') != FORMAT or params.get('label_order') != labels
            or params.get('stage_order') != list(STAGES) or type(params.get('factored')) is not bool):
        raise ValueError('The fitted stage capsule does not match the trace schema')
    if not params['factored']:
        return log_softmax(mix(scores, portable_weights(params, 'flat_weights')), axis=1)
    mass, variant = decompose(scores, family_index)
    family_logp = log_softmax(mix(mass, portable_weights(params, 'family_weights')), axis=1)
    variant_score = mix(variant, portable_weights(params, 'variant_weights'))
    output = np.empty_like(variant_score)
    for i in range(len(FAMILIES)):
        columns = family_index == i
        output[:, columns] = family_logp[:, i, None] + log_softmax(variant_score[:, columns], axis=1)
    return output
