"""Coupled prompt bootstrap and smooth averaging of the fixed fusion family.

Only supplied calibration labels enter fitting. Query application accepts raw
branch scores and a fitted member, never query labels. The original data,
model, relation scores and eleven candidate recipes remain fixed.
"""
import hashlib

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from .q8_fusion import _branch_active, candidate_recipes, mix_branches


BRANCHES = ('plain', 'centered', 'gaussian', 'gaussian_noninjective')
METHODS = ('hard', 'soft', 'uniform', 'gaussian')
SOFT_SCALE = 1. / 24.


def make_coupled_bootstrap_bank(n_prompts, draws, seed):
    """Generate iid slot rankings for exact, strongly coupled multinomials.

    Full data use all n slots. Deleting prompt j uses the first n-1 slots and
    replaces only first-choice j by its second choice. Each marginal remains
    an ordinary sample of n or n-1 iid uniform prompt choices, respectively.
    """
    if n_prompts < 2 or draws < 1:
        raise ValueError('Use at least two prompts and a positive member count.')
    uniforms = np.random.default_rng(seed).random((draws, n_prompts, n_prompts))
    order = np.argsort(uniforms, axis=2)[..., :2].astype(np.int16)
    return dict(first=order[..., 0], second=order[..., 1],
                n_prompts=int(n_prompts), draws=int(draws), seed=int(seed))


def coupled_prompt_counts(bank, deleted_index=None):
    """Return an integer (draws, original prompt count) multiplicity matrix."""
    first, second = np.asarray(bank['first']), np.asarray(bank['second'])
    n = int(bank['n_prompts'])
    if first.shape != (int(bank['draws']), n) or second.shape != first.shape:
        raise ValueError('The bootstrap bank does not match its declared shape.')
    if ((first < 0) | (first >= n) | (second < 0) | (second >= n) | (first == second)).any():
        raise ValueError('Invalid first/second prompt choices.')
    if deleted_index is None:
        choices = first
    else:
        if not isinstance(deleted_index, (int, np.integer)) or not 0 <= deleted_index < n:
            raise ValueError('Deleted prompt position is outside the bank.')
        choices = np.where(first[:, :n-1] == deleted_index,
                           second[:, :n-1], first[:, :n-1])
    counts = np.zeros((len(first), n), dtype=np.int16)
    np.add.at(counts, (np.arange(len(first))[:, None], choices), 1)
    return counts


def prepare_member_context(rows, branches, labels):
    """Cache fixed calibration coordinates; caller supplies holdout provenance."""
    rows = rows.reset_index(drop=True)
    labels = list(labels)
    if len(labels) != 18 or len(set(labels)) != 18 or not rows.model.isin(labels).all():
        raise ValueError('Use the recorded eighteen-label order and valid calibration labels.')
    prompt_values = rows.prompt_id.astype(str).to_numpy()
    prompts = sorted(set(prompt_values))
    if not prompts:
        raise ValueError('No calibration prompts.')
    prompt_index = {p: i for i, p in enumerate(prompts)}
    row_prompt = np.asarray([prompt_index[p] for p in prompt_values], dtype=int)
    target = rows.model.map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
    arrays = {key: np.asarray(value) for key, value in branches.items()}
    if any(len(value) != len(rows) for value in arrays.values()):
        raise ValueError('Branch arrays do not align with calibration rows.')
    active = {name: _branch_active(arrays, name) for name in BRANCHES}
    return dict(prompts=prompts, row_prompt=row_prompt, target=target,
                labels=labels, branches=arrays, active=active,
                recipes=candidate_recipes(False))


def fit_weighted_temperature_records(context, row_multiplicity):
    """Use exactly the repeated-row balanced NLL without materializing repeats."""
    multiplicity = np.asarray(row_multiplicity, dtype=float)
    target_all = context['target']
    if multiplicity.shape != target_all.shape or not np.isfinite(multiplicity).all() or (multiplicity < 0).any():
        raise ValueError('Calibration multiplicities must be finite, nonnegative and aligned.')
    records = {}
    for name in BRANCHES:
        mask = context['active'][name] & (multiplicity > 0)
        y = target_all[mask]
        if not len(y):
            records[name] = dict(temperature=1., active_rows=0, unique_active_rows=0,
                                 represented_classes=0, status='inactive_identity')
            continue
        m = multiplicity[mask]
        values = np.asarray(context['branches'][name + '_lp'], dtype=float)[mask]
        counts = np.bincount(y, weights=m, minlength=18)
        weights = m / counts[y]
        weights /= weights.sum()

        def objective(log_temperature):
            lp = log_softmax(values / np.exp(log_temperature), axis=1)
            return float(-np.dot(weights, lp[np.arange(len(y)), y]))

        lower, upper = np.log(.05), np.log(20.)
        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded',
                                 options=dict(xatol=1e-6, maxiter=200))
        if not result.success or not np.isfinite(result.fun):
            raise ValueError('Weighted scalar temperature fitting did not converge.')
        best = min([0., float(result.x), lower, upper], key=objective)
        temperature = float(np.clip(np.exp(best), .05, 20.))
        records[name] = dict(temperature=temperature, active_rows=int(m.sum()),
                             unique_active_rows=len(y), represented_classes=int((counts > 0).sum()),
                             class_counts=counts.astype(int).tolist(),
                             balanced_nll_before=objective(0.),
                             balanced_nll_after=objective(np.log(temperature)),
                             boundary=bool(np.isclose(temperature, .05) or np.isclose(temperature, 20.)),
                             status='weighted_prompt_multiplicity')
    return records


def calibrate_member_branches(branches, temperatures):
    """Apply positive fitted temperatures while retaining relation activity."""
    result = dict(branches)
    for name in BRANCHES:
        temperature = float(temperatures[name]['temperature'])
        if not np.isfinite(temperature) or not .05 <= temperature <= 20.:
            raise ValueError('A temperature is outside the declared range.')
        active = _branch_active(branches, name)
        values = np.zeros_like(np.asarray(branches[name + '_lp'], dtype=float))
        values[active] = log_softmax(np.asarray(branches[name + '_lp'])[active] / temperature, axis=1)
        result[name + '_lp'] = values
    return result


def fit_member_from_counts(context, prompt_counts, *, soft_scale=SOFT_SCALE):
    """Share one temperature fit across hard, soft, uniform and G-only rules."""
    counts = np.asarray(prompt_counts)
    if (counts.shape != (len(context['prompts']),) or not np.isfinite(counts).all()
            or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all() or counts.sum() < 1):
        raise ValueError('Use positive-size integer prompt multiplicities.')
    if not np.isfinite(soft_scale) or soft_scale <= 0:
        raise ValueError('The predeclared soft scale must be positive.')
    m = counts[context['row_prompt']]
    temperatures = fit_weighted_temperature_records(context, m)
    calibrated = calibrate_member_branches(context['branches'], temperatures)
    target = context['target']
    class_counts = np.bincount(target, weights=m, minlength=18)
    represented = class_counts > 0
    scores = []
    for recipe in context['recipes']:
        correct = mix_branches(calibrated, recipe).argmax(1) == target
        correct_counts = np.bincount(target, weights=m * correct, minlength=18)
        scores.append(float(np.mean(correct_counts[represented] / class_counts[represented])))
    scores = np.asarray(scores)
    winner = int(np.argmax(scores))
    hard = np.eye(len(scores))[winner]
    soft = softmax(scores / soft_scale)
    uniform = np.full(len(scores), 1. / len(scores))
    gaussian = np.eye(len(scores))[[r['name'] for r in context['recipes']].index('gaussian_noninjective_fallback')]
    ordered = np.sort(scores)
    entropy = float(-np.sum(soft * np.log(soft)))
    return dict(temperatures=temperatures, candidate_scores=scores.tolist(),
                candidate_names=[r['name'] for r in context['recipes']],
                method_weights=dict(hard=hard.tolist(), soft=soft.tolist(), uniform=uniform.tolist(),
                                    gaussian=gaussian.tolist()),
                winner=winner, winner_name=context['recipes'][winner]['name'],
                top_two_ba_gap=float(ordered[-1] - ordered[-2]),
                exact_maximizers=int(np.sum(scores == scores[winner])),
                soft_scale=float(soft_scale), soft_entropy=entropy,
                soft_effective_candidates=float(np.exp(entropy)),
                resample_size=int(counts.sum()), unique_inbag_prompts=int(np.count_nonzero(counts)),
                counts_sha256=hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest())


def apply_member(branches, member, *, return_candidates=False):
    """Return the mixtures and calibrated G-only control; no truth is accepted."""
    recipes = candidate_recipes(False)
    if member['candidate_names'] != [r['name'] for r in recipes]:
        raise ValueError('The fitted candidate family changed.')
    calibrated = calibrate_member_branches(branches, member['temperatures'])
    candidates = np.stack([np.exp(mix_branches(calibrated, recipe)) for recipe in recipes])
    predictions = {}
    for name in METHODS:
        weights = np.asarray(member['method_weights'][name], dtype=float)
        if (weights.shape != (len(recipes),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1.)):
            raise ValueError('A fitted mixture must be a convex combination of the fixed recipes.')
        p = np.einsum('c,cnk->nk', weights, candidates)
        predictions[name] = p / p.sum(axis=1, keepdims=True)
    if return_candidates:
        predictions['candidates'] = candidates
    return predictions
