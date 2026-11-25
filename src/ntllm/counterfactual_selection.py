"""Fit a rescue-versus-harm decision on frozen incumbent/candidate scores."""

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from .sibling_odds_replay import replay_predictions


STAGES = ('stack_logp', 'glm_logp', 'sibling_logp')
GLM = ('glm-5.3', 'glm-5.3-flash')
FABLE = 'claude-fable-5'
SPACED_DASH = ('claude-opus-4-6', 'claude-opus-5', *GLM)


def member_family(name):
    if name.startswith('native__'):
        return 'native'
    stem = name.removesuffix('_twin').removesuffix('_gauss')
    if stem in ('infer__q14v5', 'infer__q8v5', 'infer__n14b_clean'):
        return 'infer__qwen3'
    if stem in ('infer__tfidf', 'infer__svc4', 'infer__distort'):
        return 'infer__lexical'
    return stem


def member_support(members, coverage, shape, incumbent):
    group_data = {}
    local_rows = np.arange(shape[0])
    for name, scores in members.items():
        mask = np.asarray(coverage.get(name, np.ones(shape, dtype=bool)), dtype=bool)
        if mask.shape == (shape[0],):
            mask = np.broadcast_to(mask[:, None], shape)
        if scores.shape != shape or mask.shape != shape or not np.isfinite(scores[mask]).all():
            raise ValueError(f'Invalid observed member scores or coverage for {name}')
        ranks = np.argsort(np.argsort(-np.where(mask, scores, -np.inf), axis=1, kind='stable'), axis=1) + 1
        covered = mask & mask[local_rows, incumbent, None]
        values = (mask & (ranks == 1), mask & (ranks <= 2), covered)
        group = member_family(name)
        if group in group_data:
            values = tuple(prior | added for prior, added in zip(group_data[group], values, strict=True))
        group_data[group] = values
    result = np.zeros((3, *shape), dtype=np.int16)
    for values in group_data.values():
        result += np.stack(values)
    return result


def build_actions(truth, trace, baseline, members=None, coverage=None):
    """Return one row for each candidate alternative, indexed by frozen passage ID."""
    labels = pd.Index(trace['labels'])
    ids = pd.Index(trace['ids'], name=truth.index.name)
    if not ids.equals(truth.index) or not ids.is_unique or len(labels) != 18 or not labels.is_unique:
        raise ValueError('Counterfactual actions require frozen, aligned IDs and 18 classes')
    before = labels.get_indexer(baseline)
    if len(before) != len(ids) or (before < 0).any():
        raise ValueError('Incumbent labels must belong to the original class order')

    shape = (len(truth), len(labels))
    primary = np.zeros(shape, dtype=bool)
    votes = np.zeros(shape, dtype=np.int8)
    margins = {}
    raw_row = np.arange(len(truth))
    for stage in STAGES:
        scores = np.asarray(trace[stage])
        if scores.shape != shape or not np.isfinite(scores).all():
            raise ValueError(f'Invalid frozen stage scores for {stage}')
        ranks = np.argsort(-scores, axis=1, kind='stable')[:, :3]
        primary[raw_row[:, None], ranks] = True
        votes[raw_row, ranks[:, 0]] += 1
        margins[stage] = (scores - scores[raw_row, before, None]).clip(-12., 12.)
    one, two, observed = member_support(members or {}, coverage or {}, shape, before)
    candidates = primary | (two > 0)
    candidates[raw_row, before] = False
    row, col = np.nonzero(candidates)
    anchor = before[row]
    family = np.array([label.split('-', 1)[0] for label in labels])
    passed = np.asarray(baseline)
    protected = np.asarray(trace['glm_winner'], dtype=bool) | np.asarray(trace['fable_guard'], dtype=bool)
    pre_typography = replay_predictions(trace, truth.text, typography=False)
    typography = pre_typography != passed
    text = truth.text
    forbidden = (text.str.contains(' — ', regex=False).to_numpy()[row] & ~labels[col].isin(SPACED_DASH))
    forbidden |= text.str.contains('’', regex=False).to_numpy()[row] & np.isin(family[col], ('claude', 'glm'))
    forbidden |= text.str.contains('—', regex=False).to_numpy()[row] & (family[col] == 'gemini')
    result = pd.DataFrame(dict(
        baseline=labels[anchor], candidate=labels[col], strict_allowed=~(protected[row] | typography[row] | forbidden),
        feature_same_family=family[col] == family[anchor],
        feature_short=truth.n_words.lt(100).to_numpy()[row],
        feature_stage_votes=votes[row, col], feature_incumbent_votes=votes[row, anchor],
        feature_glm_winner=np.asarray(trace['glm_winner'])[row], feature_fable_guard=np.asarray(trace['fable_guard'])[row],
        feature_glm_incumbent=labels[anchor].isin(GLM), feature_fable_incumbent=labels[anchor] == FABLE,
        feature_glm_candidate=labels[col].isin(GLM), feature_fable_candidate=labels[col] == FABLE,
        feature_original_typography_change=typography[row],
        feature_member_only=~primary[row, col], feature_member_top1_groups=one[row, col],
        feature_member_top2_groups=two[row, col], feature_member_covered_groups=observed[row, col],
        feature_incumbent_member_top1_groups=one[row, anchor],
        **{f'feature_margin_{stage}': value[row, col] for stage, value in margins.items()},
    ), index=ids.take(row))
    if 'model' in truth:
        y = labels.get_indexer(truth.model)
        if (y < 0).any():
            raise ValueError('Known outcomes must belong to the original class order')
        domain = truth.get('domain', truth.get('cohort', pd.Series('single', index=ids)))
        result['outcome'] = (col == y[row]).astype(np.int8) - (anchor == y[row]).astype(np.int8)
        result['true_model'] = truth.model.to_numpy()[row]
        result['domain'] = domain.to_numpy()[row]
    return result


def fit_scoring_model(actions, fit_ids, row_weights=None, *, C=.15):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fit_ids = pd.Index(fit_ids)
    if fit_ids.empty or not fit_ids.is_unique or not fit_ids.isin(actions.index).all():
        raise ValueError('Expected unique and known counterfactual calibration IDs')
    fit = actions.loc[actions.index.isin(fit_ids)]
    by_id = fit.loc[~fit.index.duplicated(), ['domain', 'true_model']]
    if row_weights is None:
        inverse_class = 1. / by_id.groupby(['domain', 'true_model']).true_model.transform('size')
        represented = by_id.groupby('domain').true_model.transform('nunique')
        row_weights = inverse_class / represented
    weights = pd.Series(row_weights).reindex(fit.index).to_numpy(dtype=float)
    weights /= fit.groupby(level=0).outcome.transform('size').to_numpy()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Every fit passage needs a finite positive outcome weight')
    weights /= weights.mean()
    columns = [name for name in actions.columns if name.startswith('feature_')]
    if fit.outcome.nunique() < 2:
        return columns, None, None
    x = fit[columns].to_numpy(dtype=float)
    scale = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=C, max_iter=600)
    model.fit(scale.transform(x), fit.outcome, sample_weight=weights)
    return columns, scale, model


def resolve_action_scores(target, target_ids, threshold):
    incumbent = target.loc[~target.index.duplicated(), 'baseline'].reindex(target_ids)
    result = incumbent.to_frame()
    for prefix, eligible in (('', target), ('protected_', target.loc[target.strict_allowed])):
        best = eligible.sort_values('expected_delta', ascending=False, kind='stable')
        best = best.loc[~best.index.duplicated()].reindex(target_ids)
        result[prefix + 'candidate'] = best.candidate.fillna(incumbent)
        result[prefix + 'expected_delta'] = best.expected_delta.fillna(0.)
        result[prefix + 'prediction'] = result[prefix + 'candidate'].where(
            result[prefix + 'expected_delta'].gt(threshold), incumbent)
    return result


def target_actions(actions, target_ids):
    target_ids = pd.Index(target_ids, name=actions.index.name)
    if not target_ids.is_unique or not target_ids.isin(actions.index).all():
        raise ValueError('Expected unique and known counterfactual prediction IDs')
    return actions.loc[actions.index.isin(target_ids)].copy(), target_ids


def fit_predict(actions, fit_ids, target_ids, row_weights=None, *, C=.15, threshold=0.):
    """Fit weighted action outcomes and select positive expected gain over keep."""
    target, target_ids = target_actions(actions, target_ids)
    columns, scale, model = fit_scoring_model(actions, fit_ids, row_weights, C=C)
    target['expected_delta'] = 0.
    if model is not None and len(target):
        z = target[columns].to_numpy(dtype=float)
        target['expected_delta'] = model.predict_proba(scale.transform(z)) @ model.classes_.astype(float)
    return resolve_action_scores(target, target_ids, threshold)


def export_scoring_model(actions, fit_ids, row_weights, threshold):
    columns, scale, model = fit_scoring_model(actions, fit_ids, row_weights)
    if model is None:
        raise ValueError('A deployable model requires both correct and incorrect actions')
    return dict(features=columns, threshold=float(threshold), mean=scale.mean_.tolist(),
                scale=scale.scale_.tolist(), coefficient=model.coef_.tolist(),
                intercept=model.intercept_.tolist(), outcomes=model.classes_.tolist())


def replay_scoring_model(actions, payload, target_ids):
    target, target_ids = target_actions(actions, target_ids)
    values = target[payload['features']].to_numpy(dtype=float)
    standardized = (values - np.asarray(payload['mean'])) / np.asarray(payload['scale'])
    logits = standardized @ np.asarray(payload['coefficient']).T + np.asarray(payload['intercept'])
    outcomes = np.asarray(payload['outcomes'], dtype=float)
    if len(outcomes) == 2:
        positive = expit(logits[:, 0])
        probability = np.column_stack([1. - positive, positive])
    else:
        probability = softmax(logits, axis=1)
    target['expected_delta'] = probability @ outcomes
    return resolve_action_scores(target, target_ids, payload['threshold'])
