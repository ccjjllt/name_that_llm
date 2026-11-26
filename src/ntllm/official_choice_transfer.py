"""Test whether official prompt-OOF scores improve a selector on actual Sibling Odds traces."""

import json

import numpy as np
import pandas as pd

from .cached_evidence_oof import restore_legacy_anchor
from .candidate_choice import CUTOFFS, choice_features, guard_all, propose, train_choice
from .counterfactual_selection import STAGES
from .counterfactual_selection_audit import calibration_ids, canonical_prompt, class_weights, make_blocks
from .glm_prefix_response_replay import with_glm_scores
from .heldout_source_audit import text_key
from .kiro_pb115_oof import GLM, load_features, load_native_oof, select_original_route
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, same_ids, same_meta
from .sibling_odds_replay import replay_predictions
from .support_censored_selector import prompt_folds


ARMS = {
    'external_original': (0., False),
    'half_official': (.50, False),
    'official_only': (1., False),
    'external_normalized': (0., True),
    'half_official_normalized': (.50, True),
}


def build_official(official, native, paired, features):
    from pngy_residual import Residual

    rows, labels, folds, before = load_native_oof(official, native)
    rows = rows.assign(n_words=rows.text.str.split().str.len(), cohort='official_surrogate', fold=folds)
    archived = read_ids(paired)
    same_ids(archived.index, rows.index, 'Matched Legacy54 folds')
    archived = archived.loc[rows.index]
    same_meta(archived, rows, ['model', 'prompt_id', 'n_words', 'fold'], 'Matched Legacy54 folds')
    x, _, tokens = load_features(features, rows.index)
    costs = tokens + 30
    native_route = select_original_route(before, folds, pd.Index(labels).get_indexer(GLM), costs)
    if (native_route.sum() != 587 or costs[native_route].sum() != 87409
            or not np.array_equal(native_route, archived.routed)):
        raise ValueError('Original official GLM route or budget changed')
    post = restore_legacy_anchor(x, rows.model.to_numpy(), labels, folds, before, native_route, archived.legacy54)
    sibling = np.empty_like(before)
    for fold in sorted(set(folds)):
        held = folds == fold
        readout = Residual(labels).fit(rows.loc[~held, 'text'], rows.loc[~held, 'model'])
        sibling[held] = readout.predict_log_probability(before[held], rows.loc[held, 'text'])
    trace = with_glm_scores(dict(ids=rows.index.to_numpy(dtype=str), labels=labels, stack_logp=before,
                                 sibling_logp=sibling), post)
    return rows, trace, replay_predictions(trace, rows.text)


def build_features(rows, trace, baseline):
    raw, pool, names = choice_features(rows, trace, baseline)
    normalized = dict(trace)
    for key in STAGES:
        values = np.asarray(trace[key], dtype=np.float64)
        normalized[key] = values / np.maximum(values.std(axis=1, keepdims=True), .50)
    transformed, check, normalized_names = choice_features(rows, normalized, baseline)
    if not np.array_equal(pool, check) or names != normalized_names:
        raise ValueError('Row-wise score normalization must preserve candidate sets')
    return (raw, transformed), pool


def run(historical, prior, gen003, kimi, inputs, held, source, official, native, paired, features,
        archived_choice, dash_report, output, draws=3000):
    historical_blocks, source_truth = make_blocks(historical, prior, gen003, kimi, inputs, held, source)
    off, off_trace, base = build_official(official, native, paired, features)
    labels = off_trace['labels']
    frames, values, pools, captures = [], [], [], []
    blocks = [(truth, trace, incumbent) for truth, trace, incumbent, _, _ in historical_blocks]
    for (truth, trace, incumbent), directory in zip(blocks, (historical, gen003, kimi, held), strict=True):
        captures.append((truth, trace, incumbent, json.loads((directory / 'glm_report.json').read_text())))
    for truth, trace, incumbent in (*blocks, (off, off_trace, base)):
        if not np.array_equal(labels, trace['labels']):
            raise ValueError('All trace labels must be in original model order')
        candidates, pool = build_features(truth, trace, incumbent)
        frames.append(truth[['model', 'prompt_id', 'cohort', 'n_words', 'text']].assign(
            baseline=incumbent, strict=truth.get('strict_topic_holdout', False)))
        values.append(candidates)
        pools.append(pool)
    rows = pd.concat(frames, verify_integrity=True)
    pools = np.concatenate(pools)
    by_scale = [np.concatenate(parts) for parts in zip(*values, strict=True)]
    true_class = pd.Index(labels).get_indexer(rows.model)
    if (true_class < 0).any():
        raise ValueError('Every observed truth must be in the original class order')
    c1, extras, separation = calibration_ids(rows, source_truth)
    calibration = c1.append(extras)
    target = rows.index[rows.cohort.isin(['evaluation', 'prvsiyan'])]
    reference = read_ids(archived_choice).loc[target]
    patch = read_ids(dash_report).loc[target]
    for prior_score in (reference, patch):
        same_ids(prior_score.index, target, 'Archived candidate decisions', ordered=True)
        if not np.array_equal(prior_score.baseline, rows.loc[target, 'baseline']):
            raise ValueError('The archive does not use the same incumbent')
    body = text_key(rows.text)
    prompt = canonical_prompt(rows.prompt_id)
    reserved = calibration.append(target)
    official_ids = off.index[~body.loc[off.index].isin(body.loc[reserved])
                             & ~prompt.loc[off.index].isin(prompt.loc[reserved])]
    if rows.loc[official_ids, 'model'].nunique() != len(labels):
        raise ValueError('Official calibration must contain all 18 exact labels')

    def predict(arm, fit, predict_ids):
        official_weight, normalized = ARMS[arm]
        x = by_scale[int(normalized)]
        use = np.empty(0, dtype=int)
        weights = []
        for ids, mass in ((fit, 1. - official_weight), (official_ids, official_weight)):
            if mass <= 0:
                continue
            indices = rows.index.get_indexer(ids)
            value = class_weights(rows, ids)
            value.index = indices
            weights.append(mass * value / value.sum())
            use = np.r_[use, indices]
        capsule = train_choice(x, pools, true_class, pd.concat(weights, verify_integrity=True), use)
        target_pos = rows.index.get_indexer(predict_ids)
        keep = rows.loc[predict_ids, 'baseline']
        raw = propose(x[target_pos], pools[target_pos], keep, labels, capsule, ())
        candidate, margin = raw['pairwise__open_candidates']
        proposals = pd.DataFrame(dict(baseline=keep, candidate=candidate, margin=margin), index=predict_ids)
        return capsule, proposals.assign(candidate=guard_all(captures, proposals).native_and_fable)

    source_groups = rows.loc[calibration].groupby('cohort').groups
    evaluation = rows.loc[target].drop(columns='text').assign(scored_dash_patch=patch.scored_dash_patch)
    report = dict(calibration=separation, official_observed=len(off), official_fit=len(official_ids),
                  official_fit_prompts=int(off.loc[official_ids].prompt_id.nunique()), arms={}, panels={})
    for name, (share, normalized) in ARMS.items():
        held_predictions = [predict(name, fit, validate)[1] for fit, validate in prompt_folds(rows.loc[calibration])]
        calibrated = pd.concat(held_predictions, verify_integrity=True).loc[calibration]
        sweep = {}
        for cutoff in CUTOFFS:
            verdict = calibrated.candidate.where(calibrated.margin.gt(cutoff), calibrated.baseline)
            sweep[cutoff] = float(np.mean([balanced_accuracy(rows.loc[group, 'model'], verdict.loc[group])
                                         for group in source_groups.values()]))
        cutoff = max(CUTOFFS, key=lambda value: (sweep[value], value))
        capsule, tested = predict(name, calibration, target)
        evaluation[name] = tested.candidate.where(tested.margin.gt(cutoff), tested.baseline)
        used_dash_patch = evaluation.scored_dash_patch.ne(evaluation.baseline)
        evaluation['dash_plus__' + name] = evaluation[name].where(~used_dash_patch, evaluation.scored_dash_patch)
        report['arms'][name] = dict(official_objective_share=share, normalized=normalized, cutoff=cutoff,
                                   thresholds_oof=sweep, used_rows=capsule['used_rows'])
        print(name, cutoff, capsule['used_rows'], flush=True)
    if not np.array_equal(evaluation.external_original, reference['pairwise__open_candidates__native_and_fable']):
        raise ValueError('New harness must reproduce the previous raw external-only pairwise selector')
    scopes = {'c2': evaluation.cohort.eq('evaluation'), 'heldout': evaluation.cohort.eq('prvsiyan'),
              'held_strict': evaluation.strict}
    for name, mask in scopes.items():
        for length, part in (('all', evaluation.loc[mask]), ('short', evaluation.loc[mask & evaluation.n_words.lt(100)])):
            report['panels'][f'{name}__{length}'] = dict(
                against_original=[paired_stats(part, 'baseline', arm, draws) for arm in ARMS],
                versus_scored_patch=[paired_stats(part, 'scored_dash_patch', 'dash_plus__' + arm, draws) for arm in ARMS],
                value_of_official=[paired_stats(part, 'dash_plus__external_original', 'dash_plus__half_official', draws),
                                   paired_stats(part, 'dash_plus__external_normalized',
                                                'dash_plus__half_official_normalized', draws)])
    report['limits'] = ('Official Native+Legacy54 uses out-of-prompt predictions and a fold-fitted public residual. '
                        'It is a score surrogate. C2 and Prvsiyan use archived Sibling Odds and no label fitting. '
                        'Cross-corpus topic filtering uses available prompt identifiers and normalized exact body.')
    output.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
