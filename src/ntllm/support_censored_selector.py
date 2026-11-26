"""Calibrate candidate decisions only where both endpoints have source support."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.model_selection import GroupKFold

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import build_actions, fit_scoring_model, replay_scoring_model, resolve_action_scores
from .counterfactual_selection_audit import calibration_ids, canonical_prompt, class_weights, lock_threshold, make_blocks
from .heldout_source_audit import text_key
from .peer_context_report import paired_stats


MIN_PROMPTS = 5
ARMS = ('original', 'post_rank_veto', 'pre_rank_eligibility', 'censored_fit')
FOCAL = ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3')


def collect_support(rows):
    counts = rows.assign(prompt=canonical_prompt(rows.prompt_id)).groupby(['cohort', 'model']).prompt.nunique()
    by_source = {str(name): sorted(part.index.get_level_values('model')[part.ge(MIN_PROMPTS)])
                 for name, part in counts.groupby(level='cohort')}
    prompt_counts = {str(name): part.droplevel('cohort').to_dict() for name, part in counts.groupby(level='cohort')}
    pairs = sorted({(a, c) for supported in by_source.values() for a in supported for c in supported if a != c})
    return by_source, prompt_counts, pairs


def fit_capsules(actions, rows, ids):
    class_support, prompt_counts, pairs = collect_support(rows.loc[ids])
    train = actions.loc[actions.index.isin(ids)]
    kept = []
    for name, supported in class_support.items():
        mask = train.index.isin(rows.index[rows.cohort.eq(name)])
        kept.append(mask & train.baseline.isin(supported) & train.candidate.isin(supported))
    censored = train.loc[np.logical_or.reduce(kept)]
    weights = class_weights(rows, ids)

    def export(subset):
        keeps = ids[ids.isin(subset.index)]
        if not len(keeps):
            return None
        columns, scale, model = fit_scoring_model(subset, keeps, weights.loc[keeps])
        if model is None:
            return None
        return dict(features=columns, mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                    coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(),
                    outcomes=model.classes_.tolist())

    receipt = dict(fit_rows=len(ids), fit_counts=rows.loc[ids].cohort.value_counts().to_dict(),
                   fit_prompts=int(canonical_prompt(rows.loc[ids, 'prompt_id']).nunique()),
                   support_prompt_counts=prompt_counts, admitted_classes=class_support, supported_pairs=len(pairs),
                   actions=len(train), censored_actions=len(censored), censored_rows=censored.index.nunique(),
                   censored_outcomes=censored.outcome.value_counts().sort_index().to_dict())
    return dict(original=export(train), censored_fit=export(censored), pairs=pairs), receipt


def action_gains(actions, model):
    if model is None:
        return np.zeros(len(actions))
    values = actions[model['features']].to_numpy(dtype=float)
    standardized = (values - np.asarray(model['mean'])) / np.asarray(model['scale'])
    logits = standardized @ np.asarray(model['coefficient']).T + np.asarray(model['intercept'])
    outcomes = np.asarray(model['outcomes'], dtype=float)
    if len(outcomes) == 2:
        positive = expit(logits[:, 0])
        probability = np.column_stack([1. - positive, positive])
    else:
        probability = softmax(logits, axis=1)
    return probability @ outcomes


def score_proposals(actions, ids, capsules):
    target = actions.loc[actions.index.isin(ids)].copy()
    supported_pairs = [tuple(pair) for pair in capsules['pairs']]
    pairs = pd.MultiIndex.from_frame(target[['baseline', 'candidate']])
    eligible = pairs.isin(supported_pairs)
    supported = pd.Series(eligible, index=target.index).groupby(level=0).any().reindex(ids, fill_value=False)
    gains = {name: action_gains(target, capsules[name]) for name in ('original', 'censored_fit')}

    def resolve(scores):
        calls = resolve_action_scores(target.assign(expected_delta=scores), ids, 0.)
        calls = calls[['baseline', 'candidate', 'expected_delta']]
        unavailable = ~np.isfinite(calls.expected_delta)
        calls['candidate'] = calls.candidate.where(~unavailable, calls.baseline)
        calls['expected_delta'] = calls.expected_delta.where(~unavailable, 0.)
        return calls

    original = resolve(gains['original'])
    top_eligible = pd.MultiIndex.from_frame(original[['baseline', 'candidate']]).isin(supported_pairs)
    veto = original.assign(candidate=original.candidate.where(top_eligible, original.baseline),
                           expected_delta=original.expected_delta.where(top_eligible, 0.))
    return dict(original=original, post_rank_veto=veto,
                pre_rank_eligibility=resolve(np.where(eligible, gains['original'], -np.inf)),
                censored_fit=resolve(np.where(eligible, gains['censored_fit'], -np.inf))), supported, top_eligible


def accepted(calls, threshold):
    return calls.candidate.where(calls.expected_delta.gt(threshold), calls.baseline)


def prompt_folds(rows):
    prompt = canonical_prompt(rows.prompt_id)
    body = text_key(rows.text)
    for fit, valid in GroupKFold(4).split(rows, groups=prompt):
        train, held = rows.index[fit], rows.index[valid]
        train = train[~body.loc[train].isin(body.loc[held])]
        yield train, held


def calibrate(actions, rows, fit_ids):
    folded = {name: [] for name in ARMS}
    logs = []
    for train, valid in prompt_folds(rows.loc[fit_ids]):
        model, receipt = fit_capsules(actions, rows, train)
        calls, _, _ = score_proposals(actions, valid, model)
        for name in ARMS:
            folded[name].append(calls[name])
        logs.append(receipt)
    oof = {name: pd.concat(parts, verify_integrity=True).loc[fit_ids] for name, parts in folded.items()}
    thresholds, cutoff_audit = {}, {}
    for name, calls in oof.items():
        cutoff, cutoffs = lock_threshold(rows, calls, fit_ids, False)
        thresholds[name] = thresholds['original'] if name in ('post_rank_veto', 'pre_rank_eligibility') else cutoff
        cutoff_audit[name] = dict(source_mean_ba=cutoffs, best_independent_cutoff=cutoff,
                                  accepted=int(accepted(calls, thresholds[name]).ne(calls.baseline).sum()))
    model, fit = fit_capsules(actions, rows, fit_ids)
    model['thresholds'] = thresholds
    return model, dict(fit=fit, folds=logs, oof_thresholds=thresholds, oof_results=cutoff_audit)


def score_capsules(queries, trace, incumbent, receipt, capsules):
    """Replay all frozen candidates and the shared protection policy on unlabeled passages."""
    if list(queries) != ['id', 'text'] or capsules['version'] != 1 or capsules['label_order'] != list(trace['labels']):
        raise ValueError('Frozen support capsules require body-only queries with the traced label order')
    text = queries.set_index('id', verify_integrity=True).text.fillna('').astype(str)
    truth = text.to_frame().assign(n_words=text.str.split().str.len())
    actions = build_actions(truth, trace, incumbent)
    calls, supported, top_eligible = score_proposals(actions, truth.index, capsules)
    records = pd.DataFrame({'baseline': incumbent, 'supported_action': supported,
                            'original_top_supported': top_eligible}, index=truth.index)
    for name, proposals in calls.items():
        threshold = capsules['thresholds'][name]
        opened = proposals.assign(open=accepted(proposals, threshold))
        protected = guard_frame(truth, trace, incumbent, opened, receipt=receipt)
        records[f'{name}_candidate'] = proposals.candidate
        records[f'{name}_expected_delta'] = proposals.expected_delta
        records[f'{name}_open'] = opened.open
        records[name] = protected.native_and_fable if receipt.get('status') == 'applied' else records.baseline
    records['native_routed'] = protected.route
    return records


def summarize(rows, draws):
    contrasts = [('baseline', name) for name in ARMS]
    contrasts += [('original', 'post_rank_veto'), ('post_rank_veto', 'pre_rank_eligibility'),
                  ('pre_rank_eligibility', 'censored_fit'), ('original', 'censored_fit')]
    by_label = {}
    for label, part in rows.groupby('model'):
        old = part.baseline.eq(label)
        by_label[label] = dict(rows=len(part), baseline_correct=int(old.sum()),
                               comparisons={name: dict(fixed=int((part[name].eq(label) & ~old).sum()),
                                                        harmed=int((part[name].ne(label) & old).sum()))
                                            for name in ARMS})
    return dict(supported_rows=int(rows.supported_action.sum()), top_supported_rows=int(rows.original_top_supported.sum()),
                rows=len(rows),
                accepted_open={name: int(rows[f'{name}_open'].ne(rows.baseline).sum()) for name in ARMS},
                comparisons=[paired_stats(rows, before, after, draws) for before, after in contrasts],
                by_label=by_label)


def run(historical, prior, gen003, kimi, inputs, output, reference, independent=None, source=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    directories = [historical, gen003, kimi, *([independent] if independent else [])]
    labels = list(blocks[0][1]['labels'])
    frames, feature_frames, captures = [], [], []
    for (truth, trace, incumbent, _, _), folder in zip(blocks, directories, strict=True):
        if list(trace['labels']) != labels:
            raise ValueError('The traced class orders disagree')
        frames.append(truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(
            baseline=incumbent, strict=truth.get('strict_topic_holdout', False)))
        feature_frames.append(build_actions(truth, trace, incumbent))
        receipt = json.loads((folder / 'glm_report.json').read_text())
        captures.append((truth, trace, incumbent, receipt))
    rows = pd.concat(frames, verify_integrity=True)
    actions = pd.concat(feature_frames)
    dev, external, separation = calibration_ids(rows, held_truth)
    fit_ids = dev.append(external)
    capsules, calibration = calibrate(actions, rows, fit_ids)
    capsules.update(version=1, label_order=labels)
    frozen_bytes = reference.read_bytes()
    frozen = json.loads(frozen_bytes)
    if (frozen['arm'] != 'stage_multisource' or frozen['version'] != 1 or frozen['label_order'] != labels
            or frozen['threshold'] != capsules['thresholds']['original']):
        raise ValueError('Original control does not match the archived selector identity and OOF threshold')
    ref_calls = replay_scoring_model(actions, frozen, rows.index)
    live_calls, _, _ = score_proposals(actions, rows.index, capsules)
    original = live_calls['original']
    score_error = float(np.max(np.abs(original.expected_delta - ref_calls.expected_delta)))
    if score_error > 1e-10 or not accepted(original, frozen['threshold']).equals(ref_calls.prediction):
        raise ValueError('Original control does not reproduce the archived selector scores and proposals')
    capsules['original'] = frozen
    capsules['reference_sha256'] = hashlib.sha256(frozen_bytes).hexdigest()

    def evaluate(model, index):
        pieces = []
        for truth, trace, incumbent, receipt in captures:
            shared = truth.index.intersection(index, sort=False)
            if len(shared):
                predictions = score_capsules(truth.text.to_frame().reset_index(), trace, incumbent, receipt, model)
                pieces.append(predictions.loc[shared])
        result = pd.concat(pieces, verify_integrity=True).loc[index]
        return rows.loc[index].drop(columns=['text', 'baseline']).join(result)

    target = rows.index[rows.cohort.isin(['evaluation', 'reconstructed_stress', 'prvsiyan'])]
    evaluated = evaluate(capsules, target)
    scopes = {'c2': evaluated.loc[evaluated.cohort.eq('evaluation')],
              'stress': evaluated.loc[evaluated.cohort.eq('reconstructed_stress')],
              'prvsiyan_895': evaluated.loc[evaluated.cohort.eq('prvsiyan')],
              'prvsiyan_316': evaluated.loc[evaluated.cohort.eq('prvsiyan') & evaluated.strict]}
    panels = {f'{name}_{subset}': summarize(part, draws) for name, scope in scopes.items()
              for subset, part in [('all', scope), ('short', scope.loc[scope.n_words.lt(100)]),
                                   ('focal', scope.loc[scope.model.isin(FOCAL)])] if not part.empty}

    prompt = canonical_prompt(rows.prompt_id)
    body = text_key(rows.text)
    leave_source = {}
    for name, held in rows.loc[fit_ids].groupby('cohort'):
        complement = fit_ids.difference(held.index)
        usable = ~prompt.loc[complement].isin(prompt.loc[held.index]) & ~body.loc[complement].isin(body.loc[held.index])
        model, audit = calibrate(actions, rows, complement[usable])
        model.update(version=1, label_order=labels)
        result = evaluate(model, held.index)
        leave_source[name] = dict(calibration=audit, paired=summarize(result, draws))

    report = dict(separation=separation, min_fit_prompts=MIN_PROMPTS, calibration=calibration,
                  original_max_score_error=score_error, frozen_reference_sha256=capsules['reference_sha256'],
                  common_policy='native_and_fable', exported_for='frozen DRJ external evaluation',
                  leave_source=leave_source, panels=panels)
    output.mkdir(parents=True, exist_ok=True)
    evaluated.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'capsules.json').write_text(json.dumps(capsules, indent=2, allow_nan=False) + '\n')
    print('Verified archived original selector and froze four controls for DRJ', flush=True)
    return report
