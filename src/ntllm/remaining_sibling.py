"""Measure a pure sibling selector after the exact scored Gemini repair."""

import json

import numpy as np
import pandas as pd

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import STAGES, build_actions, fit_scoring_model, replay_scoring_model
from .counterfactual_selection_audit import calibration_ids, class_weights, lock_threshold, make_blocks
from .peer_context_report import paired_stats
from .sibling_odds_replay import replay_predictions
from .support_censored_selector import action_gains, collect_support, prompt_folds, score_capsules


ARMS = ('existing_broad', 'existing_supported', 'same_family_broad_head', 'same_family_refit')


def sibling_actions(truth, trace, incumbent, actions):
    result = actions.loc[actions.feature_same_family].copy()
    row = truth.index.get_indexer(result.index)
    labels = pd.Index(trace['labels'])
    destination = labels.get_indexer(result.candidate)
    families = np.asarray([name.split('-', 1)[0] for name in labels])
    shares = families[destination, None] == families[None, :]
    for stage in STAGES:
        scores = trace[stage][row]
        local = np.where(shares, scores, -np.inf)
        orders = np.argsort(-local, axis=1, kind='stable')
        candidate_rank = np.argmax(orders == destination[:, None], axis=1) + 1
        result[f'feature_family_candidate_rank_{stage}'] = candidate_rank
    return result


def admitted(actions, pairs):
    return pd.MultiIndex.from_frame(actions[['baseline', 'candidate']]).isin(pairs)


def fit_sibling(actions, rows, ids):
    source_classes, counts, pairs = collect_support(rows.loc[ids])
    training = actions.loc[actions.index.isin(ids)]
    keep = np.zeros(len(training), dtype=bool)
    for domain, covered in source_classes.items():
        keep |= (training.index.isin(rows.index[rows.cohort.eq(domain)])
                 & training.baseline.isin(covered).to_numpy() & training.candidate.isin(covered).to_numpy())
    selected = training.loc[keep]
    fit_ids = ids[ids.isin(selected.index)]
    log = dict(source_class_prompt_counts=counts, supported_pairs=len(pairs),
               sibling_actions=len(training), eligible_sibling_actions=len(selected),
               fit_rows=len(fit_ids), outcomes=selected.outcome.value_counts().to_dict())
    if fit_ids.empty:
        return None, pairs, log
    columns, scale, model = fit_scoring_model(selected, fit_ids, class_weights(rows, ids).loc[fit_ids])
    if model is None:
        return None, pairs, log
    fit = dict(features=columns, mean=scale.mean_, scale=scale.scale_, coefficient=model.coef_,
               intercept=model.intercept_, outcomes=model.classes_.tolist())
    return fit, pairs, log


def best(actions, part, pairs, score):
    options = actions.loc[actions.index.isin(part.index)].copy()
    options['expected_delta'] = action_gains(options, score)
    options = options.loc[admitted(options, pairs)].sort_values('expected_delta', ascending=False, kind='stable')
    first = options.loc[~options.index.duplicated()].reindex(part.index)
    return pd.DataFrame(dict(baseline=part.baseline, candidate=first.candidate.fillna(part.baseline),
                             expected_delta=first.expected_delta.fillna(0.)))


def calibrate(actions, rows, ids):
    parts, folds = [], []
    for fit, held in prompt_folds(rows.loc[ids]):
        model, pairs, receipt = fit_sibling(actions, rows, fit)
        parts.append(best(actions, rows.loc[held], pairs, model))
        folds.append(receipt)
    oof = pd.concat(parts, verify_integrity=True).loc[ids]
    threshold, scores = lock_threshold(rows, oof, ids, False)
    model, pairs, report = fit_sibling(actions, rows, ids)
    return model, pairs, threshold, dict(folds=folds, final=report, threshold=threshold, prompt_oof_ba=scores)


def exact_patch(truth, trace, incumbent):
    pre = replay_predictions(trace, truth.text, typography=False)
    restore = ((pre != incumbent) & np.char.startswith(pre.astype(str), 'gemini-')
               & truth.text.str.contains('—', regex=False).to_numpy())
    return np.where(restore, pre, incumbent)


def evaluate(truth, trace, incumbent, native_report, original_model, support, stages, siblings, rows, fit, pairs, threshold):
    idx = rows.index
    current = rows.scored_patch
    original = replay_scoring_model(stages, original_model, idx)
    supplied = score_capsules(truth.text.to_frame().reset_index(), trace, incumbent, native_report, support)
    options = {
        'same_family_broad_head': (best(stages.loc[stages.feature_same_family], rows, pairs, original_model),
                                   original_model['threshold']),
        'same_family_refit': (best(siblings, rows, pairs, fit), threshold),
    }
    opened = {'existing_broad': original.prediction}
    for name, (choice, value) in options.items():
        opened[name] = choice.candidate.where(choice.expected_delta.gt(value), choice.baseline)
    results = rows.drop(columns='text').copy()
    for name, candidate in opened.items():
        frame = pd.DataFrame(dict(baseline=incumbent, open=candidate), index=idx)
        guarded = guard_frame(truth, trace, incumbent, frame, receipt=native_report).native_and_fable
        guarded = guarded if native_report.get('status') == 'applied' else rows.baseline
        results[name] = guarded.where(current.eq(rows.baseline), current)
    results['existing_supported'] = supplied.pre_rank_eligibility.where(current.eq(rows.baseline), current)
    for name in ('same_family_broad_head', 'same_family_refit'):
        same_family = results[name].str.split('-').str[0].eq(current.str.split('-').str[0])
        if not same_family.all():
            raise ValueError('A pure sibling experiment changed model family')

    # Truth is used only to tally which previously fixed candidates can recover an error.
    available = siblings.loc[siblings.candidate.eq(siblings.true_model)]
    anchor_stable = current.eq(rows.baseline)
    results['true_in_sibling_stages'] = idx.isin(available.index) & anchor_stable
    results['true_in_supported_sibling_stages'] = idx.isin(available.loc[admitted(available, pairs)].index) & anchor_stable
    results['error'] = current.ne(rows.model)
    results['family'] = rows.model.str.split('-').str[0]
    results['sibling_error'] = results.error & current.str.split('-').str[0].eq(results.family)
    results['provider_route'] = truth.get('route', 'historical')
    return results


def error_counts(frame):
    wrong = frame.loc[frame.error]
    focused = wrong.loc[wrong.sibling_error]
    return dict(rows=len(frame), errors=len(wrong), sibling_errors=len(focused), cross_family_errors=len(wrong) - len(focused),
                sibling_true_in_stages=int(focused.true_in_sibling_stages.sum()),
                sibling_true_in_supported_stages=int(focused.true_in_supported_sibling_stages.sum()))


def summarize(frame, draws):
    wrong = frame.loc[frame.error]
    return dict(**error_counts(frame), by_true_family={name: error_counts(group) for name, group in frame.groupby('family')},
                by_true_model={name: error_counts(group) for name, group in frame.groupby('model')},
                by_provider_route={name: error_counts(group) for name, group in frame.groupby('provider_route')},
                highest_remaining_sibling_confusions=[dict(model=model, incumbent=before, errors=int(n))
                    for (model, before), n in wrong.loc[wrong.sibling_error].groupby(
                    ['model', 'scored_patch']).size().sort_values(ascending=False).items()],
                versus_scored_patch=[paired_stats(frame, 'scored_patch', arm, draws) for arm in ARMS],
                versus_broad=[paired_stats(frame, 'existing_broad', arm, draws) for arm in ARMS[1:]],
                refit_vs_same_family_broad=paired_stats(frame, 'same_family_broad_head', 'same_family_refit', draws))


def run(historical, prior, gen003, kimi, inputs, independent, source, model_file, support_file, output, draws=3000):
    blocks, held = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    frames, stage, sibling = [], [], []
    for truth, trace, incumbent, _, _ in blocks:
        frames.append(truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(
            baseline=incumbent, strict=truth.get('strict_topic_holdout', False),
            scored_patch=exact_patch(truth, trace, incumbent)))
        base_actions = build_actions(truth, trace, incumbent)
        stage.append(base_actions)
        sibling.append(sibling_actions(truth, trace, incumbent, base_actions))
    rows = pd.concat(frames, verify_integrity=True)
    dev, external, separation = calibration_ids(rows, held)
    ids = dev.append(external)
    ids = ids[rows.loc[ids, 'baseline'].eq(rows.loc[ids, 'scored_patch'])]
    fit, pairs, threshold, calibration = calibrate(pd.concat(sibling), rows, ids)
    original = json.loads(model_file.read_text())
    support = json.loads(support_file.read_text())
    if (original['arm'] != 'stage_multisource' or original['version'] != 1
            or support['original'] != original):
        raise ValueError('The two archived selector controls disagree')
    traces = (historical, gen003, kimi, independent)
    targets = []
    for index in (0, 3):
        truth, trace, incumbent, _, _ = blocks[index]
        if list(trace['labels']) != original['label_order']:
            raise ValueError('Model labels do not match the archived traces')
        receipt = json.loads((traces[index] / 'glm_report.json').read_text())
        targets.append(evaluate(truth, trace, incumbent, receipt, original, support,
                                stage[index], sibling[index], frames[index], fit, pairs, threshold))
    c2 = targets[0].loc[targets[0].cohort.eq('evaluation')]
    fresh = targets[1]
    panels = dict(c2=c2, prvsiyan_895=fresh, prvsiyan_316=fresh.loc[fresh.strict])
    if list(map(len, panels.values())) != [300, 895, 316]:
        raise ValueError('Officially frozen evaluation panels changed')
    report = dict(separation=separation, calibration=calibration,
                  supported_sibling_pairs=sorted((one, other) for one, other in pairs
                                                if one < other and one.split('-')[0] == other.split('-')[0]),
                  design='Frozen broad controls versus the same stage-only evidence fitted exclusively on supported sibling actions. Pinned native/Fable protections and exact prior scored dash decisions stay in force.',
                  panels={name: {size: summarize(part, draws) for size, part in
                                 [('all', frame), ('short', frame.loc[frame.n_words.lt(100)])] if len(part)}
                          for name, frame in panels.items()})
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([c2, fresh], verify_integrity=True).to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Compared frozen broad versus pure sibling decisions beyond the exact scored Gemini repair', flush=True)
    return report
