"""Compare selective candidate replacement with frozen Sibling Odds residuals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .counterfactual_selection import build_actions, fit_predict
from .heldout_source_audit import text_key
from .member_evidence import feature_tables, fit_ranker
from .member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_panel, read_trace, same_ids
from .provenance_family_replay import read_kimi


THRESHOLDS = (0., .025, .05, .10, .20, 1.)


def make_blocks(historical, prior, gen003, kimi, inputs, independent, source):
    history = read_ids(prior / 'public_truth.parquet')
    claude = read_panel(inputs / 'gen003').assign(cohort='gen003')
    raw_claude = pd.read_csv(inputs / 'gen003_trace_input.csv', dtype=str, keep_default_na=False).set_index('id')
    same_ids(raw_claude.index, claude.index, 'GEN003 trace input', ordered=True)
    if not raw_claude.text.eq(claude.text).all():
        raise ValueError('GEN003 evaluation and trace texts differ')
    pair = read_kimi(inputs / 'kimi_flash', kimi)
    if list(map(len, (history, claude, pair))) != [798, 191, 191]:
        raise ValueError('Unexpected fixed panel sizes')

    def capture(folder, truth, traced_members):
        if traced_members:
            trace, baseline, members, masks, _ = read_capture(folder, truth)
        else:
            trace, baseline = read_trace(folder, truth)
            saved = read_ids(folder / 'frozen_predictions.parquet')
            same_ids(saved.index, truth.index, 'Frozen comparison', ordered=True)
            if not (np.array_equal(saved.sibling_odds, baseline) and np.array_equal(saved.dennis, trace['final'])):
                raise ValueError('Frozen comparison calls differ')
            members, masks = {}, {}
        return truth, trace, baseline, members, masks

    held = read_prvsiyan(source, inputs / 'kimi_flash/truth.parquet')
    blocks = [capture(historical, history, True), capture(gen003, claude, False), capture(kimi, pair, False)]
    if independent:
        blocks.append(capture(independent, held, True))
    return blocks, held


def canonical_prompt(ids):
    return ids.astype('string').str.strip().str.casefold().str.replace(r'^(?:pool|prvsiyan):', '', regex=True)


def class_weights(rows, ids):
    train = rows.loc[ids]
    counts = train.groupby(['cohort', 'model']).model.transform('size')
    classes = train.groupby('cohort').model.transform('nunique')
    return 1. / (counts * classes)


def select(calls, incumbent, threshold, guarded):
    prefix = 'protected_' if guarded else ''
    return calls[prefix + 'candidate'].where(calls[prefix + 'expected_delta'].gt(threshold), incumbent)


def lock_threshold(rows, calls, ids, guarded):
    part = rows.loc[ids]
    predictions = {threshold: select(calls.loc[ids], part.baseline, threshold, guarded)
                   for threshold in THRESHOLDS}
    scores = {threshold: float(np.mean([balanced_accuracy(group.model, pred.loc[group.index])
                                       for _, group in part.groupby('cohort')]))
              for threshold, pred in predictions.items()}
    return max(THRESHOLDS, key=lambda value: (scores[value], value)), scores


def fit_arm(actions, rows, fit_ids, target_ids):
    oof = []
    train = rows.loc[fit_ids]
    canonical = canonical_prompt(train.prompt_id)
    for keep, valid in GroupKFold(4).split(train, groups=canonical):
        train_ids = train.index[keep]
        oof.append(fit_predict(actions, train_ids, train.index[valid], class_weights(rows, train_ids)))
    folds = pd.concat(oof, verify_integrity=True).loc[fit_ids]
    fitted = fit_predict(actions, fit_ids, target_ids, class_weights(rows, fit_ids))
    audit = {}
    for variant, guarded in [('open', False), ('protected', True)]:
        cutoff, scores = lock_threshold(rows, folds, fit_ids, guarded)
        audit[variant] = {'threshold': cutoff, 'fit_oof_mean_source_ba': scores}
        fitted[variant] = select(fitted, rows.loc[target_ids, 'baseline'], cutoff, guarded)
    return fitted, audit


def calibration_ids(rows, held_truth):
    c1 = rows.index[rows.cohort.eq('development')]
    external = rows.index[rows.cohort.isin(['gen003', 'kimi_flash'])]
    prompt = canonical_prompt(rows.prompt_id)
    body = text_key(rows.text)
    reserved = rows.index[rows.cohort.isin(['evaluation', 'reconstructed_stress'])]
    c2 = rows.index[rows.cohort.eq('evaluation')]
    reserved_texts = pd.concat([body.loc[reserved], text_key(held_truth.text)], ignore_index=True)
    reserved_prompts = pd.concat([prompt.loc[c2], canonical_prompt(held_truth.prompt_id)], ignore_index=True)
    usable = ~body.isin(reserved_texts) & ~prompt.isin(reserved_prompts)
    dev = c1[usable.loc[c1]]
    fit_extra = external[usable.loc[external] & ~body.loc[external].isin(body.loc[dev])]
    if not len(dev) or fit_extra.empty:
        raise ValueError('No calibration panels survive prompt and body separation')
    counts = dict(c1_available=len(c1), c1_kept=len(dev), external_available=len(external),
                  external_kept=len(fit_extra))
    return dev, fit_extra, counts


def run(historical, prior, gen003, kimi, inputs, output, independent=None, source=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    labels = np.asarray(blocks[0][1]['labels'])
    frames, features, anchors, stage_actions, member_actions = [], [], [], [], []
    for truth, trace, baseline, members, masks in blocks:
        if not np.array_equal(trace['labels'], labels):
            raise ValueError('Evaluation traces use different class orders')
        metadata = ['cohort', 'model', 'prompt_id', 'n_words', 'text']
        frame = truth[metadata].assign(baseline=baseline, strict=truth.get('strict_topic_holdout', False))
        frames.append(frame)
        views, anchor, _, _ = feature_tables(trace, baseline, {}, {}, truth.n_words.lt(100).to_numpy())
        anchors.append(anchor)
        features.append(views['stages'])
        stage_actions.append(build_actions(truth, trace, baseline))
        if members:
            member_actions.append(build_actions(truth, trace, baseline, members, masks))
    rows = pd.concat(frames, verify_integrity=True)
    c2 = rows.index[rows.cohort.eq('evaluation')]
    held = rows.index[rows.cohort.eq('prvsiyan')]
    target = c2.append(held)
    prompt = canonical_prompt(rows.prompt_id)
    dev, fit_extra, separation = calibration_ids(rows, held_truth)

    matrix = np.concatenate(features)
    anchor = np.concatenate(anchors)
    where = rows.index.get_indexer(dev)
    codes = pd.Index(labels).get_indexer(rows.model)
    residual = fit_ranker(matrix, anchor, codes, where)
    rows['stage_residual_050'] = labels[(anchor + .50 * residual).argmax(axis=1)]
    if not np.array_equal(labels[anchor.argmax(axis=1)], rows.baseline):
        raise ValueError('Zero residual failed exact original reproduction')

    stages = pd.concat(stage_actions)
    members = pd.concat(member_actions)
    specs = {'stage_c1': (stages, dev), 'stage_multisource': (stages, dev.append(fit_extra)),
             'member_c1': (members, dev)}
    summaries, out_columns, risks = {}, [], []
    for arm, (actions, fit_ids) in specs.items():
        fitted, summaries[arm] = fit_arm(actions, rows, fit_ids, target)
        summaries[arm]['fit_counts'] = rows.loc[fit_ids].cohort.value_counts().to_dict()
        for variant in ('open', 'protected'):
            result = f'{arm}_{variant}'
            rows.loc[target, result] = fitted[variant]
            out_columns.append(result)
        risk_columns = ['candidate', 'expected_delta', 'protected_candidate', 'protected_expected_delta']
        risks.append(fitted[risk_columns].add_prefix(f'{arm}_'))
        print('Locked', arm, summaries[arm], flush=True)

    # Check one source at a time with only the other two sources available.
    leave_source = {}
    combined = dev.append(fit_extra)
    for cohort, group in rows.loc[combined].groupby('cohort'):
        other = combined.difference(group.index)
        same_prompt = prompt.loc[other].isin(prompt.loc[group.index])
        other = other[~same_prompt]
        fitted, fit_report = fit_arm(stages, rows, other, group.index)
        evaluated = group.join(fitted[['open', 'protected']])
        leave_source[cohort] = {
            'training': fit_report,
            'comparisons': [paired_stats(evaluated, 'baseline', variant, draws)
                            for variant in ('open', 'protected')],
        }
    result_scopes = {'c2': c2, 'prvsiyan_895': held, 'prvsiyan_316': held[rows.loc[held, 'strict']]}
    overlaps = {}
    if independent:
        overlaps = json.loads((independent / 'body_overlap.json').read_text())
        observed = {row for key in EXPOSURE_KEYS for row in overlaps.get(key, [])}
        if not observed.issubset(held):
            raise ValueError('Runtime corpus audit mentions unknown holdout IDs')
        if observed:
            for name in ('prvsiyan_895', 'prvsiyan_316'):
                result_scopes[name + '_no_runtime_match'] = result_scopes[name].difference(observed)
    results = {}
    comparisons = [('baseline', after) for after in ('stage_residual_050', *out_columns)]
    comparisons.extend(('stage_residual_050', after) for after in out_columns)
    comparisons.extend((f'stage_c1_{variant}', f'{variant_arm}_{variant}')
                       for variant in ('open', 'protected') for variant_arm in ('stage_multisource', 'member_c1'))
    for name, index in result_scopes.items():
        scope = rows.loc[index]
        if scope.empty:
            continue
        for length, part in [('all', scope), ('short', scope[scope.n_words.lt(100)])]:
            if len(part):
                results[f'{name}_{length}'] = [paired_stats(part, before, after, draws)
                                                for before, after in comparisons]
    report = dict(fitted=summaries, leave_source=leave_source, panels=results, separated=separation,
                  thresholds=THRESHOLDS, exact_zero_weight=True, runtime_exposure=overlaps)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows.loc[target].drop(columns='text').join(pd.concat(risks, axis=1)).to_parquet(output / 'predictions.parquet')
    return report
