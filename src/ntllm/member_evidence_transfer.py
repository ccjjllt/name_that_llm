"""Fit the shared member ranker on C1 and measure untouched Prvsiyan transfer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .member_evidence import STAGES, WEIGHTS, feature_tables, fit_ranker, read_members
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids


SOURCE = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'
EXPOSURE_KEYS = ('official_train_ids', 'active_pool_ids', 'full_pool_ids', 'stack_extra_ids')


def read_prvsiyan(source_audit, prior_index):
    queries = read_ids(source_audit / 'candidate_queries.parquet')
    truth = read_ids(source_audit / 'candidate_truth.parquet')
    same_ids(queries.index, truth.index, 'Audited independent truth', ordered=True)
    if list(queries) != ['text']:
        raise ValueError('The independent query artifact must have only body and ID')
    truth = truth.loc[truth.source_dataset.eq(SOURCE + '/13')].join(queries)
    expected = truth.n_words.eq(truth.text.str.split().str.len())
    if len(truth) != 895 or truth.model.nunique() != 18 or not expected.all():
        raise ValueError('The fixed 895-row independent source changed')
    prior = read_ids(prior_index)
    previous_prompts = prior.loc[prior.source_dataset.eq(SOURCE), 'prompt_id']
    strict = ~truth.prompt_id.isin(previous_prompts)
    if strict.sum() != 316 or truth.loc[strict, 'prompt_id'].nunique() != 23:
        raise ValueError('The 316-row independent-topic subset changed')
    return truth.assign(cohort='prvsiyan', strict_topic_holdout=strict)


def verify_manifest(path, truth):
    manifest = json.loads(path.read_text())
    id_hash = hashlib.sha256(('\n'.join(truth.index) + '\n').encode()).hexdigest()
    body_hash = truth.text.map(lambda passage: hashlib.sha256(passage.encode()).hexdigest())
    if (manifest['rows'] != len(truth) or manifest['ordered_ids_sha256'] != id_hash
            or manifest['input_body_sha256'] != body_hash.tolist()):
        raise ValueError('Trace manifest does not match ordered audit IDs and plaintext bodies')


def read_capture(folder, truth):
    trace, baseline = read_trace(folder, truth)
    verify_manifest(folder / 'probe_manifest.json', truth)
    frozen = read_ids(folder / 'frozen_predictions.parquet')
    same_ids(frozen.index, truth.index, 'Original replay', ordered=True)
    if (not np.array_equal(frozen.dennis, trace['final'])
            or not np.array_equal(frozen.sibling_odds, baseline)):
        raise ValueError('Original replay does not match the frozen output')
    members, coverage = read_members(folder, trace)
    with np.load(folder / 'member_coverage.npz', allow_pickle=False) as bundle:
        if any(not np.array_equal(bundle[key], trace[key]) for key in ('ids', 'labels')):
            raise ValueError('The member coverage artifact has a different ID or class order')
    kinds = json.loads((folder / 'member_manifest.json').read_text())
    if not members or set(kinds) != set(members):
        raise ValueError('The member manifest and scores have different branch coverage')
    return trace, baseline, members, coverage, kinds


def align_members(historical, held, coverage, shape):
    scores = {key: held[key] if key in held else np.zeros(shape, dtype=np.float32) for key in historical}
    available = {key: coverage[key] if key in coverage else np.zeros(shape, dtype=bool) for key in historical}
    return scores, available


def fit_c1(historical, held, truth, anchor, labels):
    dev = np.flatnonzero(truth.cohort.eq('development'))
    if len(dev) != 180 or truth.cohort.eq('evaluation').sum() != 300:
        raise ValueError('Expected the original C1 development and C2 evaluation cohorts')
    folds = list(GroupKFold(4).split(dev, groups=truth.prompt_id.iloc[dev]))
    historical_codes = pd.Index(labels).get_indexer(truth.model)
    codes = np.r_[historical_codes, np.full(len(anchor) - len(truth), -1)]
    selected, dev_ba, predictions = {}, {}, {}
    for key, hist in historical.items():
        values = np.concatenate([hist, held[key]])
        correction = fit_ranker(values, anchor, codes, dev)
        for fit, valid in folds:
            correction[dev[valid]] = fit_ranker(values, anchor, codes, dev[fit])[dev[valid]]
        for weight in WEIGHTS:
            col = f'{key}_{weight:g}'
            predictions[col] = labels[(anchor + weight * correction).argmax(1)]
        scores = {weight: balanced_accuracy(truth.model.iloc[dev], predictions[f'{key}_{weight:g}'][dev])
                  for weight in WEIGHTS}
        selected[key] = max(scores, key=scores.get)
        dev_ba[key] = scores
        predictions[key] = predictions[f'{key}_{selected[key]:g}']
    return selected, dev_ba, predictions


def minimum_rank(members, masks, codes):
    n_labels = next(iter(members.values())).shape[1]
    minimum = np.full(len(codes), n_labels + 1)
    row = np.arange(len(codes))
    for key, values in members.items():
        ordered = np.argsort(-np.where(masks[key], values, -np.inf), axis=1, kind='stable')
        ranks = np.argsort(ordered, axis=1, kind='stable')[row, codes] + 1
        rankable = masks[key][row, codes]
        minimum[rankable] = np.minimum(minimum[rankable], ranks[rankable])
    return minimum


def oracle_calls(truth, baseline, trace, ranks, common_keys, members, masks):
    row = np.arange(len(truth))
    codes = pd.Index(trace['labels']).get_indexer(truth.model)
    stage_top3 = np.minimum.reduce([ranks[key][row, codes] for key in STAGES]) <= 3
    common = np.minimum.reduce([ranks[key][row, codes] for key in common_keys])
    exposed = minimum_rank(members, masks, codes)
    recoveries = {'stages_top3': stage_top3, 'members_top1': common <= 1,
                  'stages_top3_and_members_top2': stage_top3 | (common <= 2),
                  'stages_top3_and_all_observed_top2': stage_top3 | (exposed <= 2)}
    return {name: np.where(present, truth.model, baseline) for name, present in recoveries.items()}


def class_oracles(rows, names):
    result = []
    for label, group in rows.groupby('model'):
        correct = int(group.baseline.eq(label).sum())
        oracle = {name: int(group[name].eq(label).sum()) for name in names}
        result.append({'model': label, 'rows': len(group), 'baseline_correct': correct, 'oracle_correct': oracle,
                       'oracle_rescues': {name: count - correct for name, count in oracle.items()}})
    return result


def runtime_comparison(reference, actual, before, after, rows, draws):
    if not np.array_equal(reference['labels'], actual['labels']):
        raise ValueError('The original historical trace used a different class order')
    scored = rows.assign(prior_runtime=before, member_trace_runtime=after)
    c2 = scored[scored.cohort.eq('evaluation')]
    return {'baseline_changes': int((before != after).sum()),
            'changes_by_cohort': pd.Series(before != after, index=rows.index).groupby(rows.cohort).sum().to_dict(),
            'c2_paired': paired_stats(c2, 'prior_runtime', 'member_trace_runtime', draws),
            'c2_short_paired': paired_stats(c2[c2.n_words.lt(100)], 'prior_runtime', 'member_trace_runtime', draws),
            'stages': {key: {'max_abs_score_change': float(np.abs(reference[key] - actual[key]).max()),
                             'argmax_changes': int((reference[key].argmax(1) != actual[key].argmax(1)).sum())}
                       for key in STAGES}}


def branch_summary(coverage, rows):
    return {key: dict(rows=int(mask[rows].any(axis=1).sum()), classes_scored=int(mask[rows].sum()))
            for key, mask in coverage.items()}


def run(historical_trace, original_trace, independent_trace, source_audit, prior_kimi, output, draws=5000):
    from .member_agreement import analyze_agreement, build_actions

    historical = read_ids(original_trace / 'public_truth.parquet')
    independent = read_prvsiyan(source_audit, prior_kimi)
    if not historical.index.intersection(independent.index).empty:
        raise ValueError('Independent IDs overlap historical examples')
    hist, hist_base, hist_members, hist_masks, hist_kinds = read_capture(historical_trace, historical)
    held, held_base, held_members, held_masks, held_kinds = read_capture(independent_trace, independent)
    original, original_base = read_trace(original_trace, historical)
    if not np.array_equal(hist['labels'], held['labels']):
        raise ValueError('Historical and independent trace class order differs')
    common = hist_members.keys() & held_members.keys()
    if any(hist_kinds[key]['kind'] != held_kinds[key]['kind'] for key in common):
        raise ValueError('A shared branch changed its score type')
    scores, masks = align_members(hist_members, held_members, held_masks, held[STAGES[0]].shape)
    hist_views, hist_anchor, hist_ranks, _ = feature_tables(
        hist, hist_base, hist_members, hist_masks, historical.n_words.lt(100).to_numpy())
    held_views, held_anchor, held_ranks, _ = feature_tables(
        held, held_base, scores, masks, independent.n_words.lt(100).to_numpy())
    anchor = np.concatenate([hist_anchor, held_anchor])
    selected, dev_ba, predictions = fit_c1(hist_views, held_views, historical, anchor, hist['labels'])
    hist_oracle = oracle_calls(historical, hist_base, hist, hist_ranks, hist_members, hist_members, hist_masks)
    held_oracle = oracle_calls(independent, held_base, held, held_ranks, hist_members, held_members, held_masks)
    evidence = {name: np.r_[call, held_oracle[name]] for name, call in hist_oracle.items()}
    columns = ['cohort', 'model', 'prompt_id', 'n_words']
    rows = pd.concat([historical[columns].assign(strict_topic_holdout=False),
                      independent[[*columns, 'strict_topic_holdout']]], verify_integrity=True)
    rows = rows.assign(baseline=np.r_[hist_base, held_base], **predictions, **evidence)
    rows['prior_historical_baseline'] = pd.Series(original_base, index=historical.index)
    if not all(rows[f'{key}_0'].equals(rows.baseline) for key in selected):
        raise ValueError('Zero residual did not reproduce the exact original decisions')
    exposures = json.loads((independent_trace / 'body_overlap.json').read_text())
    mandatory = EXPOSURE_KEYS if 'infer__comp_blend' in held_members else EXPOSURE_KEYS[:-1]
    if any(key not in exposures for key in mandatory):
        raise ValueError('Runtime corpus audit is missing a required exposure check')
    exposed = {id_ for key in EXPOSURE_KEYS for id_ in exposures.get(key, [])}
    if not exposed.issubset(independent.index):
        raise ValueError('Runtime corpus audit found unknown independent IDs')
    rows['runtime_corpus_overlap'] = rows.index.isin(exposed)
    subsets = {'c2': rows[rows.cohort.eq('evaluation')],
               'prvsiyan_895': rows[rows.cohort.eq('prvsiyan')],
               'prvsiyan_316': rows[rows.strict_topic_holdout]}
    if exposed:
        for name in ('prvsiyan_895', 'prvsiyan_316'):
            panel = subsets[name]
            subsets[name + '_no_runtime_match'] = panel.loc[~panel.runtime_corpus_overlap]
    historical_actions = build_actions(historical, hist, hist_base,
                                       {key: hist_members[key] for key in common}, hist_masks)
    held_actions = build_actions(independent, held, held_base,
                                 {key: held_members[key] for key in common}, held_masks)
    rows = rows.join(pd.concat([historical_actions, held_actions], verify_integrity=True))
    agreement = analyze_agreement(rows, {name: panel.index for name, panel in subsets.items()}, draws)
    results = []
    for name, panel in subsets.items():
        for length, part in [('all', panel), ('short', panel[panel.n_words.lt(100)])]:
            if part.empty:
                continue
            results.append(dict(panel=name, length=length, rows=len(part),
                                paired=[paired_stats(part, 'baseline', key, draws) for key in predictions],
                                members_vs_stages=paired_stats(part, 'stages', 'members', draws),
                                oracle=[paired_stats(part, 'baseline', key, draws) for key in evidence],
                                oracle_by_true_class=class_oracles(part, evidence)))
    reports = {}
    for group, folder in [('prior', original_trace), ('historical', historical_trace), ('independent', independent_trace)]:
        receipt = json.loads((folder / 'glm_report.json').read_text())
        reports[group] = {key: receipt.get(key) for key in ('status', 'n_texts', 'n_selected', 'n_blended', 's10_s', 'total_s')}
    report = dict(dev_selected=selected, dev_ba_by_weight=dev_ba, zero_weight_matches_all_rows=True,
                  fit_rows=len(historical.loc[historical.cohort.eq('development')]),
                  missing_transfer_members=sorted(hist_members.keys() - held_members.keys()),
                  transfer_only_members=sorted(held_members.keys() - hist_members.keys()),
                  runtime_exposure=exposures, historical_runtime_divergence=runtime_comparison(
                      original, hist, original_base, hist_base, historical, draws), glm_runtime=reports,
                  member_coverage={
                      'c2': branch_summary(hist_masks, historical.cohort.eq('evaluation').to_numpy()),
                      'prvsiyan_895': branch_summary(held_masks, np.ones(len(independent), dtype=bool)),
                      'prvsiyan_316': branch_summary(held_masks, independent.strict_topic_holdout.to_numpy()),
                  }, summaries=results, agreement=agreement)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Historical C1 weights', selected, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('historical-trace', 'original-trace', 'independent-trace', 'source-audit', 'prior-kimi', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=5000)
    run(**vars(parser.parse_args()))
