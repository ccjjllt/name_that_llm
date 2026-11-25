"""Audit a Kimi/GLM cross-effort specialist on frozen Sibling Odds predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


TARGETS = ('glm-5.3', 'glm-5.3-flash', 'kimi-k3')
ARMS = ('classification', 'consistent')
WEIGHTS = (0., .15, .35, .70)
PRIMARY_WEIGHT = .35
CAP = 3.
SHORT_WORDS = 100


def weight_column(arm, weight):
    return f'{arm}_{weight:g}'


def read_experts(scores):
    experts = {}
    for arm in ARMS:
        frame = read_ids(scores / f'{arm}.parquet')
        frame = frame[list(TARGETS)].astype('float64')
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'{arm} contains nonfinite expert scores')
        experts[arm] = frame
    same_ids(experts[ARMS[0]].index, experts[ARMS[1]].index, 'Expert arm coverage')
    return experts


def read_holdout(panel_dir, trace_dir):
    queries = read_ids(panel_dir / 'queries.parquet')
    truth = read_ids(panel_dir / 'truth.parquet')
    same_ids(queries.index, truth.index, 'Frozen holdout labels', ordered=True)
    if list(queries.columns) != ['text'] or truth.empty or not truth.model.isin(TARGETS).all():
        raise ValueError('Holdout queries must be unlabeled and truth must contain target classes')
    count = queries.text.str.split().str.len()
    if not count.eq(truth.n_words).all():
        raise ValueError('Holdout text lengths differ from the frozen manifest')
    truth = truth.join(queries).assign(cohort='independent_holdout')
    trace, baseline = read_trace(trace_dir, truth)
    frozen = read_ids(trace_dir / 'frozen_predictions.parquet')
    same_ids(frozen.index, truth.index, 'Frozen holdout predictions', ordered=True)
    if not np.array_equal(frozen.dennis, trace['final']) or not np.array_equal(frozen.sibling_odds, baseline):
        raise ValueError('Independent predictions do not reproduce the frozen trace')

    audit = json.loads((trace_dir / 'body_overlap.json').read_text())
    manifest = json.loads((trace_dir / 'probe_manifest.json').read_text())
    ids_digest = hashlib.sha256(('\n'.join(truth.index) + '\n').encode()).hexdigest()
    bodies = [hashlib.sha256(text.encode()).hexdigest() for text in truth.text]
    if (manifest['rows'] != len(truth) or manifest['ordered_ids_sha256'] != ids_digest
            or manifest['input_body_sha256'] != bodies):
        raise ValueError('Independent trace manifest does not match query ids and bodies')
    known = set(audit['active_pool_ids']) | set(audit['official_train_ids'])
    if not known.issubset(truth.index):
        raise ValueError('The overlap audit reports unknown IDs')
    truth['known_baseline_corpus_overlap'] = truth.index.isin(known)
    return truth, trace, baseline, audit


def prediction_frame(trace, truth, baseline, experts, weights):
    labels = np.asarray(trace['labels'])
    target_index = pd.Index(labels).get_indexer(TARGETS)
    incumbent = pd.Index(labels).get_indexer(baseline)
    expert_incumbent = pd.Index(TARGETS).get_indexer(baseline)
    if min(target_index) < 0 or min(incumbent) < 0:
        raise ValueError('The trace must contain the expert labels and original predictions')
    glm_authoritative = trace['glm_winner']
    scores = np.where(glm_authoritative[:, None], trace['glm_logp'], trace['sibling_logp'])
    row = np.arange(len(truth))
    top_three = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    in_top_three = (top_three[:, :, None] == target_index[None, None, :]).any(1)
    in_top_three &= target_index[None, :] != incumbent[:, None]
    pre_typography = replay_predictions(trace, truth.text, typography=False)
    protected = trace['fable_guard'] | (pre_typography != baseline)
    eligible = (truth.n_words.lt(SHORT_WORDS).to_numpy() & ~protected
                & (expert_incumbent >= 0) & in_top_three.any(1))

    keep = [name for name in ('model', 'cohort', 'prompt_id', 'n_words', 'source_dataset',
                              'pair_id', 'effort', 'provider_route', 'served_model',
                              'known_baseline_corpus_overlap') if name in truth]
    result = truth[keep].assign(baseline=baseline, eligible=eligible, glm_authoritative=glm_authoritative)
    oracle_candidates = np.where(in_top_three, np.asarray(TARGETS)[None, :], '')
    result['oracle_correctable'] = eligible & (oracle_candidates == truth.model.to_numpy()[:, None]).any(1)
    # The original winner can differ from the raw score maximum. Require positive
    # expert support rather than helping a challenger that already scores higher.
    base_gap = np.maximum(0., scores[row, incumbent, None] - scores[:, target_index])
    for arm, expert in experts.items():
        if not truth.index.isin(expert.index).all():
            raise ValueError(f'{arm} is missing frozen evaluation rows')
        raw = expert.loc[truth.index, list(TARGETS)].to_numpy()
        result[arm + '_standalone'] = np.asarray(TARGETS)[raw.argmax(1)]
        evidence = np.clip(raw - raw[row, expert_incumbent, None], -CAP, CAP)
        for weight in weights:
            destination = baseline.copy()
            if weight:
                proposal = np.where(in_top_three & (evidence > 0),
                                    weight * evidence - base_gap, -np.inf)
                best = proposal.argmax(1)
                accepted = eligible & (proposal[row, best] > 0)
                destination = np.where(accepted, np.asarray(TARGETS)[best], baseline)
            result[weight_column(arm, weight)] = destination
    return result


def paired_results(rows, weights, primary, draws):
    chosen = {arm: weight_column(arm, primary) for arm in ARMS}
    for arm, name in chosen.items():
        rows[arm] = rows[name]
    fixed = []
    same_weight = []
    counts = []
    for cohort, group in rows.groupby('cohort'):
        short = group[group.n_words.lt(SHORT_WORDS)]
        slices = {'short': short, 'target_short': short[short.model.isin(TARGETS)]}
        if 'known_baseline_corpus_overlap' in short and short.known_baseline_corpus_overlap.eq(True).any():
            slices['short_without_known_baseline_overlap'] = short[~short.known_baseline_corpus_overlap.eq(True)]
        for subset, sample in slices.items():
            if sample.empty:
                continue
            context = dict(cohort=cohort, subset=subset)
            counts.append(dict(**context, rows=len(sample), eligible=int(sample.eligible.sum()),
                               oracle_correctable=int(sample.oracle_correctable.sum()),
                               truth_counts=sample.model.value_counts().to_dict()))
            for before, after in [('baseline', arm) for arm in ARMS] + [ARMS]:
                fixed.append({**context, **paired_stats(sample, before, after, draws)})
            for weight in weights:
                names = [weight_column(arm, weight) for arm in ARMS]
                for before, after in [('baseline', name) for name in names] + [tuple(names)]:
                    same_weight.append({**context, 'weight': weight, **paired_stats(sample, before, after, draws)})
    return fixed, same_weight, counts


def diagnostics(rows):
    summaries = []
    target = rows[rows.model.isin(TARGETS) & rows.n_words.lt(SHORT_WORDS)]
    for cohort, group in target.groupby('cohort'):
        slices = [('all', group)]
        for name in ('effort', 'source_dataset', 'provider_route'):
            if name in group:
                slices.extend((f'{name}={value}', part) for value, part in group.groupby(name, sort=True))
        for subset, part in slices:
            labels = part.model
            metrics = {}
            for col in ('baseline', *ARMS, *(arm + '_standalone' for arm in ARMS)):
                correct = labels.eq(part[col])
                summary = dict(rows=len(part), ba=balanced_accuracy(labels, part[col]),
                               recall=correct.groupby(labels).mean().to_dict())
                if 'pair_id' in part and part.pair_id.notna().all() and part.groupby('pair_id').size().eq(2).all():
                    correct_per_pair = correct.groupby(part.pair_id).sum()
                    agreement = part[col].groupby(part.pair_id).nunique().eq(1)
                    summary.update(pairs=len(correct_per_pair), both_correct_pairs=int(correct_per_pair.eq(2).sum()),
                                   prediction_agreement=float(agreement.mean()))
                metrics[col] = summary
            summaries.append(dict(cohort=cohort, subset=subset, truth_counts=labels.value_counts().to_dict(),
                                  scores=metrics))
    return summaries


def transitions(rows):
    results = []
    for cohort, group in rows.groupby('cohort'):
        for arm in ARMS:
            changed = group.loc[group.baseline.ne(group[arm]), ['baseline', arm]]
            if changed.empty:
                continue
            names = changed.rename(columns={'baseline': 'before', arm: 'after'})
            for window, collect in [('label', names), ('family', names.apply(lambda col: col.str.split('-').str[0]))]:
                counts = collect.value_counts().rename('rows').reset_index().to_dict(orient='records')
                results.append(dict(cohort=cohort, arm=arm, labels=window, changes=counts))
    return results


def run(scores, public_trace, output, *, weights=WEIGHTS, primary=PRIMARY_WEIGHT, draws=10000,
        holdout_panel=None, holdout_trace=None):
    weights = sorted(set(weights))
    if not weights or weights[0] != 0 or not np.isfinite(weights).all() or primary not in weights or draws < 1:
        raise ValueError('Require nonnegative finite weights including zero and the primary weight')
    if (holdout_panel is None) != (holdout_trace is None):
        raise ValueError('Provide both holdout paths or neither')
    experts = read_experts(scores)
    truth = read_ids(public_trace / 'public_truth.parquet')
    trace, baseline = read_trace(public_trace, truth)
    rows = prediction_frame(trace, truth, baseline, experts, weights)
    exposure = None
    if holdout_panel is not None:
        truth, trace, baseline, exposure = read_holdout(holdout_panel, holdout_trace)
        independent = prediction_frame(trace, truth, baseline, experts, weights)
        rows = pd.concat([rows.assign(known_baseline_corpus_overlap=False), independent], verify_integrity=True)
    paired, equal, candidates = paired_results(rows, weights, primary, draws)
    report = dict(primary_weight=primary, fusion_weights=weights, expert_score_cap=CAP,
                  short_definition=f'<{SHORT_WORDS} words', holdout_exposure=exposure,
                  candidate_coverage=candidates, paired=paired, equal_weight=equal, diagnostics=diagnostics(rows),
                  transitions=transitions(rows))
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    changed = rows[['baseline', *ARMS]].nunique(axis=1).gt(1)
    rows.loc[changed].to_parquet(output / 'changed_rows.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('scores', 'public-trace', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--holdout-panel', type=Path)
    parser.add_argument('--holdout-trace', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text()) if args.config else {}
    result = run(args.scores, args.public_trace, args.output, weights=settings.get('fusion_weights', WEIGHTS),
                 primary=settings.get('primary_weight', PRIMARY_WEIGHT), draws=args.draws,
                 holdout_panel=args.holdout_panel, holdout_trace=args.holdout_trace)
    print('Kimi/GLM fixed weight', result['primary_weight'])
