"""Replay a two-candidate expert on frozen Sibling Odds predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


TERRA = 'gpt-5.6-terra'
GPT55 = 'gpt-5.5'
TARGETS = (TERRA, GPT55)
ARMS = ('classification', 'pairwise')
WEIGHTS = (0., .15, .35, .70)
SHORT_WORDS = 100
MARGIN_CAP = 3.


def weight_column(arm, weight):
    return f'{arm}_{weight:g}'


def read_margins(directory):
    margins = {}
    for arm in ARMS:
        scores = read_ids(directory / f'{arm}.parquet')
        margin = scores['margin'].astype('float64')
        if not np.isfinite(margin.to_numpy()).all():
            raise ValueError(f'{arm} has invalid expert margins')
        margins[arm] = margin
    same_ids(margins[ARMS[0]].index, margins[ARMS[1]].index, 'Expert arm score coverage')
    return margins


def prediction_frame(trace, truth, baseline, margins, weights):
    labels = pd.Index(trace['labels'])
    terra, gpt55 = labels.get_indexer(TARGETS)
    if min(terra, gpt55) < 0:
        raise ValueError('The trace must contain both expert candidates')

    scores = trace['sibling_logp']
    base_margin = scores[:, terra] - scores[:, gpt55]
    top_three = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    both_in_top_three = (top_three == terra).any(axis=1) & (top_three == gpt55).any(axis=1)
    protected = trace['glm_winner'] | trace['fable_guard']
    pre_typography = replay_predictions(trace, truth.text, typography=False)
    short = truth.n_words.lt(SHORT_WORDS)
    eligible = (short.to_numpy() & np.isin(baseline, TARGETS) & both_in_top_three
                & ~protected & (pre_typography == baseline))

    keep = [column for column in ('model', 'cohort', 'prompt_id', 'n_words', 'source_dataset', 'pair_id')
            if column in truth]
    result = truth[keep].assign(baseline=baseline, eligible=eligible, baseline_margin=base_margin)
    for arm, margin in margins.items():
        if not truth.index[short].isin(margin.index).all():
            raise ValueError(f'{arm} is missing short evaluation rows')
        raw = margin.reindex(truth.index).to_numpy()
        expert = np.clip(raw, -MARGIN_CAP, MARGIN_CAP)
        result[arm + '_margin'] = raw
        for weight in weights:
            if weight == 0:
                result[weight_column(arm, weight)] = baseline
                continue
            fused = base_margin + weight * expert
            proposal = np.where(fused > 0, TERRA, GPT55)
            # A proposal needs directional expert evidence for that label.
            support = np.where(proposal == TERRA, expert > 0, expert < 0)
            update = eligible & support & (fused != 0)
            result[weight_column(arm, weight)] = np.where(update, proposal, baseline)
    return result


def choose_weights(rows, weights):
    dev = rows.loc[rows.cohort.eq('development') & rows.n_words.lt(SHORT_WORDS)]
    if dev.empty:
        raise ValueError('The C1 development cohort must contain short passages')
    values = {
        arm: {weight: balanced_accuracy(dev.model, dev[weight_column(arm, weight)]) for weight in weights}
        for arm in ARMS
    }
    selected = {arm: max(weights, key=by_weight.get) for arm, by_weight in values.items()}
    return selected, values


def read_holdout(panel_dir, trace_dir):
    queries = read_ids(panel_dir / 'queries.parquet')
    truth = read_ids(panel_dir / 'truth.parquet')
    same_ids(queries.index, truth.index, 'Frozen holdout labels', ordered=True)
    if list(queries.columns) != ['text'] or set(truth.model) != set(TARGETS):
        raise ValueError('The holdout must have unlabeled queries and both target labels')
    if not queries.text.str.split().str.len().eq(truth.n_words).all() or not truth.n_words.lt(SHORT_WORDS).all():
        raise ValueError('The holdout text length differs from its frozen manifest')
    frozen = read_ids(trace_dir / 'frozen_predictions.parquet')
    same_ids(frozen.index, truth.index, 'Frozen independent predictions', ordered=True)
    truth = truth.join(queries).assign(cohort='independent_holdout')
    trace, baseline = read_trace(trace_dir, truth)
    if not np.array_equal(frozen.dennis, trace['final']) or not np.array_equal(frozen.sibling_odds, baseline):
        raise ValueError('The independent frozen predictions do not reproduce the new trace')

    manifest = json.loads((trace_dir / 'probe_manifest.json').read_text())
    ordered_digest = hashlib.sha256(('\n'.join(truth.index) + '\n').encode()).hexdigest()
    body_digests = [hashlib.sha256(text.encode()).hexdigest() for text in truth.text]
    if (manifest['rows'] != len(truth) or manifest['ordered_ids_sha256'] != ordered_digest
            or manifest['input_body_sha256'] != body_digests):
        raise ValueError('Independent trace manifest does not match the frozen panel')
    audit = json.loads((trace_dir / 'body_overlap.json').read_text())
    exposure = set(audit['active_pool_ids']) | set(audit['official_train_ids'])
    if not exposure.issubset(truth.index):
        raise ValueError('The independent overlap audit reports unknown IDs')
    truth = truth.assign(known_baseline_corpus_overlap=truth.index.isin(exposure))
    return truth, trace, baseline, audit


def standalone_scores(rows):
    """Score the same binary specialist margins without using the base decisions."""
    output = {}
    target = rows[rows.model.isin(TARGETS) & rows.n_words.lt(SHORT_WORDS)]
    for cohort, group in target.groupby('cohort'):
        output[cohort] = {}
        for arm in ARMS:
            margin = group[arm + '_margin']
            labels = np.where(margin > 0, TERRA, GPT55)
            recall = group.model.eq(labels).groupby(group.model).mean().to_dict()
            details = dict(rows=len(group), ba=balanced_accuracy(group.model, labels), recall=recall)
            if 'pair_id' in group and group.pair_id.notna().all():
                pairs = group.assign(margin=margin).pivot(index='pair_id', columns='model', values='margin')
                difference = (pairs[TERRA] - pairs[GPT55]).dropna()
                if len(difference):
                    details['matched_pairs'] = len(difference)
                    details['matched_order_accuracy'] = float(
                        difference.gt(0).mean() + .5 * difference.eq(0).mean()
                    )
            output[cohort][arm] = details
    return output


def comparisons(rows, selected, weights, draws):
    for arm, weight in selected.items():
        rows[arm] = rows[weight_column(arm, weight)]
    outputs = []
    equal_weight = []
    candidates = []
    for cohort, group in rows.groupby('cohort'):
        short = group[group.n_words.lt(SHORT_WORDS)]
        if short.empty:
            continue
        slices = {'short': short, 'target_short': short[short.model.isin(TARGETS)]}
        if 'known_baseline_corpus_overlap' in short and short.known_baseline_corpus_overlap.eq(True).any():
            slices['short_without_known_baseline_overlap'] = short[
                ~short.known_baseline_corpus_overlap.eq(True)
            ]
        for name, subset in slices.items():
            if subset.empty:
                continue
            context = dict(cohort=cohort, subset=name)
            expert_correctable = subset.eligible & subset.model.isin(TARGETS) & subset.baseline.ne(subset.model)
            oracle = subset.baseline.where(~expert_correctable, subset.model)
            candidates.append(dict(**context, rows=len(subset), eligible=int(subset.eligible.sum()),
                                   oracle_correctable=int(expert_correctable.sum()),
                                   oracle_ba_delta_pp=100 * (
                                       balanced_accuracy(subset.model, oracle)
                                       - balanced_accuracy(subset.model, subset.baseline)
                                   ),
                                   truth_counts=subset.model.value_counts().to_dict()))
            for before, after in [('baseline', arm) for arm in ARMS] + [ARMS]:
                outputs.append({**context, **paired_stats(subset, before, after, draws)})
            for weight in weights:
                columns = [weight_column(arm, weight) for arm in ARMS]
                equal_weight.extend([
                    {**context, 'weight': weight, **paired_stats(subset, 'baseline', columns[0], draws)},
                    {**context, 'weight': weight, **paired_stats(subset, 'baseline', columns[1], draws)},
                    {**context, 'weight': weight, **paired_stats(subset, *columns, draws)},
                ])
    return outputs, equal_weight, candidates


def run(scores, public_trace, output, weights=WEIGHTS, draws=10000, holdout_panel=None, holdout_trace=None):
    weights = sorted(set(weights))
    if not weights or weights[0] != 0 or not np.isfinite(weights).all() or draws < 1:
        raise ValueError('Require nonnegative finite weights including zero and positive bootstrap draws')
    if (holdout_panel is None) != (holdout_trace is None):
        raise ValueError('Provide both holdout paths or neither')
    margins = read_margins(scores)
    public = read_ids(public_trace / 'public_truth.parquet')
    trace, baseline = read_trace(public_trace, public)
    rows = prediction_frame(trace, public, baseline, margins, weights)
    selected, development = choose_weights(rows, weights)
    exposure = None
    if holdout_panel is not None:
        holdout, trace, baseline, exposure = read_holdout(holdout_panel, holdout_trace)
        independent = prediction_frame(trace, holdout, baseline, margins, weights)
        independent['known_baseline_corpus_overlap'] = holdout.known_baseline_corpus_overlap
        rows = pd.concat([rows.assign(known_baseline_corpus_overlap=False), independent], verify_integrity=True)
    paired, equal, candidates = comparisons(rows, selected, weights, draws)
    result = dict(c1_selected_weights=selected, c1_short_ba_by_weight=development,
                  fusion_weights=weights, expert_margin_cap=MARGIN_CAP, short_definition=f'<{SHORT_WORDS} words',
                  holdout_exposure=exposure, candidate_coverage=candidates, paired=paired, equal_weight=equal,
                  standalone=standalone_scores(rows))
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    changed = rows[['baseline', *ARMS]].nunique(axis=1).gt(1)
    rows.loc[changed].to_parquet(output / 'changed_rows.parquet')
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


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
    result = run(args.scores, args.public_trace, args.output, settings.get('weights', WEIGHTS),
                 args.draws, args.holdout_panel, args.holdout_trace)
    print('C1 selected weights', result['c1_selected_weights'])
