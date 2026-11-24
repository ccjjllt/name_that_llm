"""Compare the next-token and multi-position readers on frozen Sibling Odds scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .peer_context_report import (
    balanced_accuracy,
    paired_stats,
    read_heads,
    read_ids,
    read_panel,
    read_trace,
    same_ids,
    same_meta,
    standalone_results,
)
from .sibling_odds_replay import replay_predictions


ARMS = ('next_token', 'multi_position')
SHORT_WORDS = 100
SCORE_CAP = 3.0


def check_manifest(manifest, public, independent):
    for partition, expected in [('trace', public), ('blind', independent)]:
        actual = manifest[manifest.partition.eq(partition)]
        same_ids(actual.index, expected.index, partition, ordered=partition == 'blind')
        columns = ['model', 'prompt_id', 'n_words']
        if partition == 'trace':
            columns.append('cohort')
        same_meta(actual, expected, columns, partition)
    official = manifest[manifest.partition.eq('official')]
    if set(official.fold) != set(range(5)) or not official.groupby('prompt_id').fold.nunique().eq(1).all():
        raise ValueError('Official prompts must have one of five held-out fold assignments')


def replay_arm(trace, truth, logits, weights, baseline):
    scores = logits.loc[truth.index, trace['labels']].to_numpy(dtype=np.float64)
    centered = np.clip(scores - scores.mean(axis=1, keepdims=True), -SCORE_CAP, SCORE_CAP)
    predictions = {}
    for weight in weights:
        fused = {**trace, 'sibling_logp': trace['sibling_logp'] + weight * centered}
        predictions[weight] = replay_predictions(fused, truth.text)
    if not np.array_equal(predictions[0.0], baseline):
        raise ValueError('Weight zero did not reproduce the frozen Sibling Odds predictions')
    return predictions


def select_weights(public, predictions, weights):
    development = public.cohort.eq('development').to_numpy()
    if not development.any():
        raise ValueError('The C1 development cohort is required for selecting fusion weights')
    truth = public.model.to_numpy()[development]
    values = {
        arm: {w: balanced_accuracy(truth, by_weight[w][development]) for w in weights}
        for arm, by_weight in predictions.items()
    }
    selected = {arm: max(weights, key=by_weight.get) for arm, by_weight in values.items()}
    return selected, values


def score_panel(truth, baseline, predictions, selected, weights, draws):
    rows = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(baseline=baseline)
    for arm, by_weight in predictions.items():
        for weight in weights:
            rows[f'{arm}_{weight}'] = by_weight[weight]
        rows[arm] = by_weight[selected[arm]]

    locked_pairs = [('baseline', arm) for arm in ARMS] + [ARMS]
    result = []
    equal_weight = []
    for cohort, group in rows.groupby('cohort'):
        for subset, part in [('all', group), ('short', group[group.n_words.lt(SHORT_WORDS)])]:
            if not len(part):
                continue
            context = dict(cohort=cohort, subset=subset)
            for before, after in locked_pairs:
                result.append({**context, **paired_stats(part, before, after, draws)})
            for weight in weights:
                before, after = (f'{arm}_{weight}' for arm in ARMS)
                equal_weight.append({**context, 'weight': weight, **paired_stats(part, before, after, draws)})
    return rows, result, equal_weight


def run(gpu, public_trace, independent_trace, independent_panel, config, output, draws):
    if draws < 1:
        raise ValueError('At least one prompt bootstrap draw is required')
    weights = sorted({float(value) for value in json.loads(config.read_text())['weights']})
    if not weights or weights[0] != 0 or not np.isfinite(weights).all():
        raise ValueError('Fusion weights must be finite, nonnegative, and include zero')

    manifest = read_ids(gpu / 'manifest.parquet')
    public = read_ids(public_trace / 'public_truth.parquet')
    blind = read_panel(independent_panel).assign(cohort='gen003_sparse')
    check_manifest(manifest, public, blind)
    public_scores, public_base = read_trace(public_trace, public)
    blind_scores, blind_base = read_trace(independent_trace, blind)
    if not np.array_equal(public_scores['labels'], blind_scores['labels']):
        raise ValueError('Trace label orders differ')

    frozen = read_ids(independent_trace / 'frozen_predictions.parquet')
    same_ids(frozen.index, blind.index, 'Frozen Sibling Odds predictions', ordered=True)
    if not np.array_equal(blind_base, frozen.sibling_odds) or not np.array_equal(blind_scores['final'], frozen.dennis):
        raise ValueError('The independent baseline differs from the previous Sibling Odds replay')
    exposure = json.loads((independent_trace / 'body_overlap.json').read_text())
    known = set(exposure['active_pool_ids']) | set(exposure['official_train_ids'])
    if not known.issubset(blind.index):
        raise ValueError('The GEN003 corpus overlap audit contains IDs outside the frozen panel')

    heads = read_heads(gpu, manifest, public_scores['labels'], arms=ARMS)
    public_predictions = {arm: replay_arm(public_scores, public, scores, weights, public_base)
                          for arm, scores in heads.items()}
    selected, development_ba = select_weights(public, public_predictions, weights)
    blind_predictions = {arm: replay_arm(blind_scores, blind, scores, weights, blind_base)
                         for arm, scores in heads.items()}
    public_rows, public_stats, public_equal = score_panel(
        public, public_base, public_predictions, selected, weights, draws)
    blind_rows, blind_stats, blind_equal = score_panel(
        blind, blind_base, blind_predictions, selected, weights, draws)
    standalone, _ = standalone_results(manifest, heads)

    rows = pd.concat([public_rows, blind_rows], verify_integrity=True)
    rows['known_baseline_corpus_overlap'] = rows.index.isin(known)
    changed = rows[['baseline', *ARMS]].nunique(axis=1).gt(1)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    rows.loc[changed].to_parquet(output / 'changed_rows.parquet')
    report = dict(
        c1_selected_weights=selected,
        c1_ba_by_weight=development_ba,
        fusion=dict(score_cap=SCORE_CAP, weights=weights, weight_selection='maximum full C1 BA, ties to lower weight'),
        short_definition=f'<{SHORT_WORDS} words',
        prompt_bootstrap_draws=draws,
        gen003_exposure=exposure,
        paired=public_stats + blind_stats,
        equal_weight=public_equal + blind_equal,
        standalone=standalone,
    )
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('gpu', 'public-trace', 'independent-trace', 'independent-panel', 'config', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    run(args.gpu, args.public_trace, args.independent_trace, args.independent_panel, args.config, args.output, args.draws)
