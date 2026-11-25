"""Measure and learn from disagreements between the original ensemble members."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids


WEIGHTS = (0., .25, .50, 1.)
STAGES = ('stack_logp', 'glm_logp', 'sibling_logp')


def read_members(trace_dir, trace):
    with np.load(trace_dir / 'member_scores.npz', allow_pickle=False) as saved:
        members = dict(saved)
    if any(not np.array_equal(members.pop(key), trace[key]) for key in ('ids', 'labels')):
        raise ValueError('Original stages and ensemble members must share ID and label order')
    with np.load(trace_dir / 'member_coverage.npz', allow_pickle=False) as saved:
        masks = {key: saved[key].astype(bool) for key in members}
    shape = trace[STAGES[0]].shape
    if any(value.shape != shape or not np.isfinite(value).all() for value in members.values()):
        raise ValueError('Every member must contain finite, row-aligned class scores')
    for key, mask in masks.items():
        if mask.shape == (shape[0],):
            mask = np.broadcast_to(mask[:, None], shape)
        if mask.shape != shape:
            raise ValueError(f'Member coverage does not match scores for {key}')
        masks[key] = mask
    return members, masks


def evidence(scores, available, incumbent):
    supported = np.where(available, scores, -np.inf)
    order = np.argsort(-supported, axis=1, kind='stable')
    rank = np.argsort(order, axis=1, kind='stable') + 1
    count = available.sum(1).clip(1)
    mean = np.where(available, scores, 0.).sum(1) / count
    delta = np.where(available, scores - mean[:, None], 0.)
    scale = np.sqrt((delta ** 2).sum(1) / count).clip(.01)
    top = np.where(available.any(1), supported.max(1), 0.)
    gap = np.where(available, np.minimum((scores - top[:, None]) / scale[:, None], 0.), 0.)
    row = np.arange(len(scores))
    both = available & available[row, incumbent, None]
    versus = np.where(both, (scores - scores[row, incumbent, None]) / scale[:, None], 0.)
    vote = (rank == 1) & available
    ranks = np.where(available, 1. / rank, 0.)
    features = [gap.clip(-8.), versus.clip(-8., 8.), ranks, vote, available & ~both]
    return features, rank, vote, supported


def feature_tables(trace, baseline, members, coverage, short):
    labels = np.asarray(trace['labels'])
    n, k = trace[STAGES[0]].shape
    row = np.arange(n)
    incumbent = pd.Index(labels).get_indexer(baseline)
    if min(incumbent) < 0:
        raise ValueError('Baseline emitted an unknown label')
    base_mask = np.ones((n, k), dtype=bool)
    features, ranks, votes, rankings = {}, {}, {}, {}
    for key, value in {**{key: trace[key] for key in STAGES}, **members}.items():
        mask = coverage.get(key, base_mask)
        cols, rank, vote, supported = evidence(value, mask, incumbent)
        features[key] = cols
        ranks[key] = np.where(mask, rank, k + 1)
        votes[key], rankings[key] = vote, supported
    stage = np.where((trace['glm_winner'] | trace['fable_guard'])[:, None],
                     trace['glm_logp'], trace['sibling_logp']).copy()
    # Preserve each observed original winner, including typography transitions.
    stage[row, incumbent] = stage.max(1) + .001
    family = np.asarray([label.split('-', 1)[0] for label in labels])
    is_incumbent = np.eye(k, dtype=bool)[incumbent]
    same_family = family[None, :] == family[incumbent, None]
    base_gap = stage - stage.max(1, keepdims=True)
    base_features = [is_incumbent, same_family, is_incumbent & short[:, None],
                     same_family & short[:, None], base_gap.clip(-15., 0)]
    for key in STAGES:
        base_features.extend(features[key])
    unpacked = [item for key in members for item in features[key]]
    top_count = np.stack([votes[key] for key in members]).sum(0)
    for group in sorted({key.split('__', 1)[0] for key in members}):
        keys = [key for key in members if key.split('__', 1)[0] == group]
        unpacked.append(np.stack([votes[key] for key in keys]).mean(0))
    unpacked.extend([top_count, top_count * short[:, None],
                     np.minimum.reduce([ranks[key] for key in members]) <= 2])
    return {'stages': np.stack(base_features, axis=2), 'members': np.stack(base_features + unpacked, axis=2)}, stage, ranks, rankings


def fit_ranker(features, anchor, true_codes, fit_rows):
    x = features[fit_rows]
    offset = anchor[fit_rows]
    target = true_codes[fit_rows]
    mean = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1)).clip(.10)
    x = ((x - mean) / scale).clip(-8., 8.)
    y = np.eye(x.shape[1])[target]
    counts = np.bincount(target, minlength=x.shape[1])
    weight = 1. / counts[target]
    weight /= weight.sum()

    def objective(theta):
        logits = offset + x @ theta
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -weight @ log_prob[np.arange(len(x)), target] + .10 * (theta @ theta) / 2.
        gradient = np.einsum('ij,ijk->k', (np.exp(log_prob) - y) * weight[:, None], x)
        return loss, gradient + .10 * theta

    fit = minimize(objective, np.zeros(x.shape[2]), jac=True, method='L-BFGS-B')
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'Candidate ranker did not converge: {fit.message}')
    return ((features - mean) / scale).clip(-8., 8.) @ fit.x


def report_metrics(rows, predictions, draws, before='baseline'):
    return [dict(cohort=name, subset=subset, **paired_stats(part, before, col, draws))
            for name, group in rows.groupby('cohort')
            for subset, part in (('all', group), ('short', group[group.n_words.lt(100)]))
            if len(part) for col in predictions]


def run(trace_dir, public_truth, output, draws=5000):
    truth = read_ids(public_truth)
    trace, baseline = read_trace(trace_dir, truth)
    frozen = read_ids(trace_dir / 'frozen_predictions.parquet')
    same_ids(frozen.index, truth.index, 'Frozen unmodified baseline', ordered=True)
    if not np.array_equal(frozen.sibling_odds, baseline):
        raise ValueError('Unmodified Sibling Odds decisions do not reproduce')
    members, coverage = read_members(trace_dir, trace)
    if not members:
        raise ValueError('No ensemble member scores were exposed')
    labels = np.asarray(trace['labels'])
    row = np.arange(len(truth))
    codes = pd.Index(labels).get_indexer(truth.model)
    short = truth.n_words.lt(100).to_numpy()
    views, anchor, ranks, rankings = feature_tables(trace, baseline, members, coverage, short)

    rows = truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(baseline=baseline)
    in_stage = np.minimum.reduce([ranks[key][row, codes] for key in STAGES]) <= 3
    member_ranks = np.minimum.reduce([ranks[key][row, codes] for key in members])
    oracles = {'stages_top3': in_stage, 'members_top1': member_ranks <= 1,
               'stages_top3_and_members_top2': in_stage | (member_ranks <= 2)}
    for name, present in oracles.items():
        rows[name] = np.where(present, truth.model, baseline)

    member_summaries = {}
    for key in members:
        observed = coverage[key].any(1)
        one = labels[rankings[key].argmax(1)]
        rows[key] = np.where(observed, one, None)
        member_summaries[key] = dict(rows=int(observed.sum()),
                                     correct=int((one[observed] == truth.model.to_numpy()[observed]).sum()),
                                     rescues=int((observed & (one == truth.model) & (one != baseline)).sum()))

    dev = np.flatnonzero(rows.cohort.eq('development'))
    folds = list(GroupKFold(4).split(dev, groups=rows.prompt_id.iloc[dev]))
    proposed, selected = [], {}
    for name, x in views.items():
        correction = fit_ranker(x, anchor, codes, dev)
        for fit, valid in folds:
            correction[dev[valid]] = fit_ranker(x, anchor, codes, dev[fit])[dev[valid]]
        for weight in WEIGHTS:
            key = f'{name}_{weight:g}'
            rows[key] = labels[(anchor + weight * correction).argmax(1)]
            proposed.append(key)
        dev_scores = {weight: balanced_accuracy(rows.model.iloc[dev], rows[f'{name}_{weight:g}'].iloc[dev])
                      for weight in WEIGHTS}
        selected[name] = max(dev_scores, key=dev_scores.get)
        rows[name] = rows[f'{name}_{selected[name]:g}']
    if not all(rows[f'{key}_0'].equals(rows.baseline) for key in selected):
        raise ValueError('Zero residual changed an original prediction')
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    report = dict(dev_selected=selected, branches=member_summaries,
                  oracle=report_metrics(rows, oracles, draws),
                  paired=report_metrics(rows, [*selected, *proposed], draws),
                  incremental=report_metrics(rows, ['members'], draws, before='stages'))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print({key: report[key] for key in ('dev_selected', 'branches')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('trace', 'public-truth', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=5000)
    args = parser.parse_args()
    run(args.trace, args.public_truth, args.output, args.draws)
