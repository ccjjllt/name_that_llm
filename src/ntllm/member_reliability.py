"""Measure whether branch confusion models add portable evidence beyond score ranks."""

import json

import numpy as np
import pandas as pd

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import STAGES, member_family
from .counterfactual_selection_audit import canonical_prompt, class_weights
from .heldout_source_audit import text_key
from .member_evidence_transfer import read_capture, read_prvsiyan
from .peer_context_report import balanced_accuracy, paired_stats, read_ids
from .sibling_odds_replay import replay_predictions
from .support_censored_selector import prompt_folds


WEIGHTS = (0., .25, .5, 1., 2., 4.)
METHODS = ('ranks', 'pooled_confusion', 'class_confusion')
PRIOR_SIZE = 12.


def groups_for(records):
    common = set.intersection(*[set(v[2]) for v in records])
    groups = {}
    for key in sorted(common):
        kinds = {item[4][key]['kind'] for item in records}
        if len(kinds) != 1:
            raise ValueError(f'{key} has different score representations across captures')
        if all(mask[key].all() for _, _, _, mask, _ in records):
            groups.setdefault(member_family(key), []).append(key)
    if len(groups) < 3:
        raise ValueError('Expected multiple distinct branch architecture groups')
    return groups


def pool_rankings(trace, members, groups, baseline):
    labels = np.asarray(trace['labels'])
    anchor = pd.Index(labels).get_indexer(baseline)
    if (anchor < 0).any():
        raise ValueError('Unexpected original label')
    n, k = trace['stack_logp'].shape
    selected = np.asarray(trace['glm_winner'] | trace['fable_guard'], dtype=bool)
    score_anchor = np.where(selected[:, None], trace['glm_logp'], trace['sibling_logp']).astype(float)
    score_anchor -= score_anchor.max(axis=1, keepdims=True)
    score_anchor[np.arange(n), anchor] = 1e-6
    available = np.zeros((n, k), dtype=bool)
    by_row = np.arange(n)[:, None]
    available[by_row, anchor[:, None]] = True
    for stage in STAGES:
        available[by_row, np.argsort(-trace[stage], axis=1, kind='stable')[:, :3]] = True
    scores = []
    for keys in groups.values():
        scores_by_branch = []
        for key in keys:
            order = np.argsort(-members[key], axis=1, kind='stable')
            scores_by_branch.append(1. / (np.argsort(order, axis=1, kind='stable') + 1))
            available[by_row, order[:, :2]] = True
        scores.append(np.mean(scores_by_branch, axis=0))
    ranks = np.stack(scores, axis=1)
    return score_anchor, available, ranks


def fit_confusion(votes, gold, weights, buckets):
    k = len(buckets)
    bucket_widths = np.stack([(buckets == kind).sum(1) for kind in range(3)], axis=1)
    counts = []
    for branch in votes.T:
        matrix = np.zeros((k, k))
        np.add.at(matrix, (gold, branch), weights)
        bucket_probability = np.bincount(buckets[gold, branch], weights=weights, minlength=3) + 1.
        pooled = bucket_probability[buckets] / np.take_along_axis(bucket_widths, buckets, axis=1)
        pooled /= pooled.sum(axis=1, keepdims=True)
        tuned = matrix + PRIOR_SIZE * pooled
        tuned /= tuned.sum(axis=1, keepdims=True)
        counts.append((pooled, tuned))
    return np.log(np.asarray(counts, dtype=float))


def confidence_scores(rank, folds, gold, weights, buckets):
    votes = rank.argmax(axis=2)
    fit = fit_confusion(votes[folds], gold[folds], weights, buckets)
    score = {'ranks': np.log(rank.mean(axis=1))}
    cols = np.arange(votes.shape[1])
    for arm, index in [('pooled_confusion', 0), ('class_confusion', 1)]:
        score[arm] = fit[cols[None, :], index, :, votes].mean(axis=1)
    return score


def run(historical, original, independent, source, prior_kimi, saved_patch, output, draws=2500):
    public = read_ids(original / 'public_truth.parquet')
    holdout = read_prvsiyan(source, prior_kimi)
    panels = [(public, historical), (holdout, independent)]
    records = [read_capture(folder, truth) for truth, folder in panels]
    groups = groups_for(records)
    labels = np.asarray(records[0][0]['labels'])
    families = np.array(['qwen' if x.startswith('qwen') else x.split('-', 1)[0] for x in labels])
    bucket = np.where(families[:, None] == families[None, :], 1, 2)
    np.fill_diagonal(bucket, 0)
    row_blocks, anchors, masks, ranks, captures = [], [], [], [], []
    for (truth, folder), (trace, baseline, members, _, _) in zip(panels, records, strict=True):
        if not np.array_equal(labels, trace['labels']):
            raise ValueError('All captures must agree on the class order')
        anchor, mask, rank = pool_rankings(trace, members, groups, baseline)
        appeals = replay_predictions(trace, truth.text, typography=False)
        dash = truth.text.str.contains('—', regex=False).to_numpy()
        restore = np.char.startswith(appeals.astype(str), 'gemini-') & (appeals != baseline) & dash
        row_blocks.append(truth[['model', 'prompt_id', 'n_words', 'text', 'cohort']].assign(
            baseline=baseline, scored_dash_patch=np.where(restore, appeals, baseline),
            strict=truth.get('strict_topic_holdout', False)))
        anchors.append(anchor)
        masks.append(mask)
        ranks.append(rank)
        captures.append((truth, trace, baseline, json.loads((folder / 'glm_report.json').read_text())))
    rows = pd.concat(row_blocks, verify_integrity=True)
    anchor, candidates, rank = map(np.concatenate, (anchors, masks, ranks))
    gold = pd.Index(labels).get_indexer(rows.model)
    if (gold < 0).any():
        raise ValueError('All recorded truth labels must have a score')
    test_ids = rows.index[rows.cohort.isin(['evaluation', 'prvsiyan'])]
    fixed = read_ids(saved_patch).loc[test_ids]
    if not np.array_equal(fixed.scored_dash_patch, rows.loc[test_ids, 'scored_dash_patch']):
        raise ValueError('Exact scored Gemini dash patch changed')
    body = text_key(rows.text)
    prompt = canonical_prompt(rows.prompt_id)
    eligible = ~body.isin(body.loc[test_ids]) & ~prompt.isin(prompt.loc[test_ids])
    calibration = ['development', 'reconstructed_stress']
    train_sets = {'c1': rows.index[eligible & rows.cohort.eq(calibration[0])],
                  'c1_and_stress': rows.index[eligible & rows.cohort.isin(calibration)]}

    def predict(ids, evaluation):
        fit = rows.index.get_indexer(ids)
        dest = rows.index.get_indexer(evaluation)
        weights = class_weights(rows, ids).to_numpy()
        weights /= weights.mean()
        evidence = confidence_scores(rank, fit, gold, weights, bucket)
        results = {}
        for arm, signal in evidence.items():
            for weight in WEIGHTS:
                raw = anchor[dest] + weight * signal[dest]
                choice = labels[np.where(candidates[dest], raw, -np.inf).argmax(axis=1)]
                proposal = pd.DataFrame(dict(open=choice), index=evaluation)
                protected = []
                for truth, trace, baseline, receipt in captures:
                    shared = truth.index.intersection(evaluation, sort=False)
                    if not len(shared):
                        continue
                    frame = guard_frame(truth, trace, baseline, proposal.loc[shared], receipt=receipt)
                    protected.append(frame.native_and_fable if receipt.get('status') == 'applied' else frame.baseline)
                call = pd.concat(protected, verify_integrity=True).loc[evaluation]
                part = rows.loc[evaluation]
                if weight == 0 and not np.array_equal(call, part.baseline):
                    raise ValueError('Zero member evidence changed the original baseline')
                results[(arm, weight)] = call.where(part.baseline.eq(part.scored_dash_patch), part.scored_dash_patch)
        return pd.DataFrame(results, index=evaluation)

    results = rows.loc[test_ids].drop(columns='text')
    report = dict(groups=groups, excluded_nonuniversal=sorted(set(records[0][2]) - set(sum(groups.values(), []))),
                  fits={}, panels={})
    for source, ids in train_sets.items():
        held = pd.concat([predict(fit, valid) for fit, valid in prompt_folds(rows.loc[ids])], verify_integrity=True)
        panel = rows.loc[ids]
        groups_by_source = panel.groupby('cohort').groups
        evaluation = predict(ids, test_ids)
        selected = {}
        for arm in METHODS:
            scores = {weight: float(np.mean([balanced_accuracy(rows.loc[group, 'model'], held.loc[group, (arm, weight)])
                                             for group in groups_by_source.values()])) for weight in WEIGHTS}
            locked = max(WEIGHTS, key=lambda weight: (scores[weight], -weight))
            selected[arm] = dict(weight=locked, source_mean_ba=scores)
            results[source + '__' + arm] = evaluation[(arm, locked)]
        report['fits'][source] = dict(rows=panel.cohort.value_counts().to_dict(), prompts=int(panel.prompt_id.nunique()),
                                      represented_labels=sorted(panel.model.unique()), selected=selected)
    names = [column for column in results if column.startswith('c1__') or column.startswith('c1_and_stress__')]
    modes = {'c2': results.cohort.eq('evaluation'), 'heldout': results.cohort.eq('prvsiyan'), 'strict': results.strict}
    for mode, mask in modes.items():
        for length, part in (('all', results.loc[mask]), ('short', results.loc[mask & results.n_words.lt(100)])):
            report['panels'][mode + '__' + length] = [paired_stats(part, 'scored_dash_patch', name, draws)
                                                     for name in names]
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Scored', len(names), 'fixed member combinations on', len(test_ids), 'passages', flush=True)
