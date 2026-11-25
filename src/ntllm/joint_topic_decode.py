"""Measure globally consistent soft duplicate suppression on frozen predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .peer_collision import LEXICAL_THRESHOLD, collision_predictions, peer_edges
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


PENALTIES = (0., .25, .5, 1., 2., 4.)
GROUP_CAP = 6


def true_groups(frame):
    return list(frame.groupby(['cohort', 'prompt_id'], sort=False).indices.values())


def text_groups(similarity, cohort):
    left, right, _ = peer_edges(similarity, LEXICAL_THRESHOLD, neighbors=3)
    directed = np.zeros(similarity.shape, dtype=bool)
    directed[left, right] = True
    row, col = np.where(np.triu(directed & directed.T, k=1))
    order = np.argsort(-similarity[row, col], kind='stable')
    anchor = np.arange(len(similarity))
    groups = {i: [i] for i in anchor}
    for i, j in zip(row[order], col[order], strict=True):
        a, b = anchor[i], anchor[j]
        if a == b or cohort[i] != cohort[j] or len(groups[a]) + len(groups[b]) > GROUP_CAP:
            continue
        if (similarity[np.ix_(groups[a], groups[b])] < LEXICAL_THRESHOLD).any():
            continue
        anchor[groups[b]] = a
        groups[a].extend(groups.pop(b))
    return [np.asarray(group) for group in groups.values()]


def decision_scores(trace, texts, baseline):
    labels = pd.Index(trace['labels'])
    incumbent = labels.get_indexer(baseline)
    rows = np.arange(len(incumbent))
    before_type = replay_predictions(trace, texts, typography=False)
    frozen = trace['glm_winner'] | trace['fable_guard'] | (before_type != baseline)
    scores = trace['sibling_logp'].astype(np.float64, copy=True)
    scores[rows, incumbent] = np.maximum(scores[rows, incumbent], scores.max(axis=1) + .001)
    admissible = np.ones(scores.shape, dtype=bool)
    allowed = labels.isin(('claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'))
    for marker, forbid in [(' — ', ~allowed),
                           ('’', labels.str.startswith(('claude-', 'glm-'))),
                           ('—', labels.str.startswith('gemini-'))]:
        marked = texts.str.contains(marker, regex=False).to_numpy()
        admissible[np.ix_(marked, forbid)] = False
    admissible[frozen] = False
    admissible[rows, incumbent] = True
    return np.where(admissible, scores, -np.inf), incumbent, frozen


def decode(scores, groups, penalty, *, cap=None):
    """Exact maximum a posteriori assignment with a quadratic per-label occurrence cost."""
    output = scores.argmax(axis=1)
    infeasible = []
    for group in groups:
        if len(group) < 2:
            continue
        capacity = len(group) if cap is None else min(cap, len(group))
        slot_class = np.repeat(np.arange(scores.shape[1]), capacity)
        slot_cost = np.tile(penalty * np.arange(capacity), scores.shape[1])
        value = scores[np.ix_(group, slot_class)] - slot_cost
        try:
            rows, slots = linear_sum_assignment(value, maximize=True)
        except ValueError:
            infeasible.append(group)
            continue
        if len(rows) != len(group):
            infeasible.append(group)
            continue
        output[group[rows]] = slot_class[slots]
    return output, infeasible


def group_stats(frame, groups, baseline):
    prompts, labels = frame.prompt_id.to_numpy(), frame.model.to_numpy()
    captured = same_prompt = repeated = collisions = mixed = 0
    for group in groups:
        size = len(group)
        captured += size * (size - 1) // 2
        _, counts = np.unique(prompts[group], return_counts=True)
        same_prompt += int(np.sum(counts * (counts - 1) // 2))
        repeated += len(set(labels[group])) < size
        collisions += len(set(baseline[group])) < size
        mixed += len(counts) > 1
    return dict(groups=len(groups), multi_groups=sum(len(group) > 1 for group in groups),
                grouped_rows=sum(len(group) for group in groups if len(group) > 1),
                inferred_pairs=captured, correct_topic_pairs=same_prompt,
                pair_precision=same_prompt/captured if captured else None, mixed_topic_groups=mixed,
                groups_with_repeated_true_label=repeated, groups_with_baseline_collision=collisions)


def candidate_ceiling(scores, frame, groups, labels, incumbent):
    truth = pd.Index(labels).get_indexer(frame.model)
    idx = np.arange(len(frame))
    wrong = truth != incumbent
    collided = np.zeros(len(frame), dtype=bool)
    for group in groups:
        if len(set(incumbent[group])) < len(group):
            collided[group] = True
    top3 = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    possible = collided & wrong & np.isfinite(scores[idx, truth])
    return dict(baseline_errors=int(wrong.sum()), errors_in_collision_groups=int((wrong & collided).sum()),
                admitted_true_labels=int(possible.sum()), admitted_true_top3=int(
                    (possible & (top3 == truth[:, None]).any(1)).sum()))


def repeated_label_groups(frame):
    """Construct a false-merge stress set; labels never enter the decoder."""
    merged = []
    for _, same_label in frame.assign(position=np.arange(len(frame))).groupby('model'):
        positions = same_label.drop_duplicates('prompt_id').position.to_numpy()
        merged.extend(positions[i:i + 2] for i in range(0, len(positions) - 1, 2))
    used = {int(i) for group in merged for i in group}
    return merged + [np.array([i]) for i in range(len(frame)) if i not in used]


def choose_visible(frame, maximum):
    salt = f'joint-topic-visible/{maximum}/'.encode()
    digest = [hashlib.sha256(salt + label.encode()).hexdigest() for label in frame.index]
    ordered = frame.assign(position=np.arange(len(frame)), hash=digest).sort_values('hash')
    return np.sort(ordered.groupby('prompt_id').head(maximum).position.to_numpy())


def measures(frame, args, groups, penalty, labels, draws):
    matrix, incumbent = args
    columns = {'zero': (0., None), 'soft': (penalty, None), 'fixed_soft_1': (1., None),
               'repeat_cap2': (penalty, 2), 'hard_unique': (0., 1)}
    failures = {}
    for name, (weight, cap) in columns.items():
        result, skipped = decode(matrix, groups, weight, cap=cap)
        frame[name] = labels[result]
        failures[name] = dict(groups=len(skipped), rows=sum(map(len, skipped)))
    if not np.array_equal(frame.zero, frame.baseline):
        raise ValueError('Zero duplicate cost must reproduce every frozen decision')
    report = dict(groups=group_stats(frame, groups, frame.baseline.to_numpy()),
                  ceiling=candidate_ceiling(matrix, frame, groups, labels, incumbent), infeasible=failures,
                  all=[paired_stats(frame, 'baseline', arm, draws) for arm in columns])
    short = frame[frame.n_words.lt(100)]
    if len(short):
        report['short'] = [paired_stats(short, 'baseline', arm, draws) for arm in columns]
    return report


def run(trace_dir, affinity_file, output, draws=2500):
    truth = read_ids(trace_dir / 'public_truth.parquet')
    truth['n_words'] = truth.text.str.split().str.len()
    trace, baseline = read_trace(trace_dir, truth)
    with np.load(affinity_file, allow_pickle=False) as stored:
        ids = pd.Index(stored['ids'], name='id')
        same_ids(ids, truth.index, 'Similarity archive', ordered=True)
        affinity = stored['similarity']
    if affinity.shape != (len(truth), len(truth)) or not np.isfinite(affinity).all():
        raise ValueError('Expected the archived textual affinity matrix for these bodies')
    strings = trace['labels']
    scores, incumbent, freeze = decision_scores(trace, truth.text, baseline)
    rows = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(baseline=baseline, protected=freeze)
    cohorts = rows.cohort.to_numpy()
    inferred = text_groups(affinity, cohorts)
    dev = rows.cohort.eq('development')
    weight_scores = {}
    for weight in PENALTIES:
        prediction, _ = decode(scores, inferred, weight)
        weight_scores[weight] = balanced_accuracy(rows.model[dev], strings[prediction][dev.to_numpy()])
    penalty = max(weight_scores, key=weight_scores.get)
    rows['prior_greedy'] = collision_predictions(trace, truth.text, affinity, threshold=LEXICAL_THRESHOLD, weight=1.)
    diagnostics, full_predictions = {}, []
    for cohort, frame in rows.groupby('cohort'):
        pos = rows.index.get_indexer(frame.index)
        matrix = affinity[np.ix_(pos, pos)]
        group_refs = {'text_only': text_groups(matrix, frame.cohort.to_numpy()), 'true_prompt_control': true_groups(frame)}
        sections = {}
        for name, groups in group_refs.items():
            pred = frame.copy()
            sections[name] = measures(pred, (scores[pos], incumbent[pos]), groups, penalty, strings, draws)
            full_predictions.append(pred.assign(group_method=name))
        greedy = {'all': paired_stats(frame, 'baseline', 'prior_greedy', draws)}
        if frame.n_words.lt(100).any():
            greedy['short'] = paired_stats(frame[frame.n_words.lt(100)], 'baseline', 'prior_greedy', draws)
        diagnostics[cohort] = dict(full=sections, existing_greedy=greedy, stress={})
        if cohort == 'evaluation':
            for cap in (1, 2, 4):
                keep = choose_visible(frame, cap)
                part = frame.iloc[keep].copy()
                grouped = text_groups(matrix[np.ix_(keep, keep)], part.cohort.to_numpy())
                diagnostics[cohort]['stress'][f'at_most_{cap}_per_prompt'] = measures(
                    part, (scores[pos[keep]], incumbent[pos[keep]]), grouped, penalty, strings, draws)
            diagnostics[cohort]['stress']['false_merge_repeated_label'] = measures(
                frame.copy(), (scores[pos], incumbent[pos]), repeated_label_groups(frame), penalty, strings, draws)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(full_predictions).reset_index().to_parquet(output / 'paired_predictions.parquet', index=False)
    report = dict(penalty=penalty, development_BA_by_penalty=weight_scores, lexical_threshold=LEXICAL_THRESHOLD,
                  protection=dict(rows=int(freeze.sum()), mode='Preserve original GLM, Fable and typography changes'),
                  design_limits=['True topic groups are diagnostic controls', 'False-merge repeats are synthetic',
                                 'Sparse scenarios resample the fixed full-batch scores',
                                 'A cap-infeasible group falls back to its original decisions'],
                  cohorts=diagnostics)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Selected C1 penalty:', penalty, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('trace-dir', 'affinity-file', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2500)
    run(**vars(parser.parse_args()))
