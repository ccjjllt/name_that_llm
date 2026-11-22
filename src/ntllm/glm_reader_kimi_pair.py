"""Measure and, if reachable, rerank Kimi inside the fixed READER replay."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from gemini_sentence_order import paired_score
from glm_fable_typography import GLM, compare, inputs
from glm_reader_apex_ab import READER_WEIGHT, reader_scores
from glm_reader_pair_ab import fingerprints
from padilla_apex import apex_decisions
from pngy_residual import Residual


KIMI = 'kimi-k3'
PAIR_WEIGHT = 0.15


def replay():
    frame, labels, base, post, *unused = inputs()
    target = frame.model.to_numpy()
    text = frame.text.fillna('').astype(str).to_numpy()
    folds = frame.fold.to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    restore = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    locked = restore | np.isin(current, GLM)
    refined = np.empty_like(base)
    for fold in np.unique(folds):
        hold = folds == fold
        model = Residual(labels).fit(text[~hold], target[~hold])
        refined[hold] = model.predict_log_probability(base[hold], text[hold])
    score = (1 - READER_WEIGHT) * refined + READER_WEIGHT * reader_scores(frame, labels)['dc_ac']
    predictions, probabilities = apex_decisions(score, labels, text, current, return_probs=True)
    predictions[restore] = current[restore]
    control = compare(target, predictions, predictions, np.ones(len(frame), dtype=bool))['original']
    if not np.isclose(control, 0.7031270196370868, rtol=0, atol=1e-9):
        raise ValueError(f'Could not reproduce the fixed 0.70313 control: {control}')
    return frame, labels, text, current, locked, score, predictions, probabilities, control


def report_reach(target, predictions, probabilities, labels, locked):
    top = np.argsort(-probabilities, axis=1, kind='stable')[:, :3]
    top_names = np.asarray(labels)[top]
    top_kimi = (top_names == KIMI).any(axis=1) & (probabilities[:, labels.index(KIMI)] > 0)
    eligible = top_kimi & ~locked
    missed = (target == KIMI) & (predictions != KIMI)
    false_kimi = (target != KIMI) & (predictions == KIMI)
    reachable = missed & eligible
    true_rival_present = (top_names == target[:, None]).any(axis=1)
    return {
        'true_kimi': int((target == KIMI).sum()),
        'missed_kimi': int(missed.sum()),
        'missed_kimi_in_top3': int((missed & top_kimi).sum()),
        'missed_kimi_reachable': int(reachable.sum()),
        'missed_kimi_top3_locked': int((missed & top_kimi & locked).sum()),
        'all_missed_kimi_winners': dict(Counter(predictions[missed]).most_common()),
        'reachable_kimi_rivals': dict(Counter(predictions[reachable]).most_common()),
        'false_kimi': int(false_kimi.sum()),
        'false_kimi_true_rival_reachable': int((false_kimi & eligible & true_rival_present).sum()),
        'reachable_false_kimi_truth': dict(Counter(target[false_kimi & eligible & true_rival_present]).most_common()),
        'rows_eligible': int(eligible.sum()),
    }, top_names, eligible


def held_out_evidence(frame, eligible):
    vectors = fingerprints(frame)
    positive = frame.loc[frame.model.eq(KIMI)].set_index('prompt_id').assign(row=np.flatnonzero(frame.model.eq(KIMI)))
    rivals = frame.loc[frame.model.ne(KIMI), ['fold', 'prompt_id']].assign(row=np.flatnonzero(frame.model.ne(KIMI)))
    pairs = rivals.join(positive['row'].rename('kimi'), on='prompt_id').dropna(subset=['kimi'])
    folds = frame.fold.to_numpy()
    evidence = np.zeros(len(frame))
    counts = {}
    with threadpool_limits(limits=2):
        for fold in np.unique(folds):
            training = pairs.loc[pairs.fold.ne(fold), ['kimi', 'row']].to_numpy(dtype=int)
            hold = (folds == fold) & eligible
            evidence[hold] = paired_score(vectors[training[:, 0]], vectors[training[:, 1]], vectors[hold])
            counts[str(fold)] = len(training)
    return evidence, counts


def experiment(frame, labels, text, current, score, original, top, eligible):
    target = frame.model.to_numpy()
    evidence, counts = held_out_evidence(frame, eligible)
    updated = score[eligible].copy()
    updated[:, labels.index(KIMI)] += PAIR_WEIGHT * evidence[eligible].clip(-6, 6)
    proposed = apex_decisions(updated, labels, text[eligible], current[eligible])
    changed = (proposed == KIMI) | (original[eligible] == KIMI)
    within_top3 = (top[eligible] == proposed[:, None]).any(axis=1)
    result = original.copy()
    result[np.flatnonzero(eligible)[changed & within_top3]] = proposed[changed & within_top3]
    folds = frame.fold.to_numpy()
    kimi = target == KIMI
    return {
        'pair_weight': PAIR_WEIGHT, 'training_pairs_by_fold': counts,
        'all': compare(target, original, result, np.ones(len(frame), dtype=bool)),
        'kimi': compare(target, original, result, kimi),
        'other': compare(target, original, result, ~kimi),
        'by_fold': {str(fold): compare(target, original, result, folds == fold) for fold in np.unique(folds)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args()
    frame, labels, text, current, locked, score, original, probabilities, control = replay()
    reach, top, eligible = report_reach(frame.model.to_numpy(), original, probabilities, labels, locked)
    print(json.dumps(reach, indent=2), flush=True)
    report = {'global_replay_ba': control, 'reach': reach}
    if args.fit and 2 * reach['missed_kimi_reachable'] > reach['missed_kimi']:
        report['experiment'] = experiment(frame, labels, text, current, score, original, top, eligible)
    output = json.dumps(report, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_kimi_pair_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
