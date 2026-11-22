"""Replay Qwen3.5 DC+AC while holding the pre-READER Apex family mass fixed."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from glm_fable_typography import GLM, compare, inputs
from glm_reader_apex_ab import READER_WEIGHT, reader_scores
from padilla_apex import apex_decisions
from pngy_residual import Residual


FAMILIES = ('claude', 'gemini', 'glm', 'gpt', 'kimi', 'qwen')


def family(names):
    return np.asarray(['qwen' if name.startswith('qwen') else name.split('-', 1)[0] for name in names])


def fix_mass(base, blended, labels):
    groups = family(labels)
    fixed = blended.copy()
    for group in FAMILIES:
        cols = groups == group
        fixed[:, cols] += (logsumexp(base[:, cols], axis=1, keepdims=True)
                           - logsumexp(blended[:, cols], axis=1, keepdims=True))
    return fixed


def flip_counts(target, original, prediction, rows):
    flips = rows & (prediction != original)
    fixed = int((flips & (prediction == target)).sum())
    lost = int((flips & (original == target)).sum())
    return {'changed': int(flips.sum()), 'fixed': fixed, 'lost': lost, 'net_correct': fixed - lost}


def flow(target, original, prediction):
    truth, before, after = map(family, (target, original, prediction))
    same = before == after
    return {
        'same_family': flip_counts(target, original, prediction, same),
        'cross_family': flip_counts(target, original, prediction, ~same),
        'by_true_family': {
            group: {'same_family': flip_counts(target, original, prediction, (truth == group) & same),
                    'cross_family': flip_counts(target, original, prediction, (truth == group) & ~same)}
            for group in FAMILIES
        },
        'cross_family_directions': {
            f'{old}→{new}': flip_counts(target, original, prediction, (before == old) & (after == new))
            for old, new in sorted(set(zip(before[~same], after[~same])))
        },
    }


def evaluate(target, original, full, prediction, folds, short):
    groups = family(target)
    all_rows = np.ones(len(target), dtype=bool)
    return {
        'versus_original': compare(target, original, prediction, all_rows),
        'versus_global': compare(target, full, prediction, all_rows),
        'short': compare(target, original, prediction, short),
        'by_fold': {str(fold): compare(target, original, prediction, folds == fold) for fold in np.unique(folds)},
        'by_true_family': {group: compare(target, original, prediction, groups == group) for group in FAMILIES},
        'flips_from_original': flow(target, original, prediction),
    }


def main():
    frame, labels, base, post, selected, budget, weight = inputs()
    target = frame.model.to_numpy()
    text = frame.text.fillna('').astype(str)
    texts = text.to_numpy()
    short = text.str.split().str.len().to_numpy() < 100
    folds = frame.fold.to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    restore = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    refined = np.empty_like(base)
    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        refined[hold] = residual.predict_log_probability(base[hold], texts[hold])

    reader = reader_scores(frame, labels)['dc_ac']
    blended = (1 - READER_WEIGHT) * refined + READER_WEIGHT * reader
    preserved = fix_mass(refined, blended, labels)
    original, full, fixed = [apex_decisions(score, labels, texts, current)
                             for score in (refined, blended, preserved)]
    original[restore] = full[restore] = fixed[restore] = current[restore]
    reference = compare(target, original, full, np.ones(len(frame), dtype=bool))
    if not np.allclose([reference['original'], reference['selective']],
                       [0.6946060154113846, 0.7031270196370868], rtol=0, atol=1e-9):
        raise ValueError('Could not reproduce the fixed DC+AC 0.20 Apex + Fable replay')
    report = {
        'reader_weight': READER_WEIGHT, 'glm_rows': len(selected), 'glm_budget_positions': budget,
        'glm_weight': weight, 'restored_fable_rows': int(restore.sum()),
        'global': evaluate(target, original, full, full, folds, short),
        'family_mass_preserved': evaluate(target, original, full, fixed, folds, short),
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_family_mass_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
