"""Replay pre-READER margin routing at three fixed hidden-batch coverages."""

import json
from pathlib import Path

import numpy as np

from glm_fable_typography import GLM, compare, inputs
from glm_reader_apex_ab import READER_WEIGHT, reader_scores
from padilla_apex import apex_decisions
from pngy_residual import Residual
from reader_margin import rank_reader_rows


COVERAGES = (0.30, 0.50, 0.70)
MAX_READER_TOKENS = 384


def main():
    frame, labels, base, post, glm_rows, glm_budget, glm_weight = inputs()
    target = frame.model.to_numpy()
    text = frame.text.fillna('').astype(str)
    texts = text.to_numpy()
    folds = frame.fold.to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    glm_winners = np.isin(current, GLM)
    v7 = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    refined = np.empty_like(base)
    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        refined[hold] = residual.predict_log_probability(base[hold], texts[hold])

    ranked = rank_reader_rows(refined, glm_winners, v7)
    budgets = {str(coverage): ranked[:int(coverage * len(frame))] for coverage in COVERAGES}
    reader = reader_scores(frame, labels)['dc_ac']
    baseline = apex_decisions(refined, labels, texts, current)
    full = apex_decisions((1 - READER_WEIGHT) * refined + READER_WEIGHT * reader, labels, texts, current)
    baseline[v7] = full[v7] = current[v7]

    full_rows = np.ones(len(frame), dtype=bool)
    reference = compare(target, baseline, full, full_rows)
    if not np.allclose([reference['original'], reference['selective']],
                       [0.6946060154113846, 0.7031270196370868], rtol=0, atol=1e-9):
        raise ValueError('The original and full READER results with v7 did not reproduce')

    words = text.str.split().str.len().to_numpy()
    length_groups = {'lt_100': words < 100, '100_to_199': (words >= 100) & (words < 200), 'ge_200': words >= 200}
    estimated = np.clip((text.str.len().to_numpy() + 3) // 4, 1, MAX_READER_TOKENS)

    def summarize(result, mask):
        metrics = compare(target, baseline, result, mask)
        full_metrics = compare(target, baseline, full, mask)
        metrics['full_reader_ba'] = full_metrics['selective']
        metrics['delta_vs_full_reader_ba'] = metrics['selective'] - full_metrics['selective']
        return metrics

    results = {}
    for share, rows in budgets.items():
        guess = baseline.copy()
        # After the fixed OOF style fit, Apex decisions and v7 are row-local.
        guess[rows] = full[rows]
        all_metrics = summarize(guess, full_rows)
        positions = int(estimated[rows].sum())
        results[share] = {
            'reader_rows': len(rows), 'share_of_all_rows': len(rows) / len(frame),
            'estimated_token_positions': positions, 'share_of_full_estimated_tokens': positions / int(estimated.sum()),
            'retained_full_ba_gain': all_metrics['delta_ba'] / reference['delta_ba'], 'all': all_metrics,
            'length_bands': {key: {'selected': int(mask[rows].sum()), **summarize(guess, mask)}
                             for key, mask in length_groups.items()},
            'by_fold': {str(fold): {'selected': int((folds[rows] == fold).sum()), **summarize(guess, folds == fold)}
                        for fold in np.unique(folds)},
        }

    report = {
        'glm_selected_rows': len(glm_rows), 'glm_budget_positions': glm_budget, 'glm_weight': glm_weight,
        'reader_weight': READER_WEIGHT, 'v7_eligible_rows': int(v7.sum()), 'reader_eligible_rows': len(ranked),
        'cost_proxy': 'ceil(text character count / 4), clipped to 1–384',
        'full_estimated_token_positions': int(estimated.sum()), 'original_and_full_reader': reference,
        'results': results,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_margin_budget_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
