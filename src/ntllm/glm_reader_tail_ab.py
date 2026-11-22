"""Compare two saved READER readouts under the same 0.20 Apex weight."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from glm_fable_typography import GLM, compare, find_artifact, inputs
from glm_reader_apex_ab import READER_WEIGHT
from padilla_apex import apex_decisions
from pngy_residual import Residual
from reader_margin import rank_reader_rows


BLENDS = {'dc_ac': (READER_WEIGHT, 0.0),
          'equal_mix': (READER_WEIGHT / 2, READER_WEIGHT / 2),
          'last': (0.0, READER_WEIGHT)}


def load_scores(frame, labels):
    keys = ['prompt_id', 'model', 'fold']
    columns = {view: [f'{view}__{label}' for label in labels] for view in ('dc_ac', 'last')}
    stored = pd.read_parquet(find_artifact('reader_qwen35_oof.parquet'),
                             columns=['id', *keys, *columns['dc_ac'], *columns['last']])
    stored[['id', 'prompt_id']] = stored[['id', 'prompt_id']].astype(str)
    stored = stored.set_index('id', verify_integrity=True)
    if len(stored) != len(frame) or not frame.index.isin(stored.index).all():
        raise ValueError('Reader scores and Low160 replay have different IDs')
    stored = stored.loc[frame.index]
    if not np.array_equal(stored[keys].to_numpy(), frame[keys].to_numpy()):
        raise ValueError('Reader scores and Low160 replay have different official prompt folds')
    dc_ac, last = (stored[names].to_numpy(dtype=float) for names in columns.values())
    if not np.isfinite(dc_ac).all() or not np.isfinite(last).all():
        raise ValueError('Reader scores must be finite')
    return dc_ac, last


def overlap(target, original, predictions):
    dc, last, mixed = (predictions[name] for name in ('dc_ac', 'last', 'equal_mix'))
    dc_changed, last_changed = dc != original, last != original
    dc_fixed = dc_changed & (dc == target)
    last_fixed = last_changed & (last == target)
    mix_fixed = (mixed == target) & (original != target)
    dc_harmed = dc_changed & (original == target)
    last_harmed = last_changed & (original == target)
    return {'changed_by_both_heads': int((dc_changed & last_changed).sum()),
            'both_heads_agree_on_changed_label': int((dc_changed & last_changed & (dc == last)).sum()),
            'both_fix': int((dc_fixed & last_fixed).sum()),
            'both_harm': int((dc_harmed & last_harmed).sum()),
            'only_dc_ac_fixes': int((dc_fixed & ~last_fixed).sum()),
            'only_last_fixes': int((last_fixed & ~dc_fixed).sum()),
            'mix_keeps_dc_ac_only_fixes': int((mix_fixed & dc_fixed & ~last_fixed).sum()),
            'mix_keeps_last_only_fixes': int((mix_fixed & last_fixed & ~dc_fixed).sum()),
            'mix_fixes_neither_head_fixed': int((mix_fixed & ~dc_fixed & ~last_fixed).sum())}


def summarize(target, original, predictions, folds, words):
    masks = {'all': np.ones(len(target), dtype=bool), 'short': words < 100, 'long': words >= 100,
             **{f'fold{fold}': folds == fold for fold in np.unique(folds)}}
    comparisons = {'original_to_dc_ac': (original, predictions['dc_ac']),
                   'original_to_last': (original, predictions['last']),
                   'original_to_equal_mix': (original, predictions['equal_mix']),
                   'dc_ac_to_equal_mix': (predictions['dc_ac'], predictions['equal_mix'])}
    return {'comparisons': {name: {part: compare(target, old, new, mask) for part, mask in masks.items()}
                            for name, (old, new) in comparisons.items()},
            'overlap': overlap(target, original, predictions)}


def main():
    frame, labels, base, post, glm_rows, budget, glm_weight = inputs()
    dc_ac, last = load_scores(frame, labels)
    target = frame.model.to_numpy()
    text = frame.text.fillna('').astype(str)
    texts = text.to_numpy()
    words = text.str.split().str.len().to_numpy()
    folds = frame.fold.to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    glm_winners = np.isin(current, GLM)
    restore = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')

    refined = np.empty_like(base)
    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        refined[hold] = residual.predict_log_probability(base[hold], texts[hold])
    original = apex_decisions(refined, labels, texts, current)
    full = {name: apex_decisions((1 - READER_WEIGHT) * refined + dc_weight * dc_ac + last_weight * last,
                                 labels, texts, current)
            for name, (dc_weight, last_weight) in BLENDS.items()}
    for pred in (original, *full.values()):
        pred[restore] = current[restore]

    selected = rank_reader_rows(refined, glm_winners, restore)[:len(frame) // 2]
    half = {name: original.copy() for name in full}
    for name, pred in full.items():
        half[name][selected] = pred[selected]

    full_report = summarize(target, original, full, folds, words)
    half_report = summarize(target, original, half, folds, words)
    control = [report['comparisons']['original_to_dc_ac']['all']['selective']
               for report in (full_report, half_report)]
    if not np.allclose(control, 0.7031270196370868, rtol=0, atol=1e-9):
        raise ValueError(f'Did not reproduce the existing full and 50% DC+AC result: {control}')
    result = {'glm_rows': len(glm_rows), 'glm_budget_positions': budget, 'glm_weight': glm_weight,
              'preserved_fable_rows': int(restore.sum()), 'weights_dc_ac_and_last': BLENDS,
              'margin50_rows': len(selected), 'full': full_report, 'margin50': half_report}
    output = json.dumps(result, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_tail_ab_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
