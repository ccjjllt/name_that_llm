"""Test fixed Qwen3.5 READER blends inside the Padilla Apex OOF replay."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from glm_fable_typography import GLM, compare, find_artifact, inputs
from padilla_apex import apex_decisions
from pngy_residual import Residual


READER_WEIGHT = 0.20
CONDITIONS = ('original', 'dc', 'dc_ac')


def reader_scores(frame, labels):
    metadata = ['prompt_id', 'model', 'fold']
    columns = {condition: [f'{condition}__{label}' for label in labels] for condition in CONDITIONS[1:]}
    reader = pd.read_parquet(find_artifact('reader_qwen35_oof.parquet'),
                             columns=['id', *metadata, *[name for names in columns.values() for name in names]])
    reader[['id', 'prompt_id']] = reader[['id', 'prompt_id']].astype(str)
    reader = reader.set_index('id', verify_integrity=True)
    if set(reader.index) != set(frame.index):
        raise ValueError('The READER and original Low/user replay have different official IDs')
    reader = reader.loc[frame.index]
    if not np.array_equal(reader[metadata].to_numpy(), frame[metadata].to_numpy()):
        raise ValueError('The READER row labels, prompts, or evaluation folds differ')
    result = {condition: reader[names].to_numpy(dtype=float) for condition, names in columns.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('The saved READER scores are not finite')
    return result


def main():
    frame, labels, base, post, selected, budget, likelihood_weight = inputs()
    readers = reader_scores(frame, labels)
    folds = frame.fold.to_numpy()
    target = frame.model.to_numpy()
    texts = frame.text.fillna('').astype(str).to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    preserve = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    predictions = {name: current.copy() for name in CONDITIONS}

    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        refined = residual.predict_log_probability(base[hold], texts[hold])
        predictions['original'][hold] = apex_decisions(refined, labels, texts[hold], current[hold])
        for name, logits in readers.items():
            blended = (1 - READER_WEIGHT) * refined + READER_WEIGHT * logits[hold]
            predictions[name][hold] = apex_decisions(blended, labels, texts[hold], current[hold])

    short = frame.text.fillna('').str.split().str.len().to_numpy() < 100
    all_rows = np.ones(len(frame), dtype=bool)
    results = {'plain_padilla': {}, 'padilla_v7_restore': {}}
    for mode, restore in (('plain_padilla', False), ('padilla_v7_restore', True)):
        reference = predictions['original'].copy()
        if restore:
            reference[preserve] = current[preserve]
        for name, values in predictions.items():
            guess = values.copy()
            if restore:
                guess[preserve] = current[preserve]
            results[mode][name] = {
                'all': compare(target, reference, guess, all_rows),
                'short_lt_100_words': compare(target, reference, guess, short),
                'by_fold': {str(fold): compare(target, reference, guess, folds == fold) for fold in np.unique(folds)},
                'recall': {label: float((guess[target == label] == label).mean()) for label in labels},
                'restored_by_v7': int((guess != values).sum()),
            }

    report = {
        'selected_rows': len(selected), 'budget_positions': budget, 'likelihood_weight': likelihood_weight,
        'reader_weight': READER_WEIGHT, 'typography_multiplier': 0.0, 'v7_eligible_rows': int(preserve.sum()),
        'fusion': '0.8 * OOF sibling-refined logits + 0.2 * OOF READER logits before Padilla argmax and hard typography. '
                  'The original GLM winner mask and the optional v7 restore mask both come from the unchanged GLM path.',
        'results': results,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_apex_ab_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
