"""Measure shorter Low/user GLM windows on the original 587 candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

from glm_user_prefix import feature_block, held_out_member, metrics, select_rows, token_features
from likelihood_refine import transfer_glm_posterior
from likelihood_train import DIFFERENCES, GLM


MODELS = ('flash', 'glm53')
PARTS = ('prefix_part1', 'prefix_part2')
WINDOWS = (160, 128, 96, 64)
BLEND_WEIGHT = 2.706821502316715
ORIGINAL_BA = 0.6968282376336068


def read_traces(cache, tag, ids):
    by_id = {}
    for part in PARTS:
        directory = cache / part
        pooled = pd.read_csv(directory / f'{tag}_features.csv', dtype={'id': str}).set_index('id')
        # The public writer emits NPZ rows and CSV IDs from the same `ok` list.
        # Only numeric NPZ members are read, so no serialized objects are needed.
        with np.load(directory / f'{tag}_scores.npz', allow_pickle=False) as raw:
            offsets, scores, prefix, tokens = (raw[key] for key in ('offsets', 'scores', 'n_prefix', 'tokens'))
        if (not pooled.index.is_unique or set(pooled.index).intersection(by_id)
                or offsets.shape != (len(pooled) + 1,) or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0 or not np.array_equal(np.diff(offsets), pooled.n_tok.to_numpy())
                or not pooled.n_tok.between(1, 160).all() or prefix.shape != (len(pooled),)
                or (prefix < 1).any() or scores.shape != (offsets[-1], 4)
                or tokens.shape != (len(scores),) or not np.isfinite(scores).all()):
            raise ValueError(f'Cannot align {part}/{tag} token trajectories with their per-text CSV')
        for row, start, stop, prefix_length in zip(pooled.index, offsets[:-1], offsets[1:], prefix):
            by_id[row] = (scores[start:stop], int(prefix_length))
    if set(by_id) != set(ids):
        raise ValueError(f'{tag} trajectories do not cover the official rows exactly once')
    return [by_id[row] for row in ids]


def window_features(traces, chars, window, columns):
    tables = {
        model: pd.DataFrame([token_features(scores[:window], size)
                             for (scores, _), size in zip(traces[model], chars, strict=True)], dtype=np.float64)
        for model in MODELS
    }
    joint = pd.concat([tables[model].add_prefix(f'{model}:') for model in MODELS], axis=1)
    for key in DIFFERENCES:
        joint[f'd:{key}'] = tables['flash'][key] - tables['glm53'][key]
    return joint.loc[:, columns]


def token_cost(traces, chosen, window):
    result = {}
    for model in MODELS:
        selected = [traces[model][row] for row in chosen]
        body = sum(min(len(scores), window) for scores, _ in selected)
        saved = sum(len(scores) for scores, _ in selected) - body
        result[model] = {'body': body, 'body_saved': saved, 'prefix': sum(prefix for _, prefix in selected)}
    result['both'] = {key: sum(result[model][key] for model in MODELS) for key in ('body', 'body_saved', 'prefix')}
    return result


def verify_public_features(reconstructed, published):
    raw, csv = reconstructed.to_numpy(), published.to_numpy()
    absolute = np.abs(raw - csv)
    # CSV round-tripping a float32 statistic can perturb close differences.
    # Check the underlying model columns on their own scales for alignment.
    model_columns = np.array([not column.startswith('d:') for column in reconstructed.columns])
    close = np.isclose(raw, csv, rtol=2e-6, atol=1e-7)
    mismatches = {
        column: {'rows': int((~close[:, i]).sum()), 'max_absolute': float(absolute[:, i].max())}
        for i, column in enumerate(reconstructed.columns) if (~close[:, i]).any()
    }
    if not close[:, model_columns].all():
        raise ValueError(f'160-token raw and CSV statistics disagree: {json.dumps(mismatches)}')
    check = {'max_absolute': float(absolute.max()), 'outside_csv_tolerance': mismatches}
    print('160-token feature comparison', json.dumps(check), flush=True)
    return check


def compare(train_path, reference_path, cache):
    train = pd.read_csv(train_path, dtype={'id': str, 'prompt_id': str})
    labels = sorted(train.model.unique())
    columns = [f'reference__{name}' for name in labels]
    reference = pd.read_parquet(reference_path, columns=['id', 'fold', *columns])
    reference['id'] = reference.id.astype(str)
    if len(labels) != 18 or set(reference.id) != set(train.id):
        raise ValueError('The reference must cover the same official rows and classes')
    train = train.merge(reference, on='id', validate='one_to_one', sort=False)
    folds = train.fold.to_numpy()
    if len(np.unique(folds)) != 5 or train.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Use the original five prompt folds')
    base = log_softmax(train[columns].to_numpy(), axis=1)
    labels_array = np.asarray(labels)
    baseline = labels_array[base.argmax(axis=1)]
    target = train.model.to_numpy()
    text = train.text.fillna('')
    chars = text.str.len().to_numpy()
    short = text.str.split().str.len().to_numpy() < 100

    pooled_160, costs = feature_block(cache, 'low_user_160', train)
    glm = [labels.index(name) for name in GLM]
    chosen = select_rows(base, folds, glm, costs, 0.30, 130_000)
    if len(chosen) != 587 or int(costs[chosen].sum()) != 87409 or pooled_160.shape[1] != 54:
        raise ValueError('The inputs do not reproduce the original 587 candidates and 87,409 budget positions')
    traces = {model: read_traces(cache, model, train.id) for model in MODELS}
    results = {}
    for window in WINDOWS:
        block = window_features(traces, chars, window, pooled_160.columns)
        if window == 160:
            feature_check = verify_public_features(block, pooled_160)
        likelihood = held_out_member(block, target, folds)
        updated = base.copy()
        updated[chosen] = transfer_glm_posterior(base[chosen], base[chosen] + BLEND_WEIGHT * likelihood[chosen], labels)
        prediction = labels_array[updated.argmax(axis=1)]
        report = metrics(target, prediction, baseline, labels, folds)
        report['under_100_words'] = float(balanced_accuracy_score(target[short], prediction[short]))
        report['tokens'] = token_cost(traces, chosen, window)
        if window == 160 and not np.isclose(report['balanced_accuracy'], ORIGINAL_BA, rtol=0, atol=1e-9):
            raise ValueError(f"The 160-token reference BA did not reproduce: {report['balanced_accuracy']}")
        original = results.get('160', report)
        report['delta_160'] = {
            'balanced_accuracy': report['balanced_accuracy'] - original['balanced_accuracy'],
            'under_100_words': report['under_100_words'] - original['under_100_words'],
            'by_fold': {fold: value - original['by_fold'][fold] for fold, value in report['by_fold'].items()},
            'recall': {name: value - original['recall'][name] for name, value in report['recall'].items()},
        }
        results[str(window)] = report
    return {'rows': len(chosen), 'budget_positions_160': int(costs[chosen].sum()),
            'blend_weight': BLEND_WEIGHT, 'short_rows': int(short.sum()), 'selected_short_rows': int(short[chosen].sum()),
            'feature_check_160': feature_check,
            'windows': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/glm_prefix_window_metrics.json'))
    args = parser.parse_args()
    report = compare(args.train, args.reference, args.cache)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for window, values in report['windows'].items():
        print(window, values['balanced_accuracy'], values['under_100_words'], values['tokens']['both'], flush=True)


if __name__ == '__main__':
    main()
