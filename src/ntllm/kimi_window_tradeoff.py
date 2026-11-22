"""Compare Kimi candidate counts and token windows using the v3 scored traces."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

if __package__:
    from .kimi_real_likelihood import KIMI, choose_rows, evaluate, features, read_reference
else:
    from kimi_real_likelihood import KIMI, choose_rows, evaluate, features, read_reference


CANDIDATES = (125, 200, 360)
WINDOWS = (64, 128)


def read_trajectories(path, expected_ids, expected_folds):
    with np.load(path, allow_pickle=False) as trace:
        if not np.array_equal(trace['ids'].astype(str), expected_ids):
            raise ValueError('Kimi v3 IDs must match the original 360 candidates in order')
        if not np.array_equal(trace['fold'], expected_folds):
            raise ValueError('Kimi v3 artifacts and the official reference have different folds')
        offsets = trace['offsets']
        scores = trace['token_scores']
    count = len(expected_ids)
    if offsets.shape != (count + 1,) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError('Expected one integer token boundary per passage and an end offset')
    lengths = np.diff(offsets)
    if (offsets[0] != 0 or (lengths < 1).any() or (lengths > WINDOWS[-1]).any()
            or scores.shape != (offsets[-1], 4) or not np.isfinite(scores).all()):
        raise ValueError('Kimi v3 token trajectories are incomplete or exceed the 128-token window')
    return [scores[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def input_file(root, pattern):
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Expected one Kaggle input matching {pattern!r}, found {matches}')
    return matches[0]


def compare(train_path, reference_path, trace_path, source_metrics=None):
    frame, labels, logits = read_reference(train_path, reference_path)
    folds = frame.fold.to_numpy()
    if len(np.unique(folds)) != 5:
        raise ValueError('The official reference must contain the original five folds')
    kimi = labels.index(KIMI)
    full, _ = choose_rows(logits, folds, kimi, CANDIDATES[-1])
    if len(full) != CANDIDATES[-1]:
        raise ValueError('The reference does not reproduce the 360 v3 candidates')
    trajectories = read_trajectories(trace_path, frame.id.iloc[full].to_numpy(dtype=str), folds[full])
    trace_index = dict(zip(full, range(len(full))))

    truth = np.searchsorted(labels, frame.model.to_numpy())
    short = frame.text.fillna('').str.split().str.len().to_numpy() < 100
    reference = logits.argmax(axis=1)
    reference_short = balanced_accuracy_score(truth[short], reference[short])
    reference_folds = {
        str(fold): balanced_accuracy_score(truth[folds == fold], reference[folds == fold])
        for fold in np.unique(folds)
    }
    report = {
        'reference': {'balanced_accuracy': balanced_accuracy_score(truth, reference),
                      'under_100_words': {'rows': int(short.sum()), 'balanced_accuracy': reference_short},
                      'balanced_accuracy_by_fold': reference_folds},
        'source': {'rows': len(full), 'max_body_tokens': WINDOWS[-1],
                   'body_tokens': sum(len(value) for value in trajectories)},
        'budgets': {},
    }
    if source_metrics is not None:
        measured = json.loads(source_metrics.read_text())
        if (measured['configuration']['max_tokens'] != WINDOWS[-1] or measured['configuration']['prefix'] != ' '
                or measured['selected_rows'] != len(full)):
            raise ValueError('Source metrics must describe the 360-row, 128-token, single-space v3 run')
        report['source']['v3_measured_segment_seconds'] = sum(segment['seconds'] for segment in measured['segments'])

    for count in CANDIDATES:
        chosen, margin = choose_rows(logits, folds, kimi, count)
        if len(chosen) != count:
            raise ValueError(f'The reference does not reproduce the {count} candidate route')
        budget = {'selected_under_100_words': int(short[chosen].sum()), 'windows': {}}
        for window in WINDOWS:
            traces = [trajectories[trace_index[row]][:window] for row in chosen]
            matrix = np.asarray([features(value) for value in traces])
            evaluation, odds = evaluate(frame, labels, logits, chosen, matrix, margin)
            evaluation['body_tokens'] = sum(len(value) for value in traces)
            for weight, metrics in evaluation['weights'].items():
                # evaluate returns the cross-fitted odds. Reuse them to measure
                # the same decisions on short passages without another fit.
                updated = logits.copy()
                updated[chosen, kimi] += float(weight) * odds
                prediction = updated.argmax(axis=1)
                short_ba = balanced_accuracy_score(truth[short], prediction[short])
                metrics['delta_ba'] = metrics['balanced_accuracy'] - report['reference']['balanced_accuracy']
                metrics['under_100_words'] = {'balanced_accuracy': short_ba, 'delta_ba': short_ba - reference_short}
                metrics['balanced_accuracy_by_fold'] = {
                    fold: reference_folds[fold] + delta for fold, delta in metrics['by_fold'].items()
                }
            budget['windows'][str(window)] = evaluation

        shorter, longer = (budget['windows'][str(window)] for window in WINDOWS)
        budget['64_minus_128'] = {
            'body_tokens': shorter['body_tokens'] - longer['body_tokens'],
            'weights': {
                weight: {
                    'balanced_accuracy': metrics['balanced_accuracy'] - longer['weights'][weight]['balanced_accuracy'],
                    'under_100_words': (metrics['under_100_words']['balanced_accuracy']
                                        - longer['weights'][weight]['under_100_words']['balanced_accuracy']),
                    'by_fold': {fold: delta - longer['weights'][weight]['by_fold'][fold]
                                for fold, delta in metrics['by_fold'].items()},
                }
                for weight, metrics in shorter['weights'].items()
            },
        }
        report['budgets'][str(count)] = budget
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--train', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--oof', type=Path)
    parser.add_argument('--source-metrics', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/kimi_window_tradeoff.json'))
    args = parser.parse_args()
    trace = args.oof or input_file(args.input, '**/kimi_real_likelihood_oof.npz')
    measured = trace.with_name('kimi_real_likelihood_metrics.json')
    report = compare(
        args.train or input_file(args.input, '**/name-that-llm/train.csv'),
        args.reference or input_file(args.input, '**/glm_dynamics_oof.parquet'),
        trace,
        args.source_metrics if args.source_metrics is not None else (measured if measured.is_file() else None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for count, budget in report['budgets'].items():
        for window, summary in budget['windows'].items():
            results = {weight: (round(values['delta_ba'], 6), round(values['under_100_words']['delta_ba'], 6))
                       for weight, values in summary['weights'].items()}
            print(f'{count} candidates, {window} max tokens, {summary["body_tokens"]} total body tokens', results, flush=True)


if __name__ == '__main__':
    main()
