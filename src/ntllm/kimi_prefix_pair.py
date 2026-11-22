"""Compare absolute and matched-token Kimi-K3 likelihoods on the same 360 rows."""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .kimi_real_likelihood import KIMI, choose_rows, crossfit, evaluate, features, fit_calibrator, read_reference
    from .kimi_window_tradeoff import input_file, read_trajectories
else:
    from kimi_real_likelihood import KIMI, choose_rows, crossfit, evaluate, features, fit_calibrator, read_reference
    from kimi_window_tradeoff import input_file, read_trajectories


SOURCE_NAMES = ('ntl-kimi-real-likelihood', 'ntl-kimi-response-likelihood')
TRACE_FILE = 'kimi_real_likelihood_oof.npz'
METRICS_FILE = 'kimi_real_likelihood_metrics.json'
COUNT = 360
MAX_TOKENS = 128


def notebook_trace(root, slug):
    roots = (root / 'notebooks' / 'zhuating' / slug, root / slug)
    for directory in roots:
        path = directory / TRACE_FILE
        if path.is_file():
            return path
    raise FileNotFoundError(f'No {TRACE_FILE} mounted from zhuating/{slug}')


def read_condition(trace_path):
    report = json.loads(trace_path.with_name(METRICS_FILE).read_text())
    if report['selected_rows'] != COUNT or report['configuration']['max_tokens'] != MAX_TOKENS:
        raise ValueError(f'Expected {COUNT} candidates and up to {MAX_TOKENS} body tokens from {trace_path}')
    return report['configuration']


def difference_features(space, response):
    ratio = response[:, 0].astype(np.float64) - space[:, 0]
    edge = min(len(ratio), 32)
    return ratio.mean(), ratio.std(), ratio[-edge:].mean() - ratio[:edge].mean()


def crossfit_other_four(values, truths, folds, prompts):
    odds = np.empty(len(values), dtype=float)
    for fold in np.unique(folds):
        held_out = folds == fold
        fit = ~held_out
        if np.intersect1d(prompts[fit], prompts[held_out]).size:
            raise ValueError('A prompt crosses the calibration and evaluation folds')
        if len(np.unique(truths[fit])) != 2:
            raise ValueError(f'Both classes are needed to calibrate fold {fold}')
        model = fit_calibrator(values[fit], truths[fit])
        odds[held_out] = model.decision_function(values[held_out])
    return odds


def contrast(result, original):
    return {
        'auc_kimi_features': result['auc_kimi_features'] - original['auc_kimi_features'],
        'weights': {
            weight: {key: row[key] - original['weights'][weight][key]
                     for key in ('balanced_accuracy', 'kimi_recall', 'auc')}
            for weight, row in result['weights'].items()
        },
    }


def compare(train_path, reference_path, space_path, response_path):
    space_config = read_condition(space_path)
    response_config = read_condition(response_path)
    if (space_config['prefix'] != ' ' or response_config.get('prefix_mode') != 'xtml_user_empty_think_response'
            or response_config.get('prefix_tokens') != 31 or response_config.get('baseline_prefix_tokens') != 1
            or space_config['model'] != response_config['model']):
        raise ValueError('Expected the single-space v3 and 31-token XTML v1 runs of the same Kimi model')

    frame, labels, logits = read_reference(train_path, reference_path)
    folds = frame.fold.to_numpy()
    if len(np.unique(folds)) != 5:
        raise ValueError('The reference must use the original five prompt folds')
    chosen, margin = choose_rows(logits, folds, labels.index(KIMI), COUNT)
    if len(chosen) != COUNT:
        raise ValueError('The reference cannot reproduce the original 360 candidates')
    ids = frame.id.iloc[chosen].to_numpy(dtype=str)
    space = read_trajectories(space_path, ids, folds[chosen])
    response = read_trajectories(response_path, ids, folds[chosen])
    lengths = tuple(map(len, space))
    if lengths != tuple(map(len, response)):
        raise ValueError('The two Kimi trajectories have different body-token offsets')

    feature_sets = {
        'single_space': np.asarray([features(values) for values in space]),
        'xtml': np.asarray([features(values) for values in response]),
        'xtml_minus_space_logp': np.asarray([difference_features(a, b) for a, b in zip(space, response, strict=True)]),
    }
    protocols = {}
    for mode, calibrate in (('within_fold_halves', crossfit), ('other_four_folds', crossfit_other_four)):
        results = {}
        for name, matrix in feature_sets.items():
            result, _ = evaluate(frame, labels, logits, chosen, matrix, margin, calibrate=calibrate)
            result['feature_count'] = matrix.shape[1]
            results[name] = result
        for name in ('xtml', 'xtml_minus_space_logp'):
            results[name]['versus_single_space'] = contrast(results[name], results['single_space'])
        protocols[mode] = results

    original = protocols['within_fold_halves']
    for name, result in protocols['other_four_folds'].items():
        result['versus_within_fold_halves'] = contrast(result, original[name])

    ratio_means = feature_sets['xtml_minus_space_logp'][:, 0]
    is_kimi = frame.model.iloc[chosen].to_numpy() == KIMI
    return {
        'selected_rows': COUNT,
        'body_tokens_per_condition': sum(lengths),
        'configuration': {'single_space': space_config, 'xtml': response_config},
        'alignment': 'Both NPZ files match official IDs, folds and token boundaries. Body IDs are reused by '
                     'the XTML runner and are not stored in the score NPZ files.',
        'scope': 'Conditional on the frozen upstream Native+GLM OOF scores and candidate ordering. '
                 'The four-fold calibrator uses fixed Kimi features and no labels from its evaluation fold.',
        'other_four_folds_counts': {
            str(fold): {'fit': int((folds[chosen] != fold).sum()), 'evaluation': int((folds[chosen] == fold).sum())}
            for fold in np.unique(folds)
        },
        'mean_text_log_ratio': {'kimi': float(ratio_means[is_kimi].mean()),
                                'other': float(ratio_means[~is_kimi].mean())},
        'calibration_protocols': protocols,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--train', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--space', type=Path)
    parser.add_argument('--xtml', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/kimi_prefix_pair_metrics.json'))
    args = parser.parse_args()
    report = compare(
        args.train or input_file(args.input, '**/name-that-llm/train.csv'),
        args.reference or input_file(args.input, '**/glm_dynamics_oof.parquet'),
        args.space or notebook_trace(args.input, SOURCE_NAMES[0]),
        args.xtml or notebook_trace(args.input, SOURCE_NAMES[1]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
