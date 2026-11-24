"""Compare the two paired-attribution arms on the same held-out passages."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax


PILOT_COMPARISONS = (('ce', 'paired'), ('reference', 'reference_ce'),
                     ('reference', 'reference_paired'), ('reference_ce', 'reference_paired'))


def load_arm(path):
    frame = pd.read_parquet(path).set_index('id').sort_index()
    if not frame.index.is_unique:
        raise ValueError(f'Duplicate prediction IDs in {path}')
    return frame


def reference_scores(path, frame, labels):
    files = sorted(path.glob('fold*/plain_centered_gaussian.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No reference fold predictions under {path}')
    reference = pd.concat([pd.read_parquet(p) for p in files]).set_index('id').sort_index()
    if (not reference.index.equals(frame.index) or not np.array_equal(reference.fold, frame.fold)
            or not np.array_equal(reference.true_model, frame.model)
            or not np.array_equal(reference.prompt_id, frame.prompt_id)):
        raise ValueError('Reference predictions must match the arm rows and prompt folds')
    probs = reference[[f'prob__{label}' for label in labels]].to_numpy(dtype=float)
    return log_softmax(np.log(np.clip(probs, 1e-12, None)), axis=1)


def paired_report(frame, labels, scores, adaptation_weight, comparisons=PILOT_COMPARISONS):
    truth = pd.Categorical(frame.model, categories=labels).codes
    if (truth < 0).any() or np.unique(truth).size != len(labels):
        raise ValueError('Expected all labeled classes in the paired evaluation')
    prompt, prompts = pd.factorize(frame.prompt_id, sort=True)
    counts = np.zeros((len(prompts), len(labels)), dtype=int)
    np.add.at(counts, (prompt, truth), 1)
    rng = np.random.default_rng(20261123)
    draws = rng.multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=20_000)
    denominators = draws @ counts
    draws = draws[(denominators > 0).all(axis=1)]
    denominators = draws @ counts
    correct = {name: truth == values.argmax(axis=1) for name, values in scores.items()}
    report = {'rows': len(frame), 'prompts': len(prompts), 'draws_by_prompt': len(draws),
              'fixed_reference_blend_weight': adaptation_weight, 'comparisons': {}}

    for before, after in comparisons:
        if before not in correct or after not in correct:
            continue
        old, new = correct[before], correct[after]
        change = new.astype(int) - old.astype(int)
        table = np.zeros_like(counts)
        np.add.at(table, (prompt, truth), change)
        sampled_delta = ((draws @ table) / denominators).mean(axis=1)

        def macro(values, keep):
            return float(np.mean([values[keep & (truth == label)].mean() for label in np.unique(truth[keep])]))

        all_rows = np.ones(len(frame), dtype=bool)
        short = frame.n_words.to_numpy() < 100
        report['comparisons'][f'{before} -> {after}'] = {
            'before_ba': macro(old, all_rows), 'after_ba': macro(new, all_rows),
            'delta_ba': macro(change, all_rows), 'prompt_bootstrap_95': np.quantile(sampled_delta, [0.025, 0.975]).tolist(),
            'fixed': int((change == 1).sum()), 'harmed': int((change == -1).sum()),
            'short_delta': macro(change, short), 'long_delta': macro(change, ~short),
            'fold_delta': {str(fold): macro(change, frame.fold.eq(fold).to_numpy()) for fold in sorted(frame.fold.unique())},
            'class_recall_delta': {label: float(change[truth == i].mean()) for i, label in enumerate(labels)},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('ce', 'paired', 'out'):
        parser.add_argument('--' + path, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-blend-weight', type=float, default=0.30)
    args = parser.parse_args()
    if not 0 <= args.reference_blend_weight <= 1:
        parser.error('reference blend weight must be in [0, 1]')

    ce, paired = load_arm(args.ce), load_arm(args.paired)
    identity = ['prompt_id', 'model', 'fold', 'n_words']
    if not ce[identity].equals(paired[identity]):
        raise ValueError('Arm predictions differ in IDs, labels or folds')
    labels = sorted(ce.model.unique())
    columns = [f'logp__{label}' for label in labels]
    scores = {name: log_softmax(frame[columns].to_numpy(), axis=1) for name, frame in [('ce', ce), ('paired', paired)]}
    if args.reference:
        base = reference_scores(args.reference, ce, labels)
        weight = args.reference_blend_weight
        scores.update(reference=base, reference_ce=(1 - weight) * base + weight * scores['ce'],
                      reference_paired=(1 - weight) * base + weight * scores['paired'])
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError('Expected finite scores from both arms and their reference')
    report = paired_report(ce, labels, scores, args.reference_blend_weight if args.reference else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, result in report['comparisons'].items():
        print(name, result['before_ba'], result['after_ba'], result['prompt_bootstrap_95'],
              result['fixed'], result['harmed'])


if __name__ == '__main__':
    main()
