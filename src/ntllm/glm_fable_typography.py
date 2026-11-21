"""Compare the two published typography probability sources with fixed GLM OOF inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from apex_0711 import apply_arrays as original_typography
from apex_mathlete import apply_arrays as selective_typography
from pngy_residual import Residual, features


GLM = ('glm-5.3', 'glm-5.3-flash')
STRENGTH = 0.10
PENALTY = 0.15


def find_artifact(filename):
    matches = list(Path('/kaggle/input').rglob(filename))
    if len(matches) != 1:
        raise ValueError(f'Expected one mounted {filename}, found {matches}')
    return matches[0]


def same_budget_rows(base, folds, glm_cols, costs):
    mass = np.exp(base[:, glm_cols]).sum(axis=1)
    picked = []
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        order = rows[np.argsort(-mass[rows], kind='stable')][:int(0.30 * len(rows))]
        picked.append(order[np.cumsum(costs[order]) <= 130_000 * len(rows) / 4000])
    return np.concatenate(picked)


def inputs():
    prefix = find_artifact('glm_user_prefix_oof.npz')
    manifest = json.loads(prefix.with_name('glm_user_prefix_metrics.json').read_text())
    labels = manifest['labels']
    weight = manifest['entropic_blend_weight']
    with np.load(prefix, allow_pickle=False) as artifact:
        ids = artifact['ids'].astype(str)
        folds = artifact['folds']
        likelihood = artifact['low_user_160_logp']
        costs = artifact['low_user_160_token_positions']
    columns = [f'reference__{label}' for label in labels]
    ref = pd.read_parquet(find_artifact('glm_dynamics_oof.parquet'),
                          columns=['id', 'prompt_id', 'model', 'fold', *columns])
    ref[['id', 'prompt_id']] = ref[['id', 'prompt_id']].astype(str)
    frame = pd.read_csv('/kaggle/input/competitions/name-that-llm/train.csv',
                        dtype={'id': str, 'prompt_id': str})
    frame = frame.merge(ref, on=['id', 'prompt_id', 'model'], validate='one_to_one')
    if len(frame) != len(ids) or set(ids) != set(frame.id):
        raise ValueError('Official passages and both OOF artifacts have different row IDs')
    frame = frame.set_index('id', verify_integrity=True).loc[ids]
    if (not np.array_equal(frame.fold.to_numpy(), folds) or len(np.unique(folds)) != 5
            or frame.groupby('prompt_id').fold.nunique().max() != 1):
        raise ValueError('OOF fold alignment differs from the original prompt split')
    base = log_softmax(frame[columns].to_numpy(), axis=1)
    if base.shape != likelihood.shape or not np.isfinite(base).all() or not np.isfinite(likelihood).all():
        raise ValueError('Continuous pre/post GLM inputs cannot be reconstructed')
    glm_cols = [labels.index(label) for label in GLM]
    selected = same_budget_rows(base, folds, glm_cols, costs)
    token_count = int(costs[selected].sum())
    expected = manifest['modes']['low_user_160']['same_130k_tokens_per_4000']
    if len(selected) != 587 or token_count != 87_409 or expected['rows'] != len(selected):
        raise ValueError(f'Expected the published 587-row budget, recovered {len(selected)} / {token_count}')
    post = base.copy()
    post[selected] += weight * likelihood[selected]
    return frame, labels, base, post, selected, token_count, weight


def evaluate(frame, labels, base, post):
    vectors = features(frame.text.fillna('').astype(str))
    texts = frame.text.fillna('').astype(str).tolist()
    target = frame.model.to_numpy()
    names = np.array(labels)
    current = names[post.argmax(axis=1)]
    original, selective = current.copy(), current.copy()
    folds = frame.fold.to_numpy()
    for fold in np.unique(folds):
        keep = folds != fold
        hold = np.flatnonzero(~keep)
        residual = Residual(labels).fit(np.asarray(texts)[keep], target[keep])
        evidence = np.ones_like(base[hold])
        for _, estimator in residual.models:
            values = estimator.predict_proba(vectors[hold])
            for column, label in enumerate(estimator.classes_):
                evidence[:, labels.index(label)] = values[:, column]
        args = (base[hold], evidence, labels, residual.families, [texts[i] for i in hold], current[hold])
        params = dict(strength=STRENGTH, corrected=False, penalty=PENALTY, post=post[hold])
        original[hold], refined_original = original_typography(*args, **params)
        selective[hold], refined_selective = selective_typography(*args, **params)
        if not np.array_equal(refined_original, refined_selective):
            raise ValueError('Rows before the typography rules differ')
    return original, selective


def ba(target, guess):
    return float(np.mean([(guess[target == label] == label).mean() for label in np.unique(target)]))


def compare(target, old, new, mask):
    y, a, b = target[mask], old[mask], new[mask]
    return {'rows': len(y), 'original': ba(y, a), 'selective': ba(y, b),
            'delta_ba': ba(y, b) - ba(y, a), 'changed': int((a != b).sum()),
            'fixed': int(((a != y) & (b == y)).sum()), 'lost': int(((a == y) & (b != y)).sum())}


def main():
    frame, labels, base, post, rows, tokens, weight = inputs()
    original, selective = evaluate(frame, labels, base, post)
    target = frame.model.to_numpy()
    folds = frame.fold.to_numpy()
    chosen = np.isin(np.arange(len(frame)), rows)
    words = frame.text.str.split().str.len().to_numpy()
    result = {'selected_rows': len(rows), 'token_positions': tokens, 'likelihood_weight': weight,
              'fable_return': {'margin_gain': 0.30, 'final_margin': 0.25},
              'typography_multiplier': PENALTY,
              'all': compare(target, original, selective, np.ones(len(frame), dtype=bool)),
              'selected': compare(target, original, selective, chosen),
              'unselected': compare(target, original, selective, ~chosen),
              'short_lt_100_words': compare(target, original, selective, words < 100),
              'folds': {str(fold): compare(target, original, selective, folds == fold)
                        for fold in np.unique(folds)},
              'families': {prefix: compare(target, original, selective, np.char.startswith(target.astype(str), prefix + '-'))
                           for prefix in ('glm', 'claude')},
              'recall': {label: compare(target, original, selective, target == label)
                         for label in labels if label.startswith(('glm-', 'claude-'))}}
    report = json.dumps(result, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_fable_typography_metrics.json').write_text(report + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
