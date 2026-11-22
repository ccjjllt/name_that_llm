"""Check whether Qwen3.5 paired sibling evidence adds to the full READER blend."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from gemini_sentence_order import paired_score
from glm_fable_typography import GLM, compare, find_artifact, inputs
from glm_reader_apex_ab import READER_WEIGHT, reader_scores
from padilla_apex import apex_decisions
from pngy_residual import Residual
from sibling_pair_data import GEMINI_FLASH, GPT56


GROUPS = {'gpt56': GPT56, 'gemini_flash': GEMINI_FLASH}
PAIR_WEIGHT = 0.15


def fingerprints(frame):
    with np.load(find_artifact('reader_qwen35_features.npz'), allow_pickle=False) as archive:
        index = pd.Index(archive['ids'].astype(str))
        positions = index.get_indexer(frame.index)
        if len(index) != len(frame) or (positions < 0).any():
            raise ValueError('The saved fingerprints and Apex replay have different IDs')
        array = archive['values'][positions]
    if array.ndim != 3 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError('Expected finite last-token, DC and AC blocks')
    return array[:, 1:].reshape(len(frame), -1)


def paired_evidence(frame, labels, vectors, allowed):
    folds = frame.fold.to_numpy()
    by_prompt = frame[['prompt_id', 'model', 'fold']].assign(row=np.arange(len(frame)))
    results = {}
    effects = np.zeros((len(frame), len(labels)))
    for group, siblings in GROUPS.items():
        width = allowed[group].sum(axis=1).clip(1)
        for left, right in combinations(siblings, 2):
            rows = by_prompt.loc[by_prompt.model.isin((left, right))].pivot(
                index=['fold', 'prompt_id'], columns='model', values='row')
            rows = rows.reindex(columns=[left, right]).dropna()
            margins = np.empty(len(frame))
            counts = {}
            for fold in np.unique(folds):
                valid = folds == fold
                training = rows.loc[rows.index.get_level_values('fold') != fold].to_numpy(dtype=int)
                counts[str(fold)] = len(training)
                margins[valid] = paired_score(vectors[training[:, 0]], vectors[training[:, 1]], vectors[valid])
            i, j = labels.index(left), labels.index(right)
            eligible = allowed[group][:, i] & allowed[group][:, j]
            margin = margins[eligible].clip(-6, 6) / width[eligible]
            effects[eligible, i] += margin
            effects[eligible, j] -= margin
            results[left, right] = margins, counts
    return effects, results


def add_pairs(scores, original, allowed, evidence, labels, texts, current):
    name_array = np.asarray(labels)
    current_indices = pd.Index(labels).get_indexer(original)
    output = original.copy()
    for eligible in allowed.values():
        active = eligible[np.arange(len(scores)), current_indices]
        if not active.any():
            continue
        cols = eligible[active]
        updated = scores[active].copy()
        group_scores = np.where(cols, updated, -np.inf)
        adjustment = group_scores + PAIR_WEIGHT * evidence[active]
        group_scores = logsumexp(group_scores, axis=1, keepdims=True) + log_softmax(adjustment, axis=1)
        updated[cols] = group_scores[cols]
        reranked = apex_decisions(updated, labels, texts[active], current[active])
        selected = cols[np.arange(len(cols)), pd.Index(name_array).get_indexer(reranked)]
        output[np.flatnonzero(active)[selected]] = reranked[selected]
    return output


def metrics(frame, old, new, allowed, labels):
    target = frame.model.to_numpy()
    folds = frame.fold.to_numpy()
    short = frame.text.str.split().str.len().to_numpy() < 100
    old_index = pd.Index(labels).get_indexer(old)
    return {'all': compare(target, old, new, np.ones(len(frame), dtype=bool)),
            'by_fold': {str(fold): compare(target, old, new, folds == fold) for fold in np.unique(folds)},
            'length': {'short': compare(target, old, new, short), 'long': compare(target, old, new, ~short)},
            'groups': {name: {'active': int(allowed[name][np.arange(len(frame)), old_index].sum()),
                              **compare(target, old, new, np.isin(target, siblings))}
                       for name, siblings in GROUPS.items()},
            'recall': {label: compare(target, old, new, target == label)
                       for siblings in GROUPS.values() for label in siblings}}


def main():
    frame, labels, base, post, glm_rows, budget, likelihood_weight = inputs()
    target = frame.model.to_numpy()
    texts = frame.text.fillna('').astype(str).to_numpy()
    folds = frame.fold.to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    restore = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    frozen = restore | np.isin(current, GLM)
    scores = np.empty_like(base)
    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        scores[hold] = residual.predict_log_probability(base[hold], texts[hold])

    reader = reader_scores(frame, labels)['dc_ac']
    blended = (1 - READER_WEIGHT) * scores + READER_WEIGHT * reader
    original = apex_decisions(scores, labels, texts, current)
    full = apex_decisions(blended, labels, texts, current)
    original[restore] = full[restore] = current[restore]
    all_rows = np.ones(len(frame), dtype=bool)
    full_reference = compare(target, original, full, all_rows)
    if not np.allclose([full_reference['original'], full_reference['selective']],
                       [0.6946060154113846, 0.7031270196370868], rtol=0, atol=1e-9):
        raise ValueError('Did not reproduce the fixed .70313 Apex + Fable READER control')

    top = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    allowed = {}
    for name, siblings in GROUPS.items():
        allowed[name] = np.zeros_like(scores, dtype=bool)
        matching = np.isin(top, [labels.index(label) for label in siblings])
        active = matching[:, 0] & (matching.sum(axis=1) >= 2) & ~frozen
        allowed[name][np.flatnonzero(active)[:, None], top[active]] = matching[active]

    vectors = fingerprints(frame)
    with threadpool_limits(limits=2):
        evidence, pair_scores = paired_evidence(frame, labels, vectors, allowed)
    pair_only = add_pairs(scores, original, allowed, evidence, labels, texts, current)
    combined = add_pairs(blended, full, allowed, evidence, labels, texts, current)

    pairs = {}
    for (left, right), (margin, counts) in pair_scores.items():
        present = np.isin(target, (left, right))
        identity = target[present] == left
        i, j = labels.index(left), labels.index(right)
        reader_margin = reader[present, i] - reader[present, j]
        pairs[f'{left} vs {right}'] = {'training_pairs': counts,
                                      'paired_auc': float(roc_auc_score(identity, margin[present])),
                                      'reader_auc': float(roc_auc_score(identity, reader_margin)),
                                      'reader_score_correlation': float(np.corrcoef(reader_margin, margin[present])[0, 1])}
    changed_reader = full != original
    changed_pair = combined != full
    results = {'reader_weight': READER_WEIGHT, 'pair_weight': PAIR_WEIGHT,
               'glm_rows': len(glm_rows), 'glm_budget_positions': budget,
               'glm_weight': likelihood_weight, 'preserved_fable_rows': int(restore.sum()),
               'original_to_full_reader': full_reference,
               'full_reader_to_combined': metrics(frame, full, combined, allowed, labels),
               'original_to_pair_only': metrics(frame, original, pair_only, allowed, labels),
               'overlap_with_reader_changes': int((changed_reader & changed_pair).sum()),
               'reversals_to_original': int((changed_reader & changed_pair & (combined == original)).sum()),
               'pairwise': pairs}
    output = json.dumps(results, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_reader_pair_ab_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
