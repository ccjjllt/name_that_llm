"""Estimate sibling preferences from matched prompts and evaluate held-out reranking."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

if __package__:
    from .sibling_pair_data import GEMINI_FLASH, GPT56, load_sibling_panels
else:
    from sibling_pair_data import GEMINI_FLASH, GPT56, load_sibling_panels


GROUPS = {'gemini_flash': GEMINI_FLASH, 'gpt56': GPT56}
PAIR_WEIGHT = (0.05, 0.10, 0.20, 0.35)
PROBABILITY_SOURCE = 'family_mass_glm_posterior'
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"”’']?\s+)|\n+")
PUNCTUATION = tuple(".,;:!?—–-()[]{}\"“”'’…/+=*") + (' — ', '— ', ' —', '\n\n', '  ')
FUNCTION_WORDS = tuple('a an and are as at be been being but by can could did do does for from had has have '
                       'he her here hers him his how i if in into is it its may me might more must my no not '
                       'of on one only or our out over she should so some than that the their them then there '
                       'these they this those through to too under up us was we were what when where which '
                       'while who will with would yet you your'.split())
FUNCTION_INDEX = {word: index for index, word in enumerate(FUNCTION_WORDS)}
SHAPES = ('function', 'number', 'upper', 'capital', 'contraction', 'ing', 'ed', 'ly',
          'ion', 's', 'short', 'medium', 'long')
SHAPE_INDEX = {shape: index for index, shape in enumerate(SHAPES)}


def word_shape(token):
    lower = token.lower()
    if lower in FUNCTION_INDEX:
        return 'function'
    if token[0].isdigit():
        return 'number'
    if token.isupper():
        return 'upper'
    if token[0].isupper():
        return 'capital'
    if "'" in token or '’' in token:
        return 'contraction'
    for ending in ('ing', 'ed', 'ly', 'ion', 's'):
        if len(token) > len(ending) + 2 and lower.endswith(ending):
            return ending
    if len(token) <= 4:
        return 'short'
    return 'medium' if len(token) <= 8 else 'long'


def normalized_counts(indices, size, denominator):
    indices = np.asarray(indices, dtype=np.intp)
    return np.bincount(indices, minlength=size).astype(np.float64) / max(1, denominator)


def style_features(text):
    sentences = [matches for part in SENTENCE_END.split(text) if (matches := WORD.findall(part))]
    words = [word for sentence in sentences for word in sentence]
    shape = [SHAPE_INDEX[word_shape(word)] for word in words]
    length = np.asarray([len(word) for word in words] or [0], dtype=float)
    sentence_length = np.asarray([len(sentence) for sentence in sentences] or [0], dtype=float)
    n_words = max(1, len(words))
    structure = [np.log1p(sentence_length.mean()), np.log1p(sentence_length.std()),
                 np.log1p(sentence_length.max()), length.mean(), length.std(),
                 np.mean(length < 4), np.mean(length >= 9)]
    structure += [text.count(mark) / n_words for mark in PUNCTUATION]
    structure += [text.count(mark) > 0 for mark in PUNCTUATION]
    structure = np.concatenate([structure, normalized_counts(shape, len(SHAPES), n_words)])

    opening = [word for sentence in sentences for word in sentence[:2]]
    ending = [word for sentence in sentences for word in sentence[-2:]]
    boundary = []
    for part in (words, opening, ending):
        indices = [FUNCTION_INDEX[word.lower()] for word in part if word.lower() in FUNCTION_INDEX]
        boundary.append(normalized_counts(indices, len(FUNCTION_WORDS), len(part)))
    for part in (opening, ending):
        indices = [SHAPE_INDEX[word_shape(word)] for word in part]
        boundary.append(normalized_counts(indices, len(SHAPES), len(part)))
    transitions = []
    for sentence in sentences:
        sequence = [SHAPE_INDEX[word_shape(word)] for word in sentence]
        transitions.extend(a * len(SHAPES) + b for a, b in zip(sequence, sequence[1:]))
    boundary.append(normalized_counts(transitions, len(SHAPES) ** 2, len(transitions)))
    return structure, np.concatenate(boundary)


def extract_views(texts):
    structure, boundary = zip(*(style_features(str(text)) for text in texts))
    structure, boundary = np.asarray(structure), np.asarray(boundary)
    return {'structure': structure, 'boundaries': boundary,
            'combined': np.concatenate([structure, boundary], axis=1)}


def matched_pairs(frame, features, left, right):
    indexed = frame[['prompt_id', 'model']].reset_index(drop=True).assign(row=np.arange(len(frame)))
    grouped = indexed[indexed.model.isin((left, right))].pivot_table(
        index='prompt_id', columns='model', values='row', aggfunc='first')
    pairs = grouped.reindex(columns=[left, right]).dropna().to_numpy(dtype=int)
    first, second = features[pairs[:, 0]], features[pairs[:, 1]]
    return first - second, (first + second) * 0.5


def predict_pair(external_difference, official_difference, official_midpoint, query, method):
    fit = np.concatenate([external_difference, official_difference], axis=0)
    scale = np.maximum(np.sqrt(np.mean(fit ** 2, axis=0)), 0.01)
    if method == 'external':
        fit = external_difference
        weights = np.full(len(fit), 150 / len(fit))
    elif method == 'official':
        fit = official_difference
        weights = np.full(len(fit), 150 / len(fit))
    else:
        weights = np.concatenate([np.full(len(external_difference), 75 / len(external_difference)),
                                  np.full(len(official_difference), 75 / len(official_difference))])
    difference = np.clip(fit / scale, -6.0, 6.0)
    head = LogisticRegression(C=0.05, fit_intercept=False, max_iter=180)
    head.fit(np.concatenate([difference, -difference]), np.repeat([1, 0], len(difference)),
             sample_weight=np.tile(weights, 2))
    midpoint = np.mean(official_midpoint, axis=0)
    return head.decision_function(np.clip((query - midpoint) / scale, -6.0, 6.0))


def paired_evidence(external, official, external_views, official_views, labels):
    folds = official.fold.to_numpy()
    effects = {}
    for view, official_x in official_views.items():
        external_x = external_views[view]
        for method in ('external', 'official', 'mixed'):
            effects[view, method] = np.zeros((len(official), len(labels)))
        for sibling in GROUPS.values():
            for left, right in combinations(sibling, 2):
                external_difference, _ = matched_pairs(external, external_x, left, right)
                if len(external_difference) == 0:
                    raise ValueError(f'No external pairs for {left} and {right}')
                for fold in np.unique(folds):
                    fit, valid = folds != fold, folds == fold
                    official_difference, official_midpoint = matched_pairs(
                        official.loc[fit], official_x[fit], left, right)
                    for method in ('external', 'official', 'mixed'):
                        score = predict_pair(external_difference, official_difference, official_midpoint,
                                             official_x[valid], method) / len(sibling)
                        effect = effects[view, method]
                        effect[valid, labels.index(left)] += score
                        effect[valid, labels.index(right)] -= score
    return effects


def rerank(logits, effect, labels, weight, top=3):
    candidates = np.argsort(-logits, axis=1, kind='stable')[:, :top]
    pred = candidates[:, 0].copy()
    eligible = np.zeros(len(logits), dtype=bool)
    for group in GROUPS.values():
        indices = [labels.index(label) for label in group]
        members = np.isin(candidates, indices)
        use = members[:, 0] & (members.sum(1) >= 2)
        eligible |= use
        scores = np.take_along_axis(logits[use] + weight * effect[use], candidates[use], axis=1)
        scores[~members[use]] = -np.inf
        pred[use] = candidates[use, scores.argmax(1)]
    return pred, eligible


def mean_recall(target, predicted, indices=None):
    classes = np.unique(target) if indices is None else indices
    return float(np.mean([(predicted[target == label] == label).mean() for label in classes]))


def measure(pred, original, eligible, target, labels, folds):
    result = {
        'balanced_accuracy': mean_recall(target, pred),
        'delta': mean_recall(target, pred) - mean_recall(target, original),
        'eligible_rows': int(eligible.sum()),
        'changed': int((pred != original).sum()),
        'fixed': int(((pred == target) & (original != target)).sum()),
        'harmed': int(((pred != target) & (original == target)).sum()),
        'fold_delta': {str(fold): mean_recall(target[folds == fold], pred[folds == fold])
                       - mean_recall(target[folds == fold], original[folds == fold]) for fold in np.unique(folds)},
    }
    result['target_recall'] = {label: float((pred[target == labels.index(label)] == labels.index(label)).mean())
                               for group in GROUPS.values() for label in group}
    result['group_delta'] = {track: float(np.mean([
        result['target_recall'][label] - (original[target == labels.index(label)] == labels.index(label)).mean()
        for label in group])) for track, group in GROUPS.items()}
    return result


def evaluate(train_path, pool_path, logits_path, output):
    train = pd.read_csv(train_path, dtype={'id': str, 'prompt_id': str})
    labels = sorted(train.model.unique())
    columns = [f'{PROBABILITY_SOURCE}__{label}' for label in labels]
    held_out = pd.read_parquet(logits_path, columns=['id', 'prompt_id', 'model', 'fold', *columns])
    held_out[['id', 'prompt_id']] = held_out[['id', 'prompt_id']].astype(str)
    official = train.merge(held_out, on=['id', 'prompt_id', 'model'], validate='one_to_one')
    if len(official) != len(train) or official.fold.nunique() != 5 or official.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Expected the same official rows and prompt-disjoint five folds as the likelihood experiment')
    external = load_sibling_panels(pool_path, official.text, gemini_prompts=900, gpt_prompts=900)
    external_counts = external.groupby(['track', 'source_dataset', 'matched_route']).prompt_id.nunique()
    print('External prompt groups', external_counts.to_dict(), flush=True)
    external_views = extract_views(external.text)
    official_views = extract_views(official.text)
    with threadpool_limits(limits=2):
        effects = paired_evidence(external, official, external_views, official_views, labels)
    logits = official[columns].to_numpy()
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite held-out likelihood logits')
    actual = pd.Categorical(official.model, categories=labels).codes
    original = logits.argmax(1)
    _, eligible = rerank(logits, np.zeros_like(logits), labels, 0.0)
    result = {'comparison': PROBABILITY_SOURCE, 'groups': GROUPS,
              'external_prompts': external.groupby('track').prompt_id.nunique().to_dict(),
              'comparison_score': measure(original, original, eligible, actual, labels, official.fold.to_numpy()),
              'variants': []}
    for (view, method), effect in effects.items():
        for weight in PAIR_WEIGHT:
            chosen, eligible = rerank(logits, effect, labels, weight)
            metrics = measure(chosen, original, eligible, actual, labels, official.fold.to_numpy())
            result['variants'].append({'features': view, 'training': method, 'weight': weight, **metrics})
            print(view, method, weight, round(metrics['balanced_accuracy'], 5), metrics['fixed'], metrics['harmed'], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--oof', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/paired_sibling_style_metrics.json'))
    args = parser.parse_args()
    root = Path('/kaggle/input')
    train = args.train if args.train is not None else next(root.glob('**/name-that-llm/train.csv'))
    pool = args.pool if args.pool is not None else next(root.rglob('ntl_open_pool.parquet'))
    oof = args.oof if args.oof is not None else next(root.rglob('glm_dynamics_oof.parquet'))
    evaluate(train, pool, oof, args.output)


if __name__ == '__main__':
    main()
