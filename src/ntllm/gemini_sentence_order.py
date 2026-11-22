"""Compare Gemini sentence-position evidence on matched Vertex responses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

if __package__:
    from .paired_sibling_style import (FUNCTION_INDEX, SENTENCE_END, SHAPE_INDEX, WORD,
                                       normalized_counts, style_features, word_shape)
    from .sibling_pair_data import GEMINI_FLASH
else:
    from paired_sibling_style import (FUNCTION_INDEX, SENTENCE_END, SHAPE_INDEX, WORD,
                                     normalized_counts, style_features, word_shape)
    from sibling_pair_data import GEMINI_FLASH


SOURCE = 'family_mass_glm_posterior'
WEIGHT = 0.15


def sentence_vector(text, words):
    words_per_sentence = np.log1p(len(words))
    punctuation = [text.count(mark) / len(words) for mark in (',', ';', ':', '(', '—', '"')]
    counts = []
    for span in (words, words[:3], words[-3:]):
        functions = [FUNCTION_INDEX[word.lower()] for word in span if word.lower() in FUNCTION_INDEX]
        shapes = [SHAPE_INDEX[word_shape(word)] for word in span]
        counts.append(normalized_counts(functions, len(FUNCTION_INDEX), len(span)))
        counts.append(normalized_counts(shapes, len(SHAPE_INDEX), len(span)))
    return np.concatenate(([words_per_sentence], punctuation, *counts))


def representations(texts):
    summaries, levels, directions = [], [], []
    for text in texts:
        parts = [(sentence, words) for sentence in SENTENCE_END.split(text)
                 if (words := WORD.findall(sentence))]
        values = np.vstack([sentence_vector(sentence, words) for sentence, words in parts]
                           or [sentence_vector('', ['a']) * 0])
        position = np.arange(len(values)) + 0.5
        weights = -np.sqrt(2.0) * np.cos(np.pi * position / len(values))
        levels.append(values.mean(axis=0))
        directions.append((weights - weights.mean()) @ values / len(values))
        summaries.append(np.concatenate(style_features(text)))
    dc = np.stack(levels)
    return {'previous_style': np.stack(summaries), 'sentence_dc': dc,
            'sentence_dc_ac1': np.concatenate((dc, np.stack(directions)), axis=1)}


def normalized_text(texts):
    return texts.str.normalize('NFKC').str.casefold().str.replace(r'\s+', ' ', regex=True).str.strip()


def vertex_pairs(path, official_text):
    fields = ['arm', 'prompt_id', 'label', 'text', 'n_words', 'model_version', 'finish_reason']
    frame = pd.read_csv(path, usecols=fields, dtype={'prompt_id': str})
    frame = frame.loc[frame.arm.isin(('A', 'C')) & frame.label.isin(GEMINI_FLASH)
                      & frame.model_version.eq(frame.label) & frame.finish_reason.eq('STOP')
                      & frame.n_words.between(40, 250) & frame.text.notna()].copy()
    text_key = normalized_text(frame.text)
    frame = frame.loc[~text_key.isin(normalized_text(official_text)) & ~text_key.duplicated(keep=False)]
    pairs = frame.pivot(index=['arm', 'prompt_id'], columns='label', values='text')
    return pairs.reindex(columns=GEMINI_FLASH).dropna()


def paired_score(left, right, query):
    difference = left - right
    scale = np.sqrt(np.mean(difference ** 2, axis=0)).clip(0.01)
    direction = np.clip(difference / scale, -6, 6)
    midpoint = ((left + right) / 2).mean(axis=0)
    model = LogisticRegression(C=0.05, fit_intercept=False, solver='liblinear', max_iter=180)
    model.fit(np.concatenate((direction, -direction)), np.repeat([1, 0], len(direction)),
              sample_weight=np.full(2 * len(direction), 150 / len(direction)))
    return model.decision_function(np.clip((query - midpoint) / scale, -6, 6))


def balanced_accuracy(y, pred):
    return float(np.mean([(pred[y == label] == label).mean() for label in np.unique(y)]))


def compare(y, reference, pred):
    return {'rows': len(y), 'balanced_accuracy': balanced_accuracy(y, pred),
            'delta': balanced_accuracy(y, pred) - balanced_accuracy(y, reference),
            'fixed': int(((pred == y) & (reference != y)).sum()),
            'harmed': int(((pred != y) & (reference == y)).sum())}


def official_oof(root):
    official = pd.read_csv(next(root.glob('**/name-that-llm/train.csv')),
                           dtype={'id': str, 'prompt_id': str})
    labels = sorted(official.model.unique())
    keys = ['id', 'prompt_id', 'model']
    columns = [f'{SOURCE}__{label}' for label in labels]
    scores = pd.read_parquet(next(root.rglob('glm_dynamics_oof.parquet')), columns=[*keys, 'fold', *columns])
    scores[['id', 'prompt_id']] = scores[['id', 'prompt_id']].astype(str)
    frame = official.merge(scores, on=keys, validate='one_to_one')
    if len(frame) != len(official) or frame.fold.nunique() != 5 or frame.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Expected the existing five prompt-disjoint held-out folds')
    logits = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite probability source')
    return frame, labels, logits


def official_pairs(frame, mask):
    labeled = frame.loc[mask, ['prompt_id', 'model']].assign(row=np.flatnonzero(mask))
    pairs = labeled.pivot(index='prompt_id', columns='model', values='row')
    return pairs.reindex(columns=GEMINI_FLASH).dropna().to_numpy(dtype=int)


def measure_scores(frame, labels, logits, evidence):
    target = pd.Categorical(frame.model, categories=labels).codes
    siblings = [labels.index(label) for label in GEMINI_FLASH]
    left, right = siblings
    top = np.argsort(-logits, axis=1, kind='stable')[:, :3]
    top_in_group = np.isin(top, siblings)
    active = top_in_group[:, 0] & top_in_group[:, 1:].any(axis=1)
    relevant = np.isin(target, siblings)
    baseline = top[:, 0]
    base_margin = logits[:, left] - logits[:, right]
    lengths = frame.text.str.split().str.len().to_numpy()
    folds = frame.fold.to_numpy()

    matched = official_pairs(frame, relevant)
    diagnostics = {}
    for name, score in {'original': np.zeros(len(frame)), **evidence}.items():
        margin = base_margin + WEIGHT * score
        pred = baseline.copy()
        pred[active] = np.where(margin[active] > 0, left, np.where(margin[active] < 0, right, baseline[active]))
        order_gap = margin[matched[:, 0]] - margin[matched[:, 1]]
        diagnostics[name] = {'overall': compare(target, baseline, pred),
                             'gemini': compare(target[relevant], baseline[relevant], pred[relevant]),
                             'recall': {labels[i]: float((pred[target == i] == i).mean()) for i in siblings},
                             'pair_auc': float(roc_auc_score(target[relevant] == left, margin[relevant])),
                             'standalone_auc': float(roc_auc_score(target[relevant] == left, score[relevant])),
                             'same_prompt_order': float(np.mean((order_gap > 0) + 0.5 * (order_gap == 0))),
                             'folds': {str(fold): compare(target[mask], baseline[mask], pred[mask])
                                       for fold in np.unique(folds) if (mask := relevant & (folds == fold)).any()},
                             'length': {name: compare(target[mask], baseline[mask], pred[mask])
                                        for name, mask in (('<100', relevant & (lengths < 100)),
                                                           ('100+', relevant & (lengths >= 100)))}}
    fixable = active & relevant & (baseline != target)
    return {'source': SOURCE, 'weight': WEIGHT, 'matched_official_prompts': len(matched),
            'active': int(active.sum()), 'oracle_fixed': int(fixable.sum()), 'variants': diagnostics}


def evaluate(root):
    frame, labels, logits = official_oof(root)
    pair_source = next(root.glob('**/ntl-vertex-gemini-gen/passages.csv'))
    pairs = vertex_pairs(pair_source, frame.text)
    print('Verified, host-style Vertex pairs', pairs.groupby(level='arm').size().to_dict(), flush=True)
    if len(pairs) < 1000:
        raise ValueError('Expected the full set of independently matched Vertex passages')
    outer = representations([*pairs.iloc[:, 0], *pairs.iloc[:, 1]])
    official_features = representations(frame.text)
    with threadpool_limits(limits=2):
        evidence = {name: paired_score(values[:len(pairs)], values[len(pairs):], official_features[name])
                    for name, values in outer.items()}
    return {'external_pairs': len(pairs), **measure_scores(frame, labels, logits, evidence)}


if __name__ == '__main__':
    results = json.dumps(evaluate(Path('/kaggle/input')), indent=2, allow_nan=False)
    Path('/kaggle/working/gemini_sentence_order.json').write_text(results + '\n')
    print(results, flush=True)
