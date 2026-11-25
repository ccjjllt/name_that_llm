"""Freeze matched short Terra and GPT-5.5 training and source-held pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .paired_attribution import read_pool, text_key


LABELS = ('gpt-5.5', 'gpt-5.6-terra')
HOLDOUT_SOURCE = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'
HOLDOUT_VERSION = 8
COLUMNS = ['id', 'text', 'model', 'pair_id', 'prompt_id', 'source_dataset', 'n_words']
PROMPT_SIMILARITY = .85


def stable_id(prefix, *parts):
    return prefix + '-' + hashlib.sha256('\n'.join(parts).encode()).hexdigest()[:20]


def match_pairs(rows, group):
    left = rows.loc[rows.model.eq(LABELS[0])].reset_index(drop=True)
    right = rows.loc[rows.model.eq(LABELS[1])].reset_index(drop=True)
    pairs = left.reset_index(names='left').merge(right.reset_index(names='right'), on=group)
    pairs['ratio'] = pairs[['n_words_x', 'n_words_y']].max(axis=1) / pairs[['n_words_x', 'n_words_y']].min(axis=1)
    pairs = pairs.loc[pairs.ratio.le(1.5)].sort_values(['ratio', 'id_x', 'id_y']).drop_duplicates(group)
    left = left.iloc[pairs.left].copy()
    right = right.iloc[pairs.right].copy()
    pair_ids = [stable_id('pair', l, r) for l, r in zip(left.id, right.id)]
    combined = pd.concat([left.assign(pair_id=pair_ids), right.assign(pair_id=pair_ids)], ignore_index=True)
    return combined.sort_values(['pair_id', 'model']).reset_index(drop=True)


def heldout_rows(source, forbidden):
    rows = pd.read_csv(source / 'passages.csv', low_memory=False)
    valid = (rows.model.isin(LABELS) & rows.split.isin(['new', 'new2'])
             & rows.passes_funnel.eq(True) & rows.finish_reason_norm.eq('stop')
             & rows.served_model_id.eq(rows.requested_model_id) & rows.requested_model_id.notna())
    rows = rows.loc[valid & rows.text.notna()].copy()
    rows['n_words'] = rows.text.str.split().str.len()
    rows = rows.loc[rows.n_words.between(40, 99)]
    keys = text_key(rows.text)
    rows = rows.loc[~keys.isin(forbidden) & ~keys.duplicated(keep=False)].copy()
    prompts = pd.read_csv(source / 'prompts.csv').set_index('prompt_id').prompt_body
    rows['prompt_body'] = rows.prompt_id.map(prompts)
    if rows.prompt_body.isna().any():
        raise ValueError('The held-out source has unknown prompt IDs')
    rows['source_dataset'] = HOLDOUT_SOURCE
    rows['id'] = rows.id.map(lambda id_: stable_id('terra-held', id_))
    # At most one pair from a prompt, with both answers from the same reported route.
    table = match_pairs(rows, ['prompt_id', 'provider_route'])
    chosen = table.drop_duplicates('prompt_id').pair_id
    return table.loc[table.pair_id.isin(chosen)].reset_index(drop=True)


def prompt_overlap(reference, held):
    ref = reference.drop_duplicates('prompt_canon').set_index('prompt_canon').prompt_text.dropna()
    queries = held.drop_duplicates('prompt_id').set_index('prompt_id').prompt_body
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, dtype=np.float32)
    candidates = vectorizer.fit_transform(ref)
    query = vectorizer.transform(queries)
    similarities = (query @ candidates.T).tocsr()
    maxima = np.asarray(similarities.max(axis=1).toarray()).ravel()
    near = set(ref.index[similarities.indices[similarities.data >= PROMPT_SIMILARITY]])
    return near, pd.Series(maxima, index=queries.index)


def prepare(pool, pool_v18, prompts_v15, prompts_v18, official, public, source, output):
    known = pd.concat([official, public], ignore_index=True)
    train, _ = read_pool(pool, known)
    forbidden = set(text_key(known.text))
    for path in (pool, pool_v18):
        forbidden.update(text_key(pd.read_parquet(path, columns=['text']).text.dropna()))
    held = heldout_rows(source, forbidden)
    disjoint, _ = prompt_overlap(pd.read_parquet(prompts_v15), held)
    _, v18_similarity = prompt_overlap(pd.read_parquet(prompts_v18), held)
    held['max_prompt_similarity_v18'] = held.prompt_id.map(v18_similarity)
    mask = (train.model.isin(LABELS) & ~train.prompt_canon.isin(disjoint)
            & ~train.source_dataset.str.startswith(('prvsiyan/', 'beroli/', 'markbarney/')))
    train = train.loc[mask].rename(columns={'prompt_canon': 'prompt_id'}).copy()
    train['n_words'] = train.text.str.split().str.len()
    train_pairs = match_pairs(train.loc[train.n_words.between(40, 99)], ['source_dataset', 'prompt_id'])
    if (held.empty or train_pairs.empty or not held.id.is_unique or not train_pairs.id.is_unique
            or not held.groupby('pair_id').model.nunique().eq(2).all()
            or not train_pairs.groupby('pair_id').model.nunique().eq(2).all()
            or set(text_key(held.text)) & set(text_key(train_pairs.text))):
        raise ValueError('Invalid training and source-held partition')

    folder = output / 'holdout'
    folder.mkdir(parents=True, exist_ok=True)
    train_pairs[COLUMNS].to_parquet(output / 'train_pairs.parquet', index=False)
    held[['id', 'text']].to_parquet(folder / 'queries.parquet', index=False)
    held.drop(columns=['text', 'prompt_body'])[[
        'id', 'model', 'prompt_id', 'n_words', 'source_dataset', 'pair_id', 'provider_route',
        'target_words', 'max_prompt_similarity_v18',
    ]].to_parquet(folder / 'truth.parquet', index=False)
    stats = dict(
        training_pairs=train_pairs.pair_id.nunique(), training_prompts=train_pairs.prompt_id.nunique(),
        training_sources=train_pairs.source_dataset.nunique(),
        training_source_counts=train_pairs.groupby('source_dataset').pair_id.nunique().to_dict(),
        heldout_source=HOLDOUT_SOURCE, heldout_source_version=HOLDOUT_VERSION,
        v15_prompt_groups_blocked=len(disjoint), prompt_block_similarity=PROMPT_SIMILARITY,
        holdout_rows=len(held), holdout_pairs=held.pair_id.nunique(),
        holdout_pairs_by_route=held.groupby('provider_route').pair_id.nunique().to_dict(),
        holdout_rows_below_v18_prompt_threshold=int(held.max_prompt_similarity_v18.lt(PROMPT_SIMILARITY).sum()),
        holdout_exact_body_overlap_v15_v18_official_or_public=0,
        ordered_training_body_sha256=hashlib.sha256('\n'.join(train_pairs.text).encode()).hexdigest(),
        ordered_holdout_body_sha256=hashlib.sha256('\n'.join(held.text).encode()).hexdigest(),
        holdout_scope='Source and near prompts are held out for the new expert. Third-party frozen baselines may have prior source exposure.',
        matching_scope='Same source and canonical prompt for training. Same source, prompt and route for held-out examples. Output length ratio at most 1.5. Exact requests may differ in target words.',
    )
    (output / 'manifest.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    print(json.dumps(stats, indent=2), flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'pool-v18', 'prompts-v15', 'prompts-v18', 'official', 'public', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pool, args.pool_v18, args.prompts_v15, args.prompts_v18,
            pd.read_csv(args.official, dtype={'id': str, 'prompt_id': str}), pd.read_parquet(args.public),
            args.source, args.output)
