"""Freeze cross-effort training pairs and a source-held Kimi/GLM panel."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .terra_pair_data import stable_id, text_key


LABELS = ('glm-5.3', 'glm-5.3-flash', 'kimi-k3')
EFFORTS = ('off', 'minimal', 'low', 'medium', 'high', 'default')
HOLDOUT_SOURCE = 'arxitekton/ntl-claude-gen-v1'
HOLDOUT_PUBLISHER = HOLDOUT_SOURCE.split('/')[0] + '/'
GROUP = ['source_dataset', 'prompt_id', 'model', 'provider_route', 'served_model']
COLUMNS = ['id', 'text', *GROUP, 'n_words', 'effort', 'pair_id']
PROMPT_SIMILARITY = .85
POOL_COLUMNS = [
    'id', 'text', 'model', 'source_dataset', 'source_file', 'prompt_canon', 'prompt_evidence',
    'gen_reasoning_evidence', 'gen_reasoning', 'gen_route', 'gen_served_model',
    'flag_kb_proxy', 'flag_kb_proxy_after_ruling', 'flag_near_train_prompt', 'flag_train_prompt_regen',
]


def read_candidates(pool, forbidden):
    rows = pd.read_parquet(pool, columns=POOL_COLUMNS)
    rows = rows.rename(columns={
        'prompt_canon': 'prompt_id', 'gen_reasoning': 'effort',
        'gen_route': 'provider_route', 'gen_served_model': 'served_model',
    })
    flags = [name for name in POOL_COLUMNS if name.startswith('flag_')]
    rows = rows[rows.model.isin(LABELS) & ~rows[flags].fillna(False).any(axis=1)].copy()
    for name in ('source_dataset', 'provider_route', 'served_model'):
        rows[name] = rows[name].fillna('').str.strip().str.lower()
    rows['source_dataset'] = rows.source_dataset.str.removeprefix('https://www.kaggle.com/datasets/')
    valid = (
        rows.text.notna()
        & rows.id.notna()
        & rows.prompt_id.fillna('').ne('')
        & rows.prompt_evidence.isin(('own_text', 'own_bank', 'other_text'))
        & rows.source_dataset.ne('')
        & rows.gen_reasoning_evidence.eq('row')
        & rows.effort.isin(EFFORTS)
        & rows.served_model.str.rsplit('/', n=1).str[-1].eq(rows.model)
        & rows.provider_route.ne('')
        & ~rows.provider_route.str.contains(r'benchmark|kaggle|proxy', regex=True)
        & ~rows.source_file.fillna('').str.contains(r'train_regen|discriminator|exact_pass', case=False)
    )
    rows = rows.loc[valid].copy()
    rows['n_words'] = rows.text.str.split().str.len()
    rows['body_key'] = text_key(rows.text)
    return rows.loc[rows.n_words.between(40, 250) & ~rows.body_key.isin(forbidden)
                    & ~rows.body_key.duplicated(keep=False)].copy()


def match_pairs(rows):
    # Greedy closest lengths with each actual passage allowed in at most one pair.
    ordered = rows.sort_values('id').reset_index(drop=True)
    matches = ordered.reset_index(names='left').merge(ordered.reset_index(names='right'), on=GROUP)
    length_max = matches[['n_words_x', 'n_words_y']].max(axis=1)
    length_min = matches[['n_words_x', 'n_words_y']].min(axis=1)
    matches['gap'] = length_max - length_min
    valid = (matches.left.lt(matches.right) & matches.effort_x.ne(matches.effort_y)
             & matches.gap.le(12) & length_max.le(length_min * 1.15))
    used = set()
    selected = []
    for match in matches.loc[valid].sort_values(['gap', 'id_x', 'id_y']).itertuples(index=False):
        if match.left not in used and match.right not in used:
            used.update((match.left, match.right))
            selected.append((match.left, match.right, stable_id('effort-pair', match.id_x, match.id_y)))
    indices = [index for left, right, _ in selected for index in (left, right)]
    pair_ids = [value for _, _, pid in selected for value in (pid, pid)]
    return ordered.iloc[indices].assign(pair_id=pair_ids)[COLUMNS].sort_values(['pair_id', 'effort']).reset_index(drop=True)


def near_prompts(registry, training_prompts, held_prompts):
    texts = registry.drop_duplicates('prompt_canon').set_index('prompt_canon').prompt_text.dropna()
    train = texts.loc[texts.index.isin(training_prompts)]
    held = texts.loc[texts.index.isin(held_prompts)]
    if train.empty or set(held.index) != set(held_prompts):
        raise ValueError('Expected registered held-out prompt bodies')
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform(train)
    sim = (vectorizer.transform(held) @ matrix.T).tocsr()
    return set(train.index[sim.indices[sim.data >= PROMPT_SIMILARITY]])


def prepare(pool, registry, official, public, output):
    forbidden = set(text_key(pd.read_csv(official, usecols=['text']).text))
    forbidden.update(text_key(pd.read_parquet(public, columns=['text']).text))
    rows = read_candidates(pool, forbidden)
    held = match_pairs(rows.loc[rows.source_dataset.eq(HOLDOUT_SOURCE)])
    pool_train = rows.loc[~rows.source_dataset.str.startswith(HOLDOUT_PUBLISHER)]
    held_prompts = set(held.prompt_id)
    near = near_prompts(pd.read_parquet(registry), set(pool_train.prompt_id), held_prompts)
    train = match_pairs(pool_train.loc[~pool_train.prompt_id.isin(held_prompts | near)])
    if (set(train.model) != set(LABELS) or set(held.model) != set(LABELS)
            or not train.id.is_unique or not held.id.is_unique or set(train.id) & set(held.id)
            or set(train.prompt_id) & set(held.prompt_id)
            or set(train.source_dataset) & set(held.source_dataset)
            or set(text_key(train.text)) & set(text_key(held.text))):
        raise ValueError('Training and held-out panels do not form disjoint three-class pairs')

    holdout = output / 'holdout'
    holdout.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output / 'train_pairs.parquet', index=False)
    held[['id', 'text']].to_parquet(holdout / 'queries.parquet', index=False)
    held.drop(columns='text').to_parquet(holdout / 'truth.parquet', index=False)
    manifest = dict(
        training_rows=len(train), training_pairs=train.pair_id.nunique(),
        training_prompt_groups=train.prompt_id.nunique(),
        training_pairs_by_class=train.groupby('model').pair_id.nunique().to_dict(),
        training_pairs_by_source=train.groupby('source_dataset').pair_id.nunique().to_dict(),
        excluded_training_prompt_groups=len(held_prompts | near),
        near_prompt_tfidf_threshold=PROMPT_SIMILARITY,
        holdout_rows=len(held), holdout_short_rows=int(held.n_words.lt(100).sum()),
        holdout_pairs=held.pair_id.nunique(), holdout_prompt_groups=len(held_prompts),
        holdout_source=HOLDOUT_SOURCE,
        holdout_rows_by_class=held.model.value_counts().to_dict(),
        holdout_short_rows_by_class=held.loc[held.n_words.lt(100)].model.value_counts().to_dict(),
        holdout_both_short_pairs=int(held.groupby('pair_id').n_words.max().lt(100).sum()),
        holdout_already_in_v18=True,
        matching_scope='Same publisher, canonical prompt, model, reported route and served ID, different explicitly reported effort. The request bytes may differ.',
        length_rule='40–250 words, at most 12 words apart and a longest:shortest ratio at most 1.15.',
        ordered_training_body_sha256=hashlib.sha256('\n'.join(train.text).encode()).hexdigest(),
        ordered_holdout_body_sha256=hashlib.sha256('\n'.join(held.text).encode()).hexdigest(),
    )
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'registry', 'official', 'public', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pool, args.registry, args.official, args.public, args.output)
