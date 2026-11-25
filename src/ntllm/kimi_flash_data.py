"""Freeze short Kimi/GLM Flash examples for a source-held experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .terra_pair_data import prompt_overlap, stable_id, text_key


PRV_SOURCE = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'
DRJ_SOURCE = 'drjcfmd/ntl-generated-v2'
ELLIOT_SOURCE = 'elliotthankinson/ntl-exact-gens-v3'
HOLDOUT_PROVIDER = {'glm-5.3-flash': 'Z.AI', 'kimi-k3': 'Moonshot AI'}
SERVED = {'glm-5.3-flash': 'z-ai/glm-5.3-flash', 'kimi-k3': 'moonshotai/kimi-k3'}
TRAIN_SOURCES = (
    'arxitekton/ntl-claude-gen-v1',
    'lxdlxdlxd/ntl-synth-generations-v1',
    'willrice/name-that-llm-openrouter-harvest',
)
TRAIN_FLAGS = (
    'flag_kb_proxy', 'flag_kb_proxy_after_ruling', 'flag_train_prompt_regen', 'flag_near_train_prompt',
)
MIN_WORDS = 40
MAX_WORDS = 99
MAX_LENGTH_RATIO = 1.5


def clean_rows(rows, forbidden):
    rows = rows.loc[rows.text.notna() & rows.id.notna() & rows.prompt_id.notna()].copy()
    rows['n_words'] = rows.text.str.split().str.len()
    rows = rows.loc[rows.n_words.between(MIN_WORDS, MAX_WORDS)]
    keys = text_key(rows.text)
    return rows.loc[~keys.isin(forbidden) & ~keys.duplicated(keep=False)].copy()


def same_prompt_pairs(rows):
    group = ['source_dataset', 'prompt_id']
    left = rows.loc[rows.model.eq('glm-5.3-flash')].reset_index(drop=True)
    right = rows.loc[rows.model.eq('kimi-k3')].reset_index(drop=True)
    pairs = left.reset_index(names='left').merge(right.reset_index(names='right'), on=group)
    lengths = pairs[['n_words_x', 'n_words_y']]
    pairs['length_ratio'] = lengths.max(axis=1) / lengths.min(axis=1)
    pairs['different_reasoning'] = pairs.reasoning_x.ne(pairs.reasoning_y)
    pairs = (pairs.loc[pairs.length_ratio.le(MAX_LENGTH_RATIO)]
             .sort_values(['different_reasoning', 'length_ratio', 'id_x', 'id_y'])
             .drop_duplicates(group))
    pair_ids = [stable_id('kimi-flash-pair', l, r) for l, r in zip(pairs.id_x, pairs.id_y)]
    chosen = pd.concat([left.iloc[pairs.left].assign(pair_id=pair_ids),
                        right.iloc[pairs.right].assign(pair_id=pair_ids)], ignore_index=True)
    return chosen.sort_values(['pair_id', 'model']).reset_index(drop=True)


def read_prvsiyan(source, forbidden):
    rows = pd.read_csv(source / 'passages.csv', low_memory=False)
    expected = rows.model.map(SERVED)
    accepted = (rows.split.eq('new') & rows.provider_route.eq('openrouter') & expected.notna()
                & rows.served_model_id.eq(expected) & rows.requested_model_id.eq(expected)
                & rows.served_provider.eq(rows.model.map(HOLDOUT_PROVIDER))
                & rows.reasoning_setting.fillna('').str.startswith('provider_default: no reasoning params sent')
                & rows.passes_funnel.eq(True) & rows.finish_reason_norm.eq('stop'))
    rows = clean_rows(rows.loc[accepted], forbidden)
    prompts = pd.read_csv(source / 'prompts.csv').set_index('prompt_id')
    registry = prompts.loc[prompts.split.eq('new')].prompt_body
    rows['prompt_body'] = rows.prompt_id.map(registry)
    if rows.prompt_body.isna().any():
        raise ValueError('Missing standalone prompt body in the holdout registry')
    rows['source_dataset'] = PRV_SOURCE
    rows['reasoning'] = 'default'
    rows['provenance'] = 'requested_and_served_ids'
    rows['prompt_id'] = 'prvsiyan:' + rows.prompt_id
    return rows


def read_drjcfmd(source, forbidden):
    rows = pd.read_csv(source / 'train_ext3.csv')
    accepted = rows.model.isin(SERVED) & rows.source.eq('gen') & rows.split.eq('new')
    rows = clean_rows(rows.loc[accepted], forbidden)
    prompts = pd.read_json(source / 'prompts_new.jsonl', lines=True).set_index('prompt_id')
    rows['prompt_body'] = rows.prompt_id.map(prompts.prompt)
    if rows.prompt_body.isna().any():
        raise ValueError('Missing standalone prompt body in the drjcfmd registry')
    return rows.assign(
        source_dataset=DRJ_SOURCE, prompt_id='drjcfmd:' + rows.prompt_id,
        provider_route='openrouter', reasoning='not_reported', provenance='publisher_api_label',
    )


def read_elliot(source, forbidden):
    rows = pd.read_parquet(source / 'or_paid_join.parquet')
    accepted = rows.model.isin(SERVED) & rows.setting.eq('host') & rows.discriminator_pass.eq(True)
    short = clean_rows(rows.loc[accepted], forbidden)
    short = short.assign(
        source_dataset=ELLIOT_SOURCE, prompt_id='elliot:' + short.prompt_key,
        provider_route='openrouter_paid', reasoning=short.model.map({'glm-5.3-flash': 'low', 'kimi-k3': 'none'}),
        provenance='publisher_verified_exact_model_id',
    )
    return short, rows.loc[rows.prompt_key.isin(short.prompt_key), ['prompt_key', 'text']]


def read_holdout(prvsiyan, drjcfmd, elliot, forbidden):
    short, elliot_siblings = read_elliot(elliot, forbidden)
    rows = pd.concat([read_prvsiyan(prvsiyan, forbidden), read_drjcfmd(drjcfmd, forbidden), short],
                     ignore_index=True)
    rows = clean_rows(rows, forbidden)
    rows['source_passage_id'] = rows.id
    rows['id'] = [stable_id('kimi-flash-held', source, passage)
                  for source, passage in zip(rows.source_dataset, rows.source_passage_id)]
    pairs = same_prompt_pairs(rows)
    rows = rows.merge(pairs[['id', 'pair_id']], on='id', how='left', validate='one_to_one').sort_values('id')
    return rows, elliot_siblings.loc[elliot_siblings.prompt_key.isin(rows.prompt_key.dropna())]


def pool_prompt_aliases(pool, siblings):
    sibling_bodies = siblings.assign(body_key=text_key(siblings.text))
    pool_keys = pool.assign(body_key=text_key(pool.text))
    matches = sibling_bodies[['prompt_key', 'body_key']].merge(
        pool_keys[['prompt_canon', 'body_key']], on='body_key',
    ).dropna(subset=['prompt_canon'])
    return set(matches.prompt_canon), matches.prompt_key.nunique()


def read_training(pool, forbidden):
    columns = ['id', 'text', 'model', 'source_dataset', 'source_file', 'prompt_canon',
               'prompt_evidence', 'gen_route', 'gen_served_model', 'gen_reasoning', *TRAIN_FLAGS]
    rows = pd.read_parquet(pool, columns=columns).rename(columns={
        'prompt_canon': 'prompt_id', 'gen_route': 'provider_route',
        'gen_served_model': 'served_model_id', 'gen_reasoning': 'reasoning',
    })
    served_label = rows.served_model_id.fillna('').str.rsplit('/', n=1).str[-1]
    valid = (rows.model.isin(SERVED) & served_label.eq(rows.model)
             & rows.source_dataset.isin(TRAIN_SOURCES) & rows.prompt_id.fillna('').ne('')
             & rows.prompt_evidence.isin(('own_text', 'own_bank', 'other_text'))
             & rows.provider_route.fillna('').ne('')
             & ~rows.source_file.fillna('').str.contains(r'train_regen|exact_pass|discriminator', case=False)
             & ~rows[list(TRAIN_FLAGS)].fillna(False).any(axis=1))
    return clean_rows(rows.loc[valid], forbidden)


def digest(rows):
    return hashlib.sha256('\n'.join(rows.text).encode()).hexdigest()


def prepare(pool, registry, pool_v15, official, public, prvsiyan, drjcfmd, elliot, output):
    known = set(text_key(pd.read_csv(official, usecols=['text']).text.dropna()))
    known.update(text_key(pd.read_parquet(public, columns=['text']).text.dropna()))
    pool_texts = pd.read_parquet(pool, columns=['text', 'prompt_canon']).dropna(subset=['text'])
    excluded = known | set(text_key(pool_texts.text))
    excluded.update(text_key(pd.read_parquet(pool_v15, columns=['text']).text.dropna()))
    original = pd.read_csv(prvsiyan / 'passages_clean.csv', usecols=['text', 'dataset_version_added'])
    admitted = original.loc[original.dataset_version_added.isin([1, 2, 4, 5]), 'text'].dropna()
    baseline_keys = set(text_key(admitted))
    held, elliot_siblings = read_holdout(prvsiyan, drjcfmd, elliot, excluded | baseline_keys)
    if held.empty or set(held.source_dataset) != {PRV_SOURCE, DRJ_SOURCE, ELLIOT_SOURCE}:
        raise ValueError('Each held-out publisher must have independently generated short examples')
    train = read_training(pool, known)
    train_prompts = pd.read_parquet(registry)
    train_prompts = train_prompts.loc[train_prompts.prompt_canon.isin(train.prompt_id)]
    if set(train.prompt_id) - set(train_prompts.dropna(subset=['prompt_text']).prompt_canon):
        raise ValueError('Missing registered training prompt bodies')
    known_held_prompts = held.dropna(subset=['prompt_body'])
    near, maximum = prompt_overlap(train_prompts, known_held_prompts)
    hold_prompt_keys = set(text_key(known_held_prompts.prompt_body))
    exact = set(train_prompts.loc[text_key(train_prompts.prompt_text).isin(hold_prompt_keys)].prompt_canon)
    aliases, mapped_elliot_keys = pool_prompt_aliases(pool_texts, elliot_siblings)
    blocked = near | exact | (aliases & set(train.prompt_id))
    held_keys = set(text_key(held.text))
    train = same_prompt_pairs(train.loc[~train.prompt_id.isin(blocked) & ~text_key(train.text).isin(held_keys)])
    if (train.empty or not train.id.is_unique or not held.id.is_unique
            or set(train.id) & set(held.id) or set(train.source_dataset) & set(held.source_dataset)
            or set(train.prompt_id) & set(held.prompt_id)
            or set(text_key(train.text)) & held_keys):
        raise ValueError('Expected disjoint, nonempty verified training and source-held short panels')
    for rows in (train, held):
        if not rows.groupby('pair_id').model.nunique().eq(2).all():
            raise ValueError('Expected two different model IDs per pair')

    holdout = output / 'holdout'
    holdout.mkdir(parents=True, exist_ok=True)
    data_cols = ['id', 'model', 'pair_id', 'prompt_id', 'source_dataset', 'n_words',
                 'provider_route', 'served_model_id']
    train[['text', *data_cols, 'reasoning']].to_parquet(output / 'train_pairs.parquet', index=False)
    held[['id', 'text']].to_parquet(holdout / 'queries.parquet', index=False)
    held[[*data_cols, 'requested_model_id', 'served_provider', 'reasoning', 'reasoning_setting',
          'target_words', 'source_passage_id', 'response_id',
          'provenance']].to_parquet(holdout / 'truth.parquet', index=False)
    manifest = dict(
        training_rows=len(train), training_short_rows=len(train), training_pairs=train.pair_id.nunique(),
        training_prompt_groups=train.prompt_id.nunique(),
        training_rows_by_class=train.model.value_counts().to_dict(),
        training_pairs_by_source=train.groupby('source_dataset').pair_id.nunique().to_dict(),
        training_same_reported_reasoning_pairs=int(train.groupby('pair_id').reasoning.nunique().eq(1).sum()),
        excluded_training_prompt_groups=len(blocked), exact_training_prompt_groups=len(exact),
        near_prompt_tfidf_threshold=.85, maximum_initial_prompt_cosine=float(maximum.max()),
        holdout_rows=len(held), holdout_short_rows=len(held), holdout_pairs=held.pair_id.nunique(),
        holdout_prompt_groups=held.prompt_id.nunique(),
        holdout_rows_by_class=held.model.value_counts().to_dict(),
        holdout_rows_by_source=held.source_dataset.value_counts().to_dict(),
        holdout_rows_by_source_and_class=held.groupby(['source_dataset', 'model']).size().unstack(fill_value=0).to_dict(orient='index'),
        holdout_matched_pairs_by_source=held.groupby('source_dataset').pair_id.nunique().to_dict(),
        holdout_source_versions={PRV_SOURCE: 13, DRJ_SOURCE: 4, ELLIOT_SOURCE: 7},
        primary_holdout_source=PRV_SOURCE,
        holdout_sources_by_role={PRV_SOURCE: 'primary', DRJ_SOURCE: 'secondary', ELLIOT_SOURCE: 'secondary'},
        elliot_prompt_keys_with_pool_alias=int(mapped_elliot_keys),
        elliot_prompt_keys=int(held.loc[held.source_dataset.eq(ELLIOT_SOURCE), 'prompt_id'].nunique()),
        elliot_pool_aliases_blocked_from_training=len(aliases & set(train_prompts.prompt_canon)),
        prvsiyan_provider=HOLDOUT_PROVIDER, prvsiyan_served_id=SERVED,
        holdout_exact_normalized_body_overlap_open_pool_v15_v18_official_public_or_prvsiyan_clean=0,
        holdout_scope='Primary Prvsiyan rows have exact returned and requested model IDs, pinned OpenRouter providers and default reasoning requests. drjcfmd documents direct API generation, with Kimi pinned to Moonshot AI, but supplies only the publisher label per row. Elliot reports verifying each returned model ID before writing a row and uses none for Kimi and low for GLM Flash. Elliot releases prompt keys with no prompt bodies, so prompt exclusion uses canonical aliases linked through sibling responses. All sources exclude known exact baseline bodies. We cannot verify the full corpus history of frozen third-party checkpoints.',
        matching_scope='Each training entry and marked holdout pair have one short passage per class on the same source prompt, with length ratio at most 1.5. Prefer compatible reported reasoning settings. Report all holdout rows and use the matched subset as a sensitivity check.',
        ordered_training_body_sha256=digest(train), ordered_holdout_body_sha256=digest(held),
    )
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'registry', 'pool-v15', 'official', 'public', 'prvsiyan', 'drjcfmd', 'elliot', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(**vars(parser.parse_args()))
