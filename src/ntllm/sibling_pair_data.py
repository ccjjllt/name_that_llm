"""Complete same-source, same-route sibling groups from the public Open Pool."""

import numpy as np
import pandas as pd


GEMINI_FLASH = ('gemini-3.7-flash', 'gemini-3.8-flash')
GPT56 = ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra')
VERTEX_PAIRED = 'davidpiln/ntl-vertex-gemini-gen'
GPT_PROXY_PAIRED = 'keshavatearth/ntl-gpt-new-prompts-v1'
MULTI_LAB_PAIRED = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'
AUTHOR_PAIRED = ('own_text', 'own_bank', 'other_text')
POOL_COLUMNS = (
    'id', 'text', 'model', 'prompt_canon', 'prompt_evidence', 'source_dataset',
    'n_words', 'gen_route', 'gen_reasoning', 'gen_served_model', 'flag_kb_proxy',
    'flag_gemini37_routing_risk', 'flag_train_prompt_regen', 'flag_near_train_prompt',
)
OUTPUT_COLUMNS = ('id', 'text', 'model', 'prompt_id', 'source_dataset', 'track',
                  'matched_route', 'gen_reasoning', 'n_words')


def _token_key(frame):
    return (frame.fillna('').astype(str).str.normalize('NFKC').str.casefold()
            .str.replace(r'\s+', ' ', regex=True).str.strip())


def _served_expected(frame):
    served = frame['gen_served_model'].str.lower().str.strip().str.rsplit('/', n=1).str[-1]
    normalized = served.str.replace(r'-\d{4}-\d{2}-\d{2}$', '', regex=True)
    return normalized.eq(frame['model']), served.eq('')


def _eligible_routes(frame):
    expected, missing = _served_expected(frame)
    source = frame['source_dataset'].str.lower().str.removeprefix('https://www.kaggle.com/datasets/')
    route = frame['gen_route'].str.lower().str.strip()
    gemini = frame['model'].isin(GEMINI_FLASH) & ~frame['flag_gemini37_routing_risk']
    gpt = frame['model'].isin(GPT56)
    vertex = route.str.contains('vertex', regex=False) | (source.eq(MULTI_LAB_PAIRED) & route.eq('google'))
    # David's release documents exact response modelVersion for every passage.
    documented_vertex = source.eq(VERTEX_PAIRED) & (expected | missing)
    first_party = route.str.fullmatch(r'openai(?:[ _-](?:api|direct|flex))?')
    router = route.str.fullmatch(r'openrouter(?:[ _-]paid)?')
    # The Keshav release predates the guidance on proxy generation and retains all
    # requests and returned IDs. It is separate from direct OpenAI API passages.
    documented_proxy = source.eq(GPT_PROXY_PAIRED) & (expected | missing)
    conditions = (gemini & (documented_vertex | (vertex & expected)),
                  gpt & expected & first_party & ~frame['flag_kb_proxy'],
                  gpt & expected & router & ~frame['flag_kb_proxy'],
                  gpt & documented_proxy)
    return np.select([condition.to_numpy(dtype=bool, na_value=False) for condition in conditions],
                     ['vertex', 'openai', 'openrouter', 'verified_proxy'], default='')


def _sample_groups(frame, siblings, track, count, rng):
    keys = ['prompt_canon', 'source_dataset', 'matched_route', 'gen_reasoning']
    frame = frame.loc[frame['model'].isin(siblings)].copy()
    frame['_pick'] = rng.random(len(frame))
    frame = frame.sort_values('_pick').drop_duplicates(keys + ['model'])
    frame['_panel'] = frame.groupby(keys, sort=False).ngroup()
    groups = frame.groupby('_panel', sort=False).agg(
        size=('model', 'size'), source=('source_dataset', 'first'),
        prompt=('prompt_canon', 'first'), route=('matched_route', 'first'),
        small=('n_words', 'min'), large=('n_words', 'max'), median=('n_words', 'median'),
    )
    groups = groups[groups['size'].eq(len(siblings))].copy()
    # Favor similar response lengths and complete groups from different sources
    # without using host labels, host-fit sample weights, or matrix features.
    groups['width'] = (groups['large'] - groups['small']) / groups['median'].clip(lower=1)
    groups['length'] = pd.cut(groups['median'], [0, 70, 105, 150, 200, np.inf], labels=False)
    groups['priority'] = groups['route'].map({'vertex': 0, 'openai': 0, 'openrouter': 1, 'verified_proxy': 2})
    groups['draw'] = rng.random(len(groups)) / np.exp(-groups['width'])
    groups = groups.sort_values(['priority', 'draw']).drop_duplicates('prompt')
    groups['within_bucket'] = groups.groupby(['source', 'length'], sort=False).cumcount()
    groups = groups.sort_values(['within_bucket', 'priority', 'draw']).head(count)
    result = frame.loc[frame['_panel'].isin(groups.index)].copy()
    result['prompt_id'] = 'pool:' + result['prompt_canon']
    result['track'] = track
    return result.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)


def load_sibling_panels(path, official_texts, *, gemini_prompts=900, gpt_prompts=900, seed=17):
    """Return matched two-label and three-label public groups for continued training.

    `path` points to `ntl_open_pool.parquet` in Open Pool v16. Pass all original
    official texts solely to remove response duplicates. Callers add official
    training rows after dividing their prompt groups into folds.
    """
    frame = pd.read_parquet(path, columns=list(POOL_COLUMNS))
    frame = frame[frame['model'].isin(GEMINI_FLASH + GPT56)].copy()
    for column in ('source_dataset', 'prompt_canon', 'gen_route', 'gen_reasoning', 'gen_served_model'):
        frame[column] = frame[column].fillna('').astype(str).str.strip()
    for column in ('flag_train_prompt_regen', 'flag_near_train_prompt',
                   'flag_gemini37_routing_risk', 'flag_kb_proxy'):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame['gen_reasoning'] = frame['gen_reasoning'].str.lower().replace('', 'unknown')
    frame['matched_route'] = _eligible_routes(frame)
    frame = frame.loc[
        frame['matched_route'].ne('') & frame['prompt_canon'].ne('')
        & frame['prompt_evidence'].isin(AUTHOR_PAIRED)
        & ~frame['flag_train_prompt_regen'] & ~frame['flag_near_train_prompt']
        & frame['n_words'].between(40, 250) & frame['text'].notna()
    ].copy()
    reference = _token_key(pd.Series(official_texts, dtype=object))
    frame['_text_key'] = _token_key(frame['text'])
    frame = frame.loc[frame['_text_key'].ne('') & ~frame['_text_key'].isin(reference)]
    frame = frame.loc[~frame.duplicated('_text_key', keep=False)]

    rng = np.random.default_rng(seed)
    return pd.concat([
        _sample_groups(frame, GEMINI_FLASH, 'gemini_flash', gemini_prompts, rng),
        _sample_groups(frame, GPT56, 'gpt56', gpt_prompts, rng),
    ], ignore_index=True)
