"""Measure untouched short responses from the pinned Zek-Takai Flash harvest."""

import hashlib
import json
from pathlib import Path
import re
import tempfile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET = 'Zek-Takai/glm53-flash-harvest'
REVISION = 'f912e86060f4b928ef2421d1398f3ea33c79c701'
ASSETS = {
    'provenance_r34.parquet': ('ef53d17e942f1e8cdaceca6093906b312108f30073a622069a8db6191d46d457', 200_000),
    'glm53_harvest.parquet': ('94b88d5aadbb35d7a3e88a4806093214d09d5dd5196670770e316db03e863578', 100_000_000),
    'glm53_harvest_r2.parquet': ('bab41a33a8353c243768e6dc651567ffdc6d4830464c0c108a08b9f8bf63e1d6', 87_000_000),
}
EXPLICIT_ASSETS = {
    'glm53_harvest_r3.parquet': ('76957e39c5b31fe1133582d22ec51824d71aa1b7ec4841ff92742613d8644049', 99_000_000),
    'glm53_harvest_r4.parquet': ('575544f2aad44166b80aec1b95f0b953316e9cf61cb4bbe6376707e2ac946510', 66_000_000),
}
REGISTERS = ('prose', 'chat')
CONTENT = ['id', 'register', 'lang', 'thinking', 'turns', 'temp', 'prompt', 'response', 'tokens', 'messages_json']
SPECIAL = re.compile(r'<\|[^\n]*?\|>|</?(?:think|analysis|tool[^>\s]*|assistant|system|user)\b[^>]*>|\[/?INST\]|<<\/?SYS>>', re.I)
MARKUP = re.compile(r'(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)|```|\[(?:Verse|Chorus|Bridge|Intro|Outro)(?:\s+\d+)?\]|\*\*[^\n]+?\*\*|(?:^|\s)\$\$|\\\[', re.I)
META = re.compile(r'^\s*(?:sure[,!]|here(?: is| are|\'s) (?:a|the|your) |as an AI\b|sorry[,!]|I (?:cannot|can\'t|am unable to)\b)', re.I)
NATURAL_END = re.compile(r'[.!?]["\'”’\)\]\}]*$')


def normalize(values):
    return (values.astype('string').str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\*\*(.*)\*\*$', r'\1', regex=True)
            .str.replace(r'\s+', ' ', regex=True).str.strip())


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def download(session, directory, name):
    digest, limit = (ASSETS | EXPLICIT_ASSETS)[name]
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}'
    path = directory / name
    checksum = hashlib.sha256()
    size = 0
    with session.get(url, stream=True, timeout=(15, 60)) as response, path.open('wb') as handle:
        response.raise_for_status()
        advertised = int(response.headers.get('Content-Length', 0))
        if advertised > limit:
            raise ValueError(f'{name} advertised {advertised} bytes, beyond the frozen limit')
        for block in response.iter_content(chunk_size=1024 * 1024):
            size += len(block)
            if size > limit:
                raise ValueError(f'{name} exceeded its {limit}-byte limit')
            checksum.update(block)
            handle.write(block)
    if checksum.hexdigest() != digest:
        raise ValueError(f'{name} differs from the published parquet checksum')
    print(f'Verified {name} ({size:,} bytes)', flush=True)
    return path


def rows_from_harvest(directory, explicit=False):
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1,
                                                              status_forcelist=(429, 500, 502, 503, 504))))
        provenance = pd.read_parquet(download(session, directory, 'provenance_r34.parquet'),
                                     columns=['id', 'temp', 'round', 'source', 'mode', 'has_system', 'finish', 'truncated'])
        if provenance[['id', 'temp']].duplicated().any():
            raise ValueError('Round 3/4 provenance keys are not unique')
        assets = ('glm53_harvest_r3.parquet', 'glm53_harvest_r4.parquet') if explicit else tuple(ASSETS)[1:]
        filters = [('lang', '=', 'en')] if explicit else [('register', 'in', list(REGISTERS))]
        inputs = [pd.read_parquet(download(session, directory, name), columns=CONTENT, filters=filters)
                  .assign(round=round_number) for round_number, name in enumerate(assets, 3 if explicit else 1)]
    rows = pd.concat(inputs, ignore_index=True).rename(columns={'id': 'source_id', 'response': 'text'})
    if rows[['round', 'source_id', 'temp']].duplicated().any():
        raise ValueError('Raw row identities are not unique')
    if explicit:
        rows = rows.merge(provenance.rename(columns={'id': 'source_id'}), on=['round', 'source_id', 'temp'],
                          how='left', validate='one_to_one', indicator=True)
        if not rows['_merge'].eq('both').all():
            raise ValueError('Round 3/4 output is missing provenance')
        rows = rows.drop(columns='_merge')
    return rows, provenance


def pool_matches(root, rows):
    paths = sorted(root.rglob('ntl_open_pool.parquet'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one mounted Open Pool, found {paths}')
    pool = paths[0]
    body = pd.read_parquet(pool, columns=['text'])
    prompts = pd.read_parquet(pool.with_name('prompts.parquet'), columns=['prompt_text'])
    official, = [path for path in root.rglob('train.csv') if path.parent.name == 'name-that-llm']
    rows['same_open_pool_body'] = rows.body_key.isin(set(normalize(body.text.dropna())))
    rows['same_open_pool_prompt'] = rows.prompt_key.isin(set(normalize(prompts.prompt_text.dropna())))
    rows['same_official_train_body'] = rows.body_key.isin(set(normalize(pd.read_csv(official, usecols=['text']).text)))
    return dict(official_body_rows=int(rows.same_official_train_body.sum()),
                pool_body_rows=int(rows.same_open_pool_body.sum()),
                pool_prompt_rows=int(rows.same_open_pool_prompt.sum()),
                reference_pool_bodies=len(body), reference_pool_prompts=len(prompts))


def describe(rows):
    return {
        'rows': len(rows),
        'prompt_groups': int(rows.prompt_group.nunique()),
        'distinct_bodies': int(rows.body_key.nunique()),
        'words_p10_p50_p90': [float(x) for x in rows.n_words.quantile([.1, .5, .9])] if len(rows) else [],
        'word_bands': {f'{low}-{high}': int(rows.n_words.between(low, high).sum())
                       for low, high in ((40, 59), (60, 79), (80, 99))},
        'em_dash': int(rows.text.str.contains('—', regex=False).sum()),
        'spaced_em_dash': int(rows.text.str.contains(r'\s—\s').sum()),
        'curly_quotes': int(rows.text.str.contains(r'[“”‘’]').sum()),
        'semicolon': int(rows.text.str.contains(';', regex=False).sum()),
        'colon': int(rows.text.str.contains(':', regex=False).sum()),
    }


def audit(rows, provenance, root, output, explicit=False):
    rows['text'] = rows.text.astype('string')
    rows['prompt_key'] = normalize(rows.prompt)
    rows['body_key'] = normalize(rows.text)
    rows['prompt_group'] = rows.prompt_key.str.replace(
        r'\s*\(alternative (?:implementation|version) #\d+:[^)]*\)\s*$', '', regex=True,
    ).map(sha, na_action='ignore')
    rows['n_words'] = rows.text.str.split().str.len()
    rows['english_single_turn'] = (rows.lang.eq('en') & rows.turns.eq(1) & rows.messages_json.isna()
                                   & rows.prompt_key.ne('') & rows.prompt_key.notna())
    rows['below_all_known_caps'] = rows.tokens.between(1, 1599)
    rows['explicit_stop'] = False
    if explicit:
        rows['english_single_turn'] &= rows['mode'].eq('single') & rows.has_system.eq(False)
        rows['explicit_stop'] = rows.finish.eq('stop') & rows.truncated.eq(False)
    completion = rows.explicit_stop if explicit else rows.below_all_known_caps
    rows['thinking_or_special'] = rows.thinking.fillna(True) | rows.text.str.contains(SPECIAL, na=True)
    rows['markup'] = rows.text.str.contains(MARKUP, na=True)
    rows['multiline'] = rows.text.str.strip().str.contains(r'[\r\n]', na=True)
    rows['meta_opening'] = rows.text.str.contains(META, na=True)
    rows['natural_end'] = rows.text.str.strip().str.contains(NATURAL_END, na=False)
    rows['short_single_turn'] = rows.english_single_turn & rows.n_words.between(40, 99)
    rows['strict_format'] = (rows.short_single_turn & completion & ~rows.thinking_or_special
                             & ~rows.markup & ~rows.multiline & ~rows.meta_opening & rows.natural_end)
    candidates = rows[rows.strict_format].copy()
    overlap = pool_matches(root, candidates)
    clean = candidates[~candidates[['same_open_pool_body', 'same_open_pool_prompt', 'same_official_train_body']].any(axis=1)]
    clean = clean.sort_values(['round', 'source_id', 'temp']).drop_duplicates('body_key').copy()
    clean['id'] = [f'zek-flash-r{r}-{i}-{temperature:.2f}' for r, i, temperature in zip(clean['round'], clean.source_id, clean.temp)]
    clean['prompt_id'] = 'zek-flash:' + clean.prompt_group
    clean['model'] = 'glm-5.3-flash'
    clean['source_dataset'] = f'{DATASET}@{REVISION}'
    clean['provider_route'] = 'publisher_self_hosted_vllm_reference_fp8'
    clean['served_model'] = 'zai-org/GLM-5.3-Flash'
    clean['completion_evidence'] = ('provenance_finish_stop_and_not_truncated' if explicit
                                    else 'below_all_known_caps_and_punctuation_no_finish_reason')
    provenance_register_keys = provenance.id.str.startswith(REGISTERS, na=False)
    if provenance_register_keys.any():
        raise ValueError('Round 3/4 provenance has prose/chat keys, expand scope to join those outputs')
    report = {
        'dataset': DATASET, 'revision': REVISION, 'downloaded_parquet_sha256': {k: v[0] for k, v in
                     (({'provenance_r34.parquet': ASSETS['provenance_r34.parquet']} | EXPLICIT_ASSETS) if explicit else ASSETS).items()},
        'pool_reference': 'crtproduction/name-that-llm-open-pool/19',
        'target_word_range': [40, 99], 'serving': 'Self-hosted vLLM using official reference FP8 weights, distinct from Z.AI API capture.',
        'rounds': {f'round_{n}_{reg}': {'input_rows': len(block),
                                       'short_english_single_turn': describe(block.loc[block.short_single_turn]),
                                       'strict_format_passes_completion_evidence': describe(block.loc[block.strict_format]),
                                       'clean_exact_overlap_screen': describe(clean.loc[clean['round'].eq(n) & clean.register.eq(reg)]),
                                       'short_format_flags': {flag: int(block.loc[block.short_single_turn, flag].sum()) for flag in
                                                              ('below_all_known_caps', 'explicit_stop', 'thinking_or_special', 'markup', 'multiline', 'meta_opening', 'natural_end')}}
                   for (n, reg), block in rows.groupby(['round', 'register'], sort=True)},
        'round34_provenance_rows': len(provenance), 'round34_prose_chat_provenance_rows': int(provenance_register_keys.sum()),
        'round12_explicit_stop_rows': 0, 'exact_overlap_among_candidates': overlap,
        'clean_all': describe(clean), 'clean_prose_only': describe(clean.loc[clean.register.eq('prose')]),
        'eligibility': 'Single-class self-hosted source only. No per-row finish_reason for prose/chat, no API equivalence, no exact inference-effort labels.',
    }
    output.mkdir(parents=True, exist_ok=True)
    clean[['id', 'text']].to_parquet(output / 'queries.parquet', index=False)
    clean.drop(columns=['text', 'body_key', 'prompt_key', 'messages_json']).to_parquet(output / 'truth.parquet', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def run(root=Path('/kaggle/input'), output=Path('/kaggle/working/flash_external_supply'), explicit=False):
    with tempfile.TemporaryDirectory(prefix='flash_external_supply_') as directory:
        rows, provenance = rows_from_harvest(Path(directory), explicit)
        audit(rows, provenance, root, output, explicit)
