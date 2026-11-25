"""Audit verified-provider reproductions of official prompts on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/matched_provider_20261125/kaggle_audit')


def build(output):
    run = '''from pathlib import Path
import json

import numpy as np
import pandas as pd

mount = Path('/kaggle/input')

def asset(slug, name):
    path, = [p for p in mount.rglob(name) if slug in p.parts]
    return path

def body(text):
    return (text.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\\*\\*(.*)\\*\\*$', r'\\1', regex=True)
            .str.replace(r'\\s+', ' ', regex=True).str.strip())

def surface(rows):
    text = rows.text
    return rows.assign(n_words=text.str.split().str.len(), em_dash=text.str.contains('—', regex=False),
                       spaced_dash=text.str.contains(' — ', regex=False),
                       curly_apostrophe=text.str.contains('’', regex=False))

names = {'gemini-3.7-flash': 'google/gemini-3.7-flash', 'gemini-3.8-flash': 'google/gemini-3.8-flash',
         'kimi-k3': 'moonshotai/kimi-k3', 'qwen3.8-max': 'qwen/qwen3.8-max-0902'}
verified = pd.read_csv(asset('ntl-matched-passages-four-models', 'passages.csv'), dtype={'id': str, 'prompt_id': str})
official = pd.read_csv(asset('name-that-llm', 'train.csv'), dtype={'id': str, 'prompt_id': str})
checks = verified.model.map(names)
if (len(verified) != 597 or not verified.id.is_unique or len(official) != 2699
        or not verified.served_model.eq(checks).all() or not verified.requested_model.eq(checks).all()
        or verified.duplicated(['prompt_id', 'model']).any() or verified.prompt_id.nunique() != 150):
    raise ValueError('Expected unique rows with the four exact published provider identities')
pool = pd.read_parquet(asset('name-that-llm-open-pool', 'ntl_open_pool.parquet'), columns=['text'])
historical = pd.read_parquet(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'), columns=['text'])
held = pd.read_parquet(asset('ntl-heldout-source-audit', 'candidate_queries.parquet'), columns=['text'])
verified, official = surface(verified), surface(official)
if not verified.n_words.eq(verified.words).all():
    raise ValueError('Provider report and actual word counts differ')
keys = body(verified.text)
for name, source in (('official', official), ('pool_v18', pool), ('historical', historical), ('held_pool', held)):
    verified['matches_' + name] = keys.isin(body(source.text))
exposure = [key for key in verified if key.startswith('matches_')]
verified['any_match'] = verified[exposure].any(axis=1)
verified['short'] = verified.n_words.lt(100)
clean = verified.loc[~verified.any_match & ~keys.duplicated(keep=False)]

gemini = ('gemini-3.7-flash', 'gemini-3.8-flash')
fields = ('n_words', 'em_dash', 'spaced_dash', 'curly_apostrophe')

def packet(rows):
    rows = rows.loc[rows.model.isin(gemini)]
    blocks = [rows.loc[rows.model.eq(label)].set_index('prompt_id')[list(fields)] for label in gemini]
    if any(not block.index.is_unique for block in blocks):
        raise ValueError('Expected one text per Gemini label and prompt')
    return blocks[0].add_suffix('_37').join(blocks[1].add_suffix('_38'), how='inner')

before = packet(official)
after = packet(clean)
shared = before.index.intersection(after.index)
before, after = before.loc[shared], after.loc[shared]
before_len = before[['n_words_37', 'n_words_38']].to_numpy()
after_len = after[['n_words_37', 'n_words_38']].to_numpy()
near = (np.abs(before_len - after_len) <= 12).all(axis=1) & (
    np.maximum(before_len, after_len) <= 1.15 * np.minimum(before_len, after_len)).all(axis=1)

def paired_surface(scope):
    output = {'prompt_groups': int(scope.sum())}
    for feature in fields:
        changes = []
        for table in (before, after):
            changes.append((table.loc[scope, feature + '_38'].astype(int)
                            - table.loc[scope, feature + '_37'].astype(int)).to_numpy())
        left, right = changes
        output[feature] = dict(official_38_higher=int((left > 0).sum()), verified_38_higher=int((right > 0).sum()),
                               official_37_higher=int((left < 0).sum()), verified_37_higher=int((right < 0).sum()),
                               both_different=int(((left != 0) & (right != 0)).sum()),
                               reversed_between_sources=int(((left * right) < 0).sum()))
    return output

report = dict(rows=len(verified), clean_rows=len(clean), clean_prompts=int(clean.prompt_id.nunique()),
              verified_by_label=verified.groupby('model')[['short', *exposure, 'any_match']].sum().to_dict('index'),
              clean_by_label=clean.groupby('model').agg(rows=('id', 'size'), short=('short', 'sum')).to_dict('index'),
              prompt_ids_shared_with_official=len(set(verified.prompt_id) & set(official.prompt_id)),
              gemini_surface={'all_shared': paired_surface(np.ones(len(shared), dtype=bool)),
                             'length_matched_across_sources': paired_surface(near)},
              limitation='The provider documentation describes the prompt bank as reconstructed, not original request bytes')
out = Path('/kaggle/working/matched_provider')
out.mkdir(exist_ok=True)
verified.drop(columns='text').to_parquet(out / 'row_audit.parquet', index=False)
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Verified-provider audit complete', len(verified), len(clean), flush=True)
'''
    compile(run, '<matched-provider>', 'exec')
    metadata = dict(id='zhuating/ntl-matched-provider-audit', title='NTL Matched Provider Audit',
                    code_file='matched-provider.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'], dataset_sources=['crtproduction/name-that-llm-open-pool/18',
                                                                          'markbarney/ntl-matched-passages-four-models/2'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/2'])
    save(output, metadata, dict(cells=[cell(run)], metadata={'kernelspec': {'display_name': 'Python 3',
                                                                          'language': 'python', 'name': 'python3'}},
                                nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
