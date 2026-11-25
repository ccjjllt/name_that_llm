"""Freeze author-generated GLM and Fable passages outside the visible base corpora."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .heldout_source_audit import locate, text_key


SOURCE = 'drjcfmd/ntl-generated-v2/4'
PRV = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
POOL = 'crtproduction/name-that-llm-open-pool/18'
STACK_FILES = {
    'seyitkaangunes/ntl-extra-api-passages/2': ('gen_passages.csv', 'ntl_extra_passages_claude.csv'),
    'abdullah1afify/ntl-generations/11': ('ntl_generations.csv',),
    'dophuchung/ntl-proxy-generations/1': ('ntl_proxy_generations.csv',),
    'lucasboesen/ntl-free-proxy-generations/12': ('ntl_generated*.csv',),
    'zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/1': ('ntl_extra_passages_claude.csv',),
    PRV: ('passages_clean.csv',),
}
MODELS = ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3')


def body_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def body_hashes(values):
    return text_key(values.dropna()).map(body_hash)


def matched_pool(pool, targets):
    matched = set()
    for batch in pq.ParquetFile(pool).iter_batches(columns=['text'], batch_size=4096):
        matched.update(targets.intersection(body_hashes(batch.to_pandas().text)))
    return matched


def extra_bodies(root):
    """Conservatively count all source rows, including those pruned by stack fit."""
    patterns = sorted({name for names in STACK_FILES.values() for name in names})
    paths = sorted({p for pattern in patterns for p in root.rglob(pattern)})
    if not any(p.name == 'passages_clean.csv' for p in paths):
        raise ValueError('Prvsiyan stack training input is missing')
    if not any(p.name == 'ntl_generations.csv' for p in paths):
        raise ValueError('Abdullah stack training input is missing')
    return {str(path.relative_to(root)): set(body_hashes(pd.read_csv(path, usecols=['text']).text))
            for path in paths}


def nearest_prompt(texts, reference):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1, dtype=np.float32)
    known = vectorizer.fit_transform(reference)
    similarities = vectorizer.transform(texts) @ known.T
    return similarities.max(axis=1).toarray().ravel()


def summarize(rows):
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()),
                by_label=rows.model.value_counts().to_dict(),
                short_by_label=rows.loc[rows.n_words.lt(100), 'model'].value_counts().to_dict(),
                identity=rows.identity_evidence.value_counts().to_dict())


def run(root, prior, output, near_threshold=None):
    source_slug = SOURCE.split('/')[1]
    published = pd.read_csv(locate(root, 'train_ext3.csv', source_slug), dtype={'prompt_id': str})
    registry = pd.read_json(locate(root, 'prompts_new.jsonl', source_slug), lines=True).set_index('prompt_id')
    eligible = published.model.isin(MODELS) & published.prompt_id.isin(registry.index) & published.text.notna()
    rows = published.loc[eligible].copy().rename(columns={'id': 'source_id', 'prompt_id': 'original_prompt_id'})
    rows['prompt_id'] = 'drjcfmd:' + rows.original_prompt_id
    rows['prompt_body'] = rows.original_prompt_id.map(registry.prompt)
    rows['n_words'] = rows.text.str.split().str.len()
    rows = rows[rows.n_words.between(40, 250)].copy()
    rows['identity_evidence'] = 'publisher_asserted'
    rows['served_model_id'] = pd.Series(pd.NA, index=rows.index, dtype='string')
    for column in ('served_model', 'gen_served_model', 'model_seen'):
        if column in rows:
            rows['served_model_id'] = rows.served_model_id.fillna(rows[column].astype('string'))
    if 'served_model_id' in published:
        rows['served_model_id'] = rows.served_model_id.fillna(published.loc[rows.index, 'served_model_id'])
    served = rows.served_model_id.str.strip().replace('', pd.NA)
    canonical_served = served.fillna('').str.rsplit('/', n=1).str[-1].str.lower()
    served_matches = canonical_served.eq(rows.model.astype('string'))
    rows.loc[served_matches, 'identity_evidence'] = 'row_served_model_match'

    rows['body_hash'] = body_hashes(rows.text)
    bodies = set(rows.body_hash)
    prior_keys = json.loads(prior.read_text())
    pool = locate(root, 'ntl_open_pool.parquet', POOL.split('/')[1])
    official = pd.read_csv(locate(root, 'train.csv', 'name-that-llm'), usecols=['text'])
    public = pd.read_parquet(locate(root, 'public_truth.parquet'), columns=['text'])
    prv_slug = PRV.split('/')[1]
    prv_bodies = pd.read_csv(locate(root, 'passages.csv', prv_slug), usecols=['text'])
    prv_prompts = pd.read_csv(locate(root, 'prompts.csv', prv_slug), usecols=['prompt_body'])
    extra = extra_bodies(root)
    forbidden = {
        'official_body': set(body_hashes(official.text)),
        'full_v18_body': matched_pool(pool, bodies),
        'previous_evaluation_body': set(body_hashes(public.text)) | set(prior_keys['bodies']),
        'prvsiyan_body': set(body_hashes(prv_bodies.text)),
        'stack_extra_source_body': set().union(*extra.values()),
    }
    flags = {name: rows.body_hash.isin(keys) for name, keys in forbidden.items()}
    known_prompts = text_key(prv_prompts.prompt_body.dropna())
    normalized_prompt = text_key(rows.prompt_body)
    flags['same_prvsiyan_prompt'] = normalized_prompt.isin(known_prompts)
    flags['same_kimi_prompt'] = rows.prompt_id.isin(prior_keys['kimi_prompt_ids'])
    flags['duplicate_candidate_body'] = rows.body_hash.duplicated(keep=False)
    flags['conflicting_served_model'] = served.notna() & ~served_matches
    if near_threshold is not None:
        scores = nearest_prompt(normalized_prompt, known_prompts.drop_duplicates())
        flags['near_prvsiyan_prompt'] = scores >= near_threshold
    flags = pd.DataFrame(flags, index=rows.index)

    keep = rows.loc[~flags.any(axis=1)].copy()
    keep['id'] = ['focal-' + body_hash(SOURCE + ':' + str(key))[:24] for key in keep.source_id]
    keep['source_dataset'] = SOURCE
    keep = keep.sort_values('id')
    if not keep.id.is_unique or not set(MODELS[:2]).issubset(keep.model):
        raise ValueError('The retained focal panel lacks unique IDs or a target class')
    report = dict(source=SOURCE, normalized_body='NFKC, case-fold, whitespace and outer bold wrapper',
                  candidates=summarize(rows), retained=summarize(keep),
                  flagged={name: int(mask.sum()) for name, mask in flags.items()},
                  stack_source_matches={path: int(rows.body_hash.isin(keys).sum()) for path, keys in extra.items()},
                  near_prompt_threshold=near_threshold,
                  prompt_scope='Original Prvsiyan prompt strings and known namespaced Kimi prompt IDs',
                  limitations=['Exact-body exclusions do not establish checkpoint training exposure',
                               'Stack extra scan covers all visible source rows, a superset of runtime-selected rows',
                               'Source prompt templates are documented, individual provider receipt and target suffix are absent'])
    output.mkdir(parents=True, exist_ok=True)
    keep[['id', 'text']].to_parquet(output / 'queries.parquet', index=False)
    columns = ['id', 'source_id', 'source_dataset', 'prompt_id', 'model', 'n_words', 'identity_evidence',
               'served_model_id', 'body_hash']
    keep[columns].to_parquet(output / 'truth.parquet', index=False)
    rows[['source_id', 'prompt_id', 'model', 'n_words', 'body_hash']].join(flags).to_parquet(
        output / 'screened_rows.parquet', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['retained']), flush=True)
