"""Screen three public Fable sources against the fixed reference corpora."""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .focal_source_audit import body_hash, body_hashes, extra_bodies, matched_pool, summarize
from .heldout_source_audit import locate, text_key


TELLS = 'elliotthankinson/ntl-tells-v4/2'
SEYIT = 'seyitkaangunes/ntl-extra-api-passages/2'
TADEU = 'tadeuhayashida/ntl-claude-fable-5-generations/1'
FOCUS = ('claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
         'glm-5.3', 'kimi-k3')
MAX_COSINE = .80


def slug(pin):
    return pin.split('/')[1]


def original_rows(raw, pin, identity, request, route, protocol, served=None):
    rows = raw.loc[raw.model.isin(FOCUS) & raw.text.notna(), ['id', 'prompt_id', 'model', 'text', 'prompt']].copy()
    rows = rows.rename(columns={'id': 'source_id', 'prompt': 'prompt_body'})
    rows['source_dataset'] = pin
    rows['identity_evidence'] = identity.loc[rows.index] if isinstance(identity, pd.Series) else identity
    rows['request_id'] = request.loc[rows.index].astype('string')
    rows['route'] = route.loc[rows.index].fillna('unspecified').astype(str) if isinstance(route, pd.Series) else route
    rows['protocol'] = protocol.loc[rows.index].fillna('unspecified').astype(str) if isinstance(protocol, pd.Series) else protocol
    rows['served_model_id'] = served.loc[rows.index].astype('string') if served is not None else None
    rows['prompt_id'] = slug(pin) + ':' + rows.prompt_id.astype('string')
    rows['n_words'] = rows.text.str.split().str.len()
    rows = rows[rows.n_words.between(40, 250)].copy()
    rows['body_hash'] = body_hashes(rows.text)
    return rows


def tells(root):
    raw = pd.read_parquet(locate(root, 'tells_v4_filtered.parquet', slug(TELLS)))
    passed = pd.read_parquet(locate(root, 'tells_v4_pass.parquet', slug(TELLS)), columns=['id'])
    raw = raw.loc[raw.provider.eq('openrouter_paid')].copy()
    request = raw[['prompt_id', 'variant', 'tw']].astype('string').fillna('').agg('|'.join, axis=1)
    rows = original_rows(raw, TELLS, 'publisher_code_model_prefix_check', request, raw.or_provider, raw.kind)
    return rows.assign(publisher_style_pass=rows.source_id.isin(passed.id))


def seyit(root):
    raw = pd.read_csv(locate(root, 'ntl_extra_passages_claude.csv', slug(SEYIT)), dtype={'prompt_id': str})
    prompts = pd.read_csv(locate(root, 'prompts.csv', slug(SEYIT)), dtype={'prompt_id': str}).set_index('prompt_id')
    raw['prompt'] = raw.prompt_id.map(prompts.prompt)
    request = raw[['prompt_id', 'target_words']].astype('string').fillna('').agg('|'.join, axis=1)
    return original_rows(raw, SEYIT, 'publisher_asserted_provider_match', request, 'anthropic', 'publisher_filtered')


def tadeu(root):
    raw = pd.read_parquet(locate(root, 'gen001_claude_fable_5_filtered.parquet', slug(TADEU)))
    raw['prompt'] = raw['prompt'] if 'prompt' in raw else None
    raw['id'] = raw.response_id
    served = raw.served_model.astype('string').str.rsplit('/', n=1).str[-1].str.lower()
    identity = pd.Series(np.where(served.eq(raw.model).fillna(False), 'row_served_model_match',
                                  'row_served_model_other'), index=raw.index)
    request = raw[['prompt_id', 'variant', 'target_words']].fillna('').astype(str).agg('|'.join, axis=1)
    rows = original_rows(raw, TADEU, identity, request, raw.provider, raw.variant, raw.served_model)
    return rows.loc[rows.source_id.notna() & rows.identity_evidence.eq('row_served_model_match')]


def tells_topics(root):
    source = locate(root, 'tells_v4_all.parquet', slug(TELLS))
    rows = pd.read_parquet(source, columns=['prompt_id', 'text']).dropna()
    return pd.DataFrame({'prompt_id': slug(TELLS) + ':' + rows.prompt_id.astype('string'),
                         'body_hash': body_hashes(rows.text)})


def nearest_bounded(candidates, known):
    known = pd.Series(known).dropna().drop_duplicates()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), dtype=np.float32)
    reference = vectorizer.fit_transform(known)
    distinct = candidates.dropna().drop_duplicates()
    parts = []
    for offset in range(0, len(distinct), 64):
        batch = distinct.iloc[offset:offset + 64]
        score = vectorizer.transform(batch) @ reference.T
        parts.append(pd.Series(score.max(axis=1).toarray().ravel(), index=batch.to_numpy()))
    scores = pd.concat(parts) if parts else pd.Series(dtype=float)
    return candidates.map(scores)


def metrics(rows):
    result = summarize(rows)
    result['by_protocol'] = rows.protocol.value_counts().to_dict()
    result['known_prompt_rows'] = int(rows.prompt_body.notna().sum())
    result['short_fable_by_protocol'] = rows.loc[rows.model.eq('claude-fable-5') & rows.n_words.lt(100),
                                                'protocol'].value_counts().to_dict()
    result['short_fable_matched_requests'] = {}
    short = rows.loc[rows.n_words.lt(100)]
    for partner in FOCUS[1:]:
        matched = short.loc[short.model.isin(('claude-fable-5', partner))].copy()
        counts = 0
        for _, group in matched.groupby(['source_dataset', 'request_id']):
            fable = group.loc[group.model.eq('claude-fable-5'), 'n_words'].to_numpy()
            other = group.loc[group.model.eq(partner), 'n_words'].to_numpy()
            if len(fable) and len(other):
                ratio = np.maximum(fable[:, None], other) / np.minimum(fable[:, None], other)
                counts += int(((ratio <= 1.15) & (np.abs(fable[:, None] - other) <= 12)).any())
        result['short_fable_matched_requests'][partner] = counts
    return result


def run(root, output):
    rows = pd.concat([read(root) for read in (tells, seyit, tadeu)], ignore_index=True)
    if not rows.source_id.notna().all() or rows[['source_dataset', 'source_id']].duplicated().any():
        raise ValueError('Source row identifiers are missing or duplicated')
    rows['publisher_style_pass'] = rows['publisher_style_pass'].astype('boolean')
    source_topics = tells_topics(root)
    bodies = set(rows.body_hash) | set(source_topics.body_hash)
    official = pd.read_csv(locate(root, 'train.csv', 'name-that-llm'), usecols=['text'])
    pool = locate(root, 'ntl_open_pool.parquet', 'name-that-llm-open-pool')
    public = pd.read_parquet(locate(root, 'public_truth.parquet'), columns=['text'])
    existing = pd.read_parquet(locate(root, 'candidate_queries.parquet', 'ntl-heldout-source-audit'), columns=['text'])
    drj = pd.read_parquet(locate(root, 'queries.parquet', 'ntl-focal-source-audit'), columns=['text'])
    stack = extra_bodies(root)
    references = {
        'same_official_body': set(body_hashes(official.text)),
        'same_v18_body': matched_pool(pool, bodies),
        'same_known_eval_body': set(body_hashes(pd.concat([public.text, existing.text, drj.text]))),
        'same_stack_input_body': set().union(*stack.values()),
    }
    flags = pd.DataFrame({name: rows.body_hash.isin(keys) for name, keys in references.items()})
    flags['duplicate_candidate_body'] = rows.body_hash.duplicated(keep=False)
    rows['body_clean'] = ~flags.any(axis=1)
    in_pool_topic = source_topics.loc[source_topics.body_hash.isin(references['same_v18_body']), 'prompt_id']
    flags['same_tells_prompt_with_v18_body'] = rows.prompt_id.isin(in_pool_topic)
    pool_prompts = pd.read_parquet(locate(root, 'prompts.parquet', 'name-that-llm-open-pool'), columns=['prompt_text'])
    prv = pd.read_csv(locate(root, 'prompts.csv', 'prvsiyans-extended-name-that-llm-dataset'), usecols=['prompt_body'])
    source_prompts = pd.read_json(locate(root, 'prompts_new.jsonl', 'ntl-generated-v2'), lines=True)
    known = text_key(pd.concat([pool_prompts.prompt_text, prv.prompt_body, source_prompts.prompt], ignore_index=True))
    canonical = text_key(rows.prompt_body.astype('string')).replace('', pd.NA)
    exact_prompt = canonical.isin(known.dropna())
    rows['nearest_known_prompt'] = nearest_bounded(canonical.where(rows.body_clean), known)
    rows['strict'] = (rows.body_clean & canonical.notna() & ~exact_prompt
                      & ~flags.same_tells_prompt_with_v18_body
                      & rows.nearest_known_prompt.lt(MAX_COSINE).fillna(False))
    rows['same_known_prompt'] = exact_prompt
    rows['explicit_served_id'] = rows.identity_evidence.eq('row_served_model_match')
    rows['id'] = ['fable-source-' + body_hash(ref + ':' + str(key))[:24]
                  for ref, key in zip(rows.source_dataset, rows.source_id, strict=True)]
    by_source = {}
    for pin in (TELLS, SEYIT, TADEU):
        group = rows.loc[rows.source_dataset.eq(pin)]
        by_source[pin] = dict(available=metrics(group), body_clean=metrics(group.loc[group.body_clean]),
                              strict=metrics(group.loc[group.strict]),
                              excluded=flags.loc[group.index].sum().astype(int).to_dict())
    report = dict(max_char_tfidf_cosine=MAX_COSINE, sources=by_source, body_clean=metrics(rows.loc[rows.body_clean]),
                  strict=metrics(rows.loc[rows.strict]),
                  stack_source_matches={name: int(rows.body_hash.isin(keys).sum()) for name, keys in stack.items()},
                  limitations=['Body matches use normalized exact strings.',
                               'Stack source-file matches are conservative and include rows omitted by fitting.',
                               'Checkpoint histories are unavailable. Style-gate failure describes publisher filtering.',
                               'Strict prompts exclude available pool, Prvsiyan and DRJ prompt strings.'])
    output.mkdir(parents=True, exist_ok=True)
    kept = rows.loc[rows.body_clean].sort_values('id')
    kept[['id', 'text']].to_parquet(output / 'queries.parquet', index=False)
    kept.drop(columns='text').to_parquet(output / 'truth.parquet', index=False)
    rows.drop(columns=['text', 'prompt_body']).join(flags).to_parquet(output / 'screened_rows.parquet', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Kept exact-body-clean rows', len(kept), 'of which strict prompts', int(kept.strict.sum()), flush=True)
    return report


def check_v19(root, output):
    source = 'ntl-fable-independent-source-audit'
    rows = pd.read_parquet(locate(root, 'truth.parquet', source))
    queries = pd.read_parquet(locate(root, 'queries.parquet', source))
    if not rows.id.is_unique or not rows.id.equals(queries.id) or not rows.body_clean.all():
        raise ValueError('Expected the frozen v18 body-clean panel in original ID order')
    if not np.array_equal(rows.body_hash, body_hashes(queries.text)):
        raise ValueError('Archived truth hashes differ from the supplied bodies')
    pool = locate(root, 'ntl_open_pool.parquet', 'name-that-llm-open-pool')
    prompts = pd.read_parquet(locate(root, 'prompts.parquet', 'name-that-llm-open-pool'), columns=['prompt_text'])
    source_topics = tells_topics(root)
    overlap = matched_pool(pool, set(rows.body_hash) | set(source_topics.body_hash))
    rows['same_v19_body'] = rows.body_hash.isin(overlap)
    in_pool_topic = source_topics.loc[source_topics.body_hash.isin(overlap), 'prompt_id']
    rows['same_tells_prompt_with_v19_body'] = rows.prompt_id.isin(in_pool_topic)
    candidate_prompts = text_key(rows.prompt_body.astype('string')).replace('', pd.NA)
    known = text_key(prompts.prompt_text)
    rows['nearest_v19_prompt'] = nearest_bounded(candidate_prompts.where(rows.strict & ~rows.same_v19_body), known)
    rows['strict_v19'] = (rows.strict & ~rows.same_v19_body & ~rows.same_tells_prompt_with_v19_body
                          & rows.nearest_v19_prompt.lt(MAX_COSINE).fillna(False))
    report = dict(
        body_clean_v18=metrics(rows),
        body_clean_both=metrics(rows.loc[~rows.same_v19_body]),
        strict_both=metrics(rows.loc[rows.strict_v19]),
        v19_prompt_only_removals=metrics(rows.loc[rows.strict & ~rows.same_v19_body & ~rows.strict_v19]),
        added_v19_matches={pin: metrics(rows.loc[rows.source_dataset.eq(pin) & rows.same_v19_body])
                           for pin in (TELLS, SEYIT, TADEU)},
    )
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'screened_rows.parquet', index=False)
    rows.loc[rows.strict_v19].to_parquet(output / 'truth.parquet', index=False)
    queries.loc[rows.strict_v19].to_parquet(output / 'queries.parquet', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('v19 adds', int(rows.same_v19_body.sum()), 'body matches to the v18-clean panel', flush=True)
    return report
