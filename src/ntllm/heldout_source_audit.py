"""Audit body overlap and prompt metadata for four public panel sources."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DATASETS = (
    'crtproduction/name-that-llm-open-pool/18',
    'enaimaan/ntl-openai-generated-responses/3',
    'tadeuhayashida/ntl-gen003-claude-panel/1',
    'coltondowning/ntl-panel-completion/7',
    'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13',
)
EVAL_FILES = (
    'runs/fingerprint_q4_20261123/gen003_blind/queries.parquet',
    'runs/effort_pair_20261124/data/holdout/queries.parquet',
    'runs/kimi_flash_20261125/data/holdout/queries.parquet',
    'runs/terra_pair_20261124/data/holdout/queries.parquet',
)
KIMI_EVAL_INDEX = 'runs/kimi_flash_20261125/data/holdout/truth.parquet'
GPT_MODELS = ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra')
CLAUDE_MODELS = ('claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5')


def text_key(values):
    return (values.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\*\*(.*)\*\*$', r'\1', regex=True)
            .str.replace(r'\s+', ' ', regex=True).str.strip())


def locate(root, name, slug=None):
    paths = [p for p in root.rglob(name) if slug is None or slug in p.parts]
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {slug or "mounted"} {name}, found {paths}')
    return paths[0]


def gpt(root):
    folder = locate(root, 'lineage-aliases.json').parent
    data = pd.read_parquet(folder / 'train.parquet')
    aliases = json.loads((folder / 'lineage-aliases.json').read_text())
    ledger = pd.read_json(folder / 'prompts.jsonl', lines=True)
    exact_messages = ledger.loc[ledger.messages.notna(), 'response_id'].astype(str)
    keep = data.model.isin(GPT_MODELS) & data.kind.eq('new') & data.id.astype(str).isin(exact_messages)
    rows = data[keep].copy()
    canon = rows.cv_group.astype('string')
    canon = canon.map(aliases).fillna(canon).astype('string')
    return rows.assign(
        source_id=rows.id.astype(str), prompt_id='enaimaan:' + canon,
        route='openai_published', served_id=pd.NA, provenance='publisher_serving_identity_check',
        protocol='additional_messages_new', source_dataset=DATASETS[1],
    ), len(data)


def claude(root):
    data = pd.read_parquet(locate(root, 'ntl_gen003_claude_panel.parquet'))
    rows = data[data.model.isin(CLAUDE_MODELS) & data.model.eq(data.served_model) & data.finish.eq('stop')]
    return rows.assign(
        source_id=rows.id.astype(str), prompt_id='pool:' + rows.prompt_canon.astype('string'),
        route='anthropic', served_id=rows.served_model, provenance='row_served_id',
        protocol=rows.audit_hard_pass.map({True: 'style_audit_pass', False: 'style_audit_fail'}),
        source_dataset=DATASETS[2],
    ), len(data)


def colton(root):
    data = pd.read_csv(locate(root, 'ntl_panel_completion.csv'), low_memory=False)
    expected = data.model.map({'gemini-3.7-flash': 'google/gemini-3.7-flash',
                               'qwen3.8-max': 'qwen/qwen3.8-max-0902'})
    rows = data[expected.notna() & data.gen_served_model.eq(expected)].copy()
    route = rows.gen_route if 'gen_route' in rows else 'openrouter_pinned_published'
    source_id = rows.id.astype(str) if 'id' in rows else rows.prompt_canon.astype(str) + ':' + rows.model
    return rows.assign(
        source_id=source_id, prompt_id='pool:' + rows.prompt_canon.astype('string'),
        route=route, served_id=rows.gen_served_model, provenance='row_served_id',
        protocol='known_typography_shift', source_dataset=DATASETS[3],
    ), len(data)


def prvsiyan(root):
    data = pd.read_csv(locate(root, 'passages.csv', 'prvsiyans-extended-name-that-llm-dataset'), low_memory=False)
    alias = {'gpt-5.5-2026-04-23': 'gpt-5.5', 'qwen3.8-max-0902': 'qwen3.8-max'}
    served = data.served_model_id.fillna('').str.rsplit('/', n=1).str[-1].replace(alias)
    direct_37 = data.model.ne('gemini-3.7-flash') | data.provider_route.eq('google')
    keep = (data.split.eq('new') & data.passes_funnel.eq(True) & data.finish_reason_norm.eq('stop')
            & data.provider_route.isin(['openrouter', 'google', 'anthropic', 'openai']) & direct_37
            & data.requested_model_id.eq(data.served_model_id) & served.eq(data.model))
    rows = data[keep]
    return rows.assign(
        source_id=rows.id.astype(str), prompt_id='prvsiyan:' + rows.prompt_id.astype('string'),
        route=rows.provider_route, served_id=rows.served_model_id, provenance='row_requested_and_served_id',
        protocol=rows.split, source_dataset=DATASETS[4],
    ), len(data)


def summarize(rows):
    prompt_support = rows.groupby('prompt_id').model.nunique()
    return {
        'rows': len(rows),
        'prompts': rows.prompt_id.nunique(),
        'same_source_prompt_with_multiple_labels': int(prompt_support.ge(2).sum()),
        'prompt_count_by_number_of_distinct_labels': prompt_support.value_counts().sort_index().to_dict(),
        'by_label': rows.model.value_counts().to_dict(),
        'short_40_99_by_label': rows.loc[rows.n_words.lt(100), 'model'].value_counts().to_dict(),
        'by_route': rows.route.fillna('unreported').value_counts().to_dict(),
        'by_protocol': rows.protocol.value_counts().to_dict(),
        'known_open_pool_prompt_rows': int(rows.pool_prompt_known.sum()),
    }


def prvsiyan_topics(root, rows, prior_index):
    before = pd.read_csv(locate(root, 'prompts.csv', 'prvsiyans-extended-name-that-llm-dataset'),
                         usecols=['prompt_id', 'prompt_body', 'split'])
    before = before[before.split.eq('new')]
    assert before.prompt_id.is_unique
    body = rows.prompt_id.str.removeprefix('prvsiyan:').map(before.set_index('prompt_id').prompt_body)
    if body.isna().any():
        raise ValueError('Missing Prvsiyan original prompt body for a retained source row')
    registry = pd.read_parquet(locate(root, 'prompts.parquet', 'name-that-llm-open-pool'), columns=['prompt_text'])
    exact_pool_texts = set(text_key(registry.prompt_text.dropna()))
    same_prompt = text_key(body).isin(exact_pool_texts)
    overlap = {'row_count_exact_v18_prompt_body': int(same_prompt.sum()),
               'prompt_count_exact_v18_prompt_body': rows.loc[same_prompt, 'prompt_id'].nunique()}
    if prior_index:
        previous = pd.read_parquet(prior_index, columns=['prompt_id', 'source_dataset'])
        previous = previous[previous.source_dataset.eq(DATASETS[4].rsplit('/', 1)[0])]
        prior = rows.prompt_id.isin(previous.prompt_id)
        overlap.update(row_count_same_prior_kimi_prompt=int(prior.sum()),
                       prompt_count_same_prior_kimi_prompt=rows.loc[prior, 'prompt_id'].nunique(),
                       exact_clean_and_without_prior_kimi_prompt=summarize(rows[~prior]))
    return overlap


def run(root, eval_body_files, out, prior_index=None):
    official = pd.read_csv(locate(root, 'train.csv', 'name-that-llm'), usecols=['text', 'model'])
    pool = pd.read_parquet(locate(root, 'ntl_open_pool.parquet'), columns=['text', 'prompt_canon'])
    admitted = pd.read_csv(locate(root, 'passages_clean.csv'), usecols=['text', 'dataset_version_added'])
    admitted = admitted[pd.to_numeric(admitted.dataset_version_added, errors='coerce').isin([1, 2, 4, 5])]
    public = [pd.read_parquet(locate(root, 'public_truth.parquet'), columns=['text']).text]
    public += [pd.read_parquet(path, columns=['text']).text for path in eval_body_files]
    known_keys = {
        'official': set(text_key(official.text)),
        'full_pool_v18': set(text_key(pool.text)),
        'prvsiyan_clean_versions_used_by_stack': set(text_key(admitted.text)),
        'earlier_public_evaluations': set(text_key(pd.concat(public, ignore_index=True))),
    }
    pool_prompts = set(pool.prompt_canon.dropna().astype(str))
    fields = ['source_dataset', 'source_id', 'text', 'model', 'prompt_id', 'route', 'served_id', 'provenance', 'protocol']
    frames, audit = [], {}
    for reader in (gpt, claude, colton, prvsiyan):
        rows, total = reader(root)
        rows = rows.loc[rows.text.notna() & rows.prompt_id.notna() & rows.model.isin(official.model), fields].copy()
        rows['n_words'] = rows.text.str.split().str.len()
        rows = rows[rows.n_words.between(40, 250)].copy()
        rows['body_key'] = text_key(rows.text)
        rows['pool_prompt_known'] = (rows.prompt_id.str.startswith('pool:')
                                     & rows.prompt_id.str.removeprefix('pool:').isin(pool_prompts))
        rejected = pd.DataFrame({name: rows.body_key.isin(keys) for name, keys in known_keys.items()})
        audit[reader.__name__] = dict(source=DATASETS[len(audit) + 1], raw_rows=total,
                                      provenance_length_ready=summarize(rows),
                                      exact_body_matches=rejected.sum().astype(int).to_dict())
        kept = rows.loc[~rejected.any(axis=1)].assign(reader=reader.__name__)
        frames.append(kept)
        print(reader.__name__, total, len(rows), len(kept), flush=True)

    joined = pd.concat(frames, ignore_index=True)
    duplicate = joined.body_key.duplicated(keep=False)
    report = {'exact_cross_candidate_duplicate_rows': int(duplicate.sum()), 'sources': audit,
              'definition': 'NFKC, case-fold, whitespace collapsed, whole-passage bold wrapper stripped',
              'limitations': ['Body overlap is exact after normalization', 'Canonical prompt links available for pool IDs',
                              'Checkpoint training corpus is unknown', 'This audit does not select final held-out panels']}
    kept = joined[~duplicate].copy()
    kept['id'] = ['source-' + hashlib.sha256((ref + ':' + key).encode()).hexdigest()[:24]
                  for ref, key in zip(kept.source_dataset, kept.source_id)]
    if kept.empty or not kept.id.is_unique:
        raise ValueError('Expected nonempty candidate queries with unique source IDs')
    kept = kept.sort_values('id')
    for name in audit:
        audit[name]['candidates'] = summarize(kept[kept.reader.eq(name)])
    audit['prvsiyan']['prompt_check'] = prvsiyan_topics(root, kept[kept.reader.eq('prvsiyan')], prior_index)
    out.mkdir(parents=True, exist_ok=True)
    kept[['id', 'text']].to_parquet(out / 'candidate_queries.parquet', index=False)
    kept.drop(columns=['text', 'body_key']).to_parquet(out / 'candidate_truth.parquet', index=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--older-queries', nargs='*', type=Path, default=[])
    parser.add_argument('--prior-kimi-index', type=Path)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/source_audit'))
    args = parser.parse_args()
    run(args.input, args.older_queries, args.out, args.prior_kimi_index)
