"""Find authenticated GPT cohorts in the frozen Prvsiyan request archive."""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .heldout_source_audit import locate, text_key
from .matched_request_supply import GPT, comparable_groups, request_hash


PRV = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
SLUG = PRV.split('/')[1]
POOL = 'crtproduction/name-that-llm-open-pool/18'
MODELS = (*GPT, 'gpt-5.5')
DIRECT_API = frozenset(('https://api.openai.com/v1/chat/completions', 'https://api.openai.com/v1/responses'))


def receive(path, ids):
    rows = []
    with path.open() as records:
        for line in records:
            item = json.loads(line)
            if item.get('call_id') not in ids:
                continue
            req = item.get('request') or {}
            raw = req.get('body')
            response = (item.get('response') or {}).get('json') or {}
            if not (isinstance(raw, dict) and isinstance(response, dict)):
                continue
            # Keep the entire logged body except the top-level model selector.
            options = {key: value for key, value in raw.items() if key != 'model'}
            effort = raw.get('reasoning_effort') or (raw.get('reasoning') or {}).get('effort')
            recorded_input = raw.get('messages') or raw.get('input')
            if not recorded_input:
                continue
            tier = response.get('service_tier', 'unreported')
            endpoint = req.get('url')
            rows.append(dict(call_id=item['call_id'], request_id=response.get('id'), request_url=endpoint,
                             requested_raw_model=raw.get('model'), returned_raw_model=response.get('model'),
                             identity_ok=(item.get('identity') or {}).get('ok') is True,
                             receipt_text=item.get('text'), receipt_status=item.get('status'), service_tier=tier,
                             raw_temperature=raw.get('temperature'), raw_effort=effort,
                             recorded_options=list(sorted(options)),
                             input_key=request_hash(recorded_input),
                             request_key=request_hash([endpoint, options, tier])))
    return pd.DataFrame(rows)


def need_inputs(root):
    pool = locate(root, 'ntl_open_pool.parquet', 'name-that-llm-open-pool')
    official = locate(root, 'train.csv', 'name-that-llm')
    public = locate(root, 'public_truth.parquet', 'ntl-0719-public-decision-trace')
    source = locate(root, 'candidate_truth.parquet', 'ntl-heldout-source-audit')
    known = pd.read_parquet(public, columns=['text', 'cohort'])
    held_truth = pd.read_parquet(source, columns=['id', 'source_dataset', 'prompt_id'])
    held_queries = pd.read_parquet(source.with_name('candidate_queries.parquet'), columns=['id', 'text'])
    held = held_truth.loc[held_truth.source_dataset.eq(PRV)].merge(held_queries, on='id', validate='one_to_one')
    if len(held) != 895:
        raise ValueError(f'Expected 895 frozen Prvsiyan rows, got {len(held)}')
    reference = {
        'official_train_body': pd.read_csv(official, usecols=['text']).text,
        'c1_body': known.loc[known.cohort.eq('development'), 'text'],
        'c2_body': known.loc[known.cohort.eq('evaluation'), 'text'],
        'prvsiyan_eval_body': held.text,
        'v18_body': pd.read_parquet(pool, columns=['text']).text,
    }
    return ({name: set(text_key(values.dropna())) for name, values in reference.items()},
            set(held.prompt_id.str.removeprefix('prvsiyan:')),
            set(text_key(pd.read_parquet(pool.with_name('prompts.parquet'), columns=['prompt_text'])
                         .prompt_text.dropna())))


def select_groups(rows, key, expected):
    chosen = rows.loc[rows.model.isin(expected)]
    complete = chosen.groupby(key, sort=True).model.nunique().eq(len(expected))
    short = chosen.loc[chosen.n_words.between(40, 99) & chosen[key].isin(complete.index[complete])]
    short_complete = short.groupby(key).model.nunique().eq(len(expected))
    matched = [group for group in comparable_groups(short, key) if group['near_short_size'] == len(expected)]
    return dict(complete_groups=int(complete.sum()), all_four_or_five_short=int(short_complete.sum()),
                near_equal_short_groups=len(matched), near_equal_short_prompt_groups=int(chosen.loc[
                    chosen.id.isin({identifier for group in matched for identifier in group['ids']}),
                    'prompt_id'].nunique())), matched


def report_protocol(rows, inventories, groups):
    for split, protocol in rows.groupby('split', sort=True):
        summary = dict(rows=len(protocol), models=protocol.model.value_counts().to_dict(),
                       short_by_model=protocol.loc[protocol.n_words.between(40, 99)].model.value_counts().to_dict(),
                       service_tier=protocol.service_tier.value_counts(dropna=False).to_dict(),
                       request_fields=dict(Counter(k for fields in protocol.recorded_options for k in fields)),
                       effort_by_model={model: part.raw_effort.fillna('omitted').value_counts().to_dict()
                                        for model, part in protocol.groupby('model')},
                       exact_exposure_counts=protocol.filter(regex='^same_').sum().astype(int).to_dict(), arms={})
        for tag, expected in (('four', GPT), ('five_including_gpt55', MODELS)):
            for key in ('prompt_id', 'input_key', 'request_key'):
                stages = {'all_authenticated': protocol, 'exclude_eval_and_reconstructed_topics': protocol[protocol.training_topic_clean],
                          'also_no_known_v18_text_or_exact_prompt': protocol[protocol.reference_clean]}
                if key == 'request_key':
                    stages['explicit_effort_no_known_exposure'] = protocol[protocol.reference_clean & protocol.raw_effort.notna()]
                for mask, subset in stages.items():
                    arm = f'{tag}__{key}__{mask}'
                    summary['arms'][arm], matches = select_groups(subset, key, expected)
                    groups.extend(dict(split=split, arm=arm, **item) for item in matches)
        inventories[split] = summary


def run(root=Path('/kaggle/input'), output=Path('/kaggle/working/gpt_request_groups')):
    folder = locate(root, 'passages.csv', SLUG).parent
    source = pd.read_csv(folder / 'passages.csv', low_memory=False)
    target = source.loc[source.model.isin(MODELS) & source.provider_route.eq('openai')].copy()
    receipt = receive(folder / 'provenance.jsonl', set(target.call_id))
    joined = target.merge(receipt, on='call_id', validate='one_to_one', how='left')
    checks = pd.DataFrame(dict(
        direct_endpoint=joined.request_url.isin(DIRECT_API),
        receipt_label=joined.requested_raw_model.eq(joined.requested_model_id)
        & joined.returned_raw_model.eq(joined.served_model_id)
        & joined.returned_raw_model.replace({'gpt-5.5-2026-04-23': 'gpt-5.5'}).eq(joined.model),
        receipt_id=joined.request_id.eq(joined.response_id), identity=joined.identity_ok.eq(True),
        body=joined.receipt_text.eq(joined.text), stop=joined.finish_reason_norm.eq('stop'),
        clean=~joined[['is_refusal', 'is_truncated', 'is_empty', 'has_markdown']].fillna(True).any(axis=1),
    ))
    rows = joined.loc[checks.all(axis=1)].copy()
    rows['n_words'] = rows.text.str.split().str.len()
    rows = rows[rows.n_words.between(40, 250)].copy()
    bodies, held_prompts, pool_prompt_texts = need_inputs(root)
    body_keys = text_key(rows.text)
    for name, keys in bodies.items():
        rows['same_' + name] = body_keys.isin(keys)
    prompt_registry = pd.read_csv(folder / 'prompts.csv', usecols=['prompt_id', 'prompt_body']).drop_duplicates()
    resolution = prompt_registry.groupby('prompt_id').prompt_body.nunique().eq(1)
    registry = prompt_registry.loc[prompt_registry.prompt_id.isin(resolution.index[resolution])].drop_duplicates('prompt_id')
    prompt_key = rows.prompt_id.map(registry.set_index('prompt_id').prompt_body)
    rows['same_v18_prompt_exact'] = text_key(prompt_key).isin(pool_prompt_texts)
    rows['same_prvsiyan_eval_prompt'] = rows.prompt_id.isin(held_prompts)
    rows['same_official_reconstructed_prompt'] = rows.split.isin(['train_regen', 'train_regen_w1']) | rows.source_prompt_id.notna()
    held_exposures = [col for col in rows.columns if str(col).startswith('same_') and 'v18' not in col]
    rows['training_topic_clean'] = ~rows[held_exposures].any(axis=1)
    rows['reference_clean'] = (rows.training_topic_clean & ~rows.same_v18_body & ~rows.same_v18_prompt_exact & prompt_key.notna())
    inventories, groups = {}, []
    report_protocol(rows, inventories, groups)
    report = dict(source=PRV, reference=POOL, prospective_groups=bool(groups), cohort_short_criteria='40–99 words, spread <= 12 and ratio <= 1.15',
                  csv_direct_openai_rows=len(target), checks=checks.sum().astype(int).to_dict(),
                  authenticated_40_250=len(rows), by_split=inventories,
                  caveats=['Exact match compares archived request body excluding only its model selector, endpoint and recorded service tier.',
                           'Absent effort or temperature are provider defaults and do not establish equal effective settings.',
                           'Prvsiyan is an already evaluated source. Cross-topic clean groups are possible training inputs after freezing evaluation topics, never independent-source validation.',
                           'v18, official and prior-evaluation body overlaps use normalized exact matches. Reconstructed competition topics and known Prvsiyan prompt IDs are excluded explicitly.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fields = ['split', 'arm', 'group', 'canonical_topics', 'available_models', 'near_short_size', 'ids', 'models', 'spread']
    pd.DataFrame(groups, columns=fields).to_parquet(output / 'matched_groups.parquet', index=False)
    rows.drop(columns=['text', 'receipt_text', 'prompt_sent']).to_parquet(output / 'receipts_screened.parquet', index=False)
    print('Authenticated passages', len(rows), 'possible near length group/arm records', len(groups), flush=True)
