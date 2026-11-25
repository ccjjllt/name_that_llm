"""Measure the supply of short four-model groups under recorded request equality."""

from collections import Counter
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd

from .peer_context_report import read_ids, same_ids


GPT = ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra')
CLAUDE = ('claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5')
REQUEST_OPTIONS = ('target_words', 'instructions', 'system', 'temperature', 'top_p', 'top_k', 'max_output_tokens',
                   'max_tokens', 'reasoning', 'reasoning_effort', 'thinking', 'verbosity',
                   'presence_penalty', 'frequency_penalty', 'seed', 'response_format',
                   'request_params', 'params', 'settings', 'options')


def request_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def gpt_ledger(path):
    details = []
    keys, option_presence = Counter(), Counter()
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            keys.update(row.keys())
            messages = row.get('messages')
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except json.JSONDecodeError:
                    continue
            if not isinstance(messages, list) or not messages:
                continue
            if any(not isinstance(message, dict) or 'role' not in message for message in messages):
                continue
            option = {key: row[key] for key in REQUEST_OPTIONS if key in row}
            option_presence.update(option.keys())
            details.append(dict(source_id=str(row['response_id']), message_key=request_hash(messages),
                                request_key=request_hash([messages, option]),
                                same_target_key=request_hash([messages, row.get('target_words')]),
                                provider_model=row.get('provider_model'), reasoning=row.get('reasoning'),
                                request_status=row.get('request_status'), settings_match=row.get('settings_match')))
    data = pd.DataFrame(details)
    if data.empty or not data.source_id.is_unique:
        raise ValueError('Ordered-message ledger must have exactly one record per response ID')
    return data.set_index('source_id'), dict(ledger_keys=dict(keys), recorded_option_keys=dict(option_presence),
                                             digest_convention='Exact ordered messages, JSON object keys sorted')


def comparable_groups(rows, key):
    grouped = []
    for actor, part in rows.groupby(key, sort=True, dropna=True):
        rows_by_model = {label: values for label, values in part.groupby('model', sort=True)}
        if len(rows_by_model) < 2:
            continue
        ranked = part.sort_values(['n_words', 'id'])
        best = ()
        # A length window may contain multiple provider records for one label.
        # Keep the maximum-cardinality valid subset, then prefer the least spread.
        for position, low in enumerate(ranked.n_words):
            maximum = min(low + 12, 1.15 * low)
            window = ranked.iloc[position:].loc[lambda x: x.n_words.le(maximum)]
            group = tuple(window.drop_duplicates('model').sort_values('id').id)
            span = int(window.loc[window.id.isin(group), 'n_words'].max() - low)
            if not best or (len(group), -span) > (len(best[0]), -best[1]):
                best = (group, span)
        maximum_group, spread = best
        chosen = part.loc[part.id.isin(maximum_group)]
        grouped.append(dict(group=str(actor), canonical_topics=int(part.prompt_id.nunique()),
                            available_models=len(rows_by_model), near_short_size=len(chosen),
                            ids=sorted(chosen.id), models=sorted(chosen.model.unique()), spread=spread))
    return grouped


def summarize(rows, fields):
    report = dict(rows=len(rows), models=rows.model.value_counts().to_dict(),
                  short_rows=int(rows.n_words.lt(100).sum()), groups={})
    records = []
    short = rows.loc[rows.n_words.between(40, 99)]
    for key in fields:
        groups = comparable_groups(short, key)
        sizes = Counter(item['near_short_size'] for item in groups)
        pairs = Counter(' / '.join(pair) for group in groups for pair in combinations(group['models'], 2))
        report['groups'][key] = dict(
            short_groups_with_multiple_models=len(groups),
            valid_k_way={str(size): sizes[size] for size in (2, 3, 4)},
            pairs_from_largest_valid_group=dict(pairs),
            short_groups_multiple_canonical_topics=sum(item['canonical_topics'] > 1 for item in groups),
        )
        records.extend(dict(item, grouping=key) for item in groups if item['near_short_size'] >= 2)
    return report, records


def run(audit, gpt_data, claude_data, out):
    queries = read_ids(audit / 'candidate_queries.parquet')
    truth = read_ids(audit / 'candidate_truth.parquet')
    same_ids(queries.index, truth.index, 'Frozen audited source assets', ordered=True)
    table = truth.join(queries)
    ledger, ledger_meta = gpt_ledger(gpt_data / 'prompts.jsonl')
    gpt = table.loc[table.reader.eq('gpt') & table.model.isin(GPT)].reset_index()
    raw_gpt = pd.read_parquet(gpt_data / 'train.parquet', columns=['id', 'model', 'text', 'kind'])
    raw_gpt['id'] = raw_gpt.id.astype(str)
    chosen = raw_gpt.loc[raw_gpt.id.isin(gpt.source_id)].set_index('id', verify_integrity=True).loc[gpt.source_id]
    if not np.array_equal(chosen.model, gpt.model) or not np.array_equal(chosen.text, gpt.text):
        raise ValueError('Audited GPT body and publisher logs do not match')
    if not chosen.kind.eq('new').all() or not gpt.source_id.isin(ledger.index).all():
        raise ValueError('Audited GPT messages should match one new provider row')
    gpt = gpt.join(ledger, on='source_id', validate='many_to_one')
    if len(gpt) != 5237:
        raise ValueError('The frozen GPT source changed')
    if not gpt.provider_model.eq(gpt.model).all():
        raise ValueError('GPT ledger model names differ from the audited publisher labels')
    gpt_report, gpt_rows = summarize(gpt, ('prompt_id', 'message_key', 'same_target_key', 'request_key'))
    gpt_report.update(ledger_meta)
    gpt_report['reported_reasoning_by_model'] = {
        model: part.reasoning.value_counts(dropna=False).to_dict() for model, part in gpt.groupby('model')
    }
    gpt_report['request_status'] = gpt.request_status.value_counts().to_dict()
    gpt_report['settings_match'] = gpt.settings_match.value_counts().to_dict()
    gpt_report['identity'] = 'Provider identity checked by publisher. Audit has no independent per-row returned model ID.'

    claude = table.loc[table.reader.eq('claude') & table.model.isin(CLAUDE)].reset_index()
    claude_raw = pd.read_parquet(claude_data / 'ntl_gen003_claude_panel.parquet')
    raw_columns = claude_raw.columns.tolist()
    claude_raw['id'] = claude_raw.id.astype(str)
    selected = claude_raw.loc[claude_raw.id.isin(claude.source_id)].set_index('id', verify_integrity=True).loc[claude.source_id]
    if not (np.array_equal(selected.model, claude.model) and np.array_equal(selected.text, claude.text)
            and selected.model.eq(selected.served_model).all() and selected.finish.eq('stop').all()):
        raise ValueError('Audited Claude bodies or returned labels differ from the recorded source')
    if len(claude) != 5639:
        raise ValueError('The frozen Claude source changed')
    settings = selected[['prompt', 'system_prompt', 'target_words']]
    # Fingerprint logged fields without inferring sampling parameters or a
    # full provider request from the common canonical topic.
    claude['recorded_text_and_target'] = [
        request_hash([prompt, system, None if pd.isna(target) else int(target)])
        if isinstance(prompt, str) and isinstance(system, str) else None
        for prompt, system, target in settings.itertuples(index=False, name=None)
    ]
    claude['recorded_settings'] = [
        request_hash([fields, 2000 if model == 'claude-opus-5' else 700]) if fields else None
        for fields, model in zip(claude.recorded_text_and_target, claude.model, strict=True)
    ]
    claude_report, claude_rows = summarize(claude, ('prompt_id', 'recorded_text_and_target', 'recorded_settings'))
    claude_report.update(provider_schema=raw_columns,
                         rows_with_recorded_text_and_target=int(claude.recorded_text_and_target.notna().sum()),
                         reported_token_budgets={model: 2000 if model == 'claude-opus-5' else 700 for model in CLAUDE},
                         style_screen_by_model={
                             model: part.protocol.value_counts().to_dict()
                             for model, part in claude.groupby('model')
                         },
                         identity='Matches published per-row returned model and stop reason.',
                         grouping='Canonical topics and exact published prompt, system prompt, target word count. The settings arm separates the author-documented token budgets. Raw API options were not archived.')
    out.mkdir(parents=True, exist_ok=True)
    archive = [dict(item, source='enaimaan') for item in gpt_rows]
    archive += [dict(item, source='tadeuhayashida') for item in claude_rows]
    pd.DataFrame(archive).to_parquet(out / 'groups.parquet', index=False)
    report = dict(gpt=gpt_report, claude=claude_report,
                  short_group='40–99 words. Maximum difference 12 words and max/min at most 1.15.',
                  limits='Published metadata suffice only for the request fields explicitly logged.')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
