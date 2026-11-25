"""Matched 18-label readers trained on two draws from the same public corpus."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paired_attribution import text_key
from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions
from .sibling_specialist import Specialist, predict, train, write_json


ARMS = ('length_matched', 'provenance_balanced')
PANELS = ('gen003', 'kimi_flash')
HELD_PUBLISHERS = ('beroli/', 'markbarney/', 'tadeuhayashida/', 'prvsiyan/',
                   'drjcfmd/', 'elliotthankinson/')
FLAGS = ('flag_kb_proxy', 'flag_kb_proxy_after_ruling', 'flag_train_prompt_regen',
         'flag_near_train_prompt', 'flag_gemini37_routing_risk')
POOL_COLUMNS = ('id', 'text', 'model', 'prompt_canon', 'source_dataset', 'source_file',
                'prompt_evidence', 'gen_route', 'gen_reasoning', 'gen_served_model', *FLAGS)
STRATA = ['model', 'length_band']


def read_inputs(official_path, public_directory, panels):
    official = pd.read_csv(official_path, dtype={'id': str, 'prompt_id': str}).set_index('id')
    official = official.assign(cohort='official_external_only')
    public = read_ids(public_directory / 'public_truth.parquet')
    frames = [official, public]
    for name in PANELS:
        queries = read_ids(panels / name / 'queries.parquet')
        targets = read_ids(panels / name / 'truth.parquet')
        same_ids(queries.index, targets.index, name, ordered=True)
        frames.append(targets.join(queries[['text']]).assign(cohort=name))
    queries = pd.concat([part[['text', 'model', 'prompt_id', 'cohort']] for part in frames],
                        verify_integrity=True)
    return queries.assign(n_words=queries.text.str.split().str.len())


def curate(pool, queries, labels):
    import pyarrow.parquet as pq

    has_aliases = 'also_in' in pq.read_schema(pool).names
    columns = list(POOL_COLUMNS) + (['also_in'] if has_aliases else [])
    rows = pd.read_parquet(pool, columns=columns).dropna(
        subset=['id', 'text', 'prompt_canon', 'source_dataset']).copy()
    for column in ('source_dataset', 'gen_route', 'gen_reasoning', 'gen_served_model'):
        rows[column] = rows[column].fillna('').str.strip().str.lower()
    rows['source_dataset'] = rows.source_dataset.str.removeprefix('https://www.kaggle.com/datasets/')
    source_held = rows.source_dataset.str.startswith(HELD_PUBLISHERS)
    if has_aliases:
        source_held |= rows.also_in.astype(str).str.lower().str.contains('|'.join(HELD_PUBLISHERS), regex=True)
    keys = text_key(rows.text)
    same_body = keys.isin(text_key(queries.text))
    held_prompts = set(rows.loc[same_body, 'prompt_canon']) | set(queries.prompt_id.astype(str))
    served = rows.gen_served_model.str.rsplit('/', n=1).str[-1].str.replace(r'-\d{4}-\d{2}-\d{2}$', '', regex=True)
    rows['returned_id_exact'] = served.eq(rows.model)
    conflict = served.isin(labels) & ~rows.returned_id_exact
    rows['n_words'] = rows.text.str.split().str.len()
    valid = (rows.model.isin(labels) & rows.n_words.between(24, 300) & rows.prompt_canon.ne('')
             & rows.source_dataset.ne('') & ~source_held & ~same_body & ~rows.prompt_canon.isin(held_prompts)
             & rows.prompt_evidence.isin(('own_text', 'own_bank', 'other_text'))
             & ~rows.source_file.fillna('').str.contains(r'train_regen|discriminator|exact_pass', case=False)
             & ~rows[list(FLAGS)].fillna(False).any(axis=1) & ~conflict)
    rows = rows.loc[valid].assign(body_key=keys.loc[valid])
    distinct = ~rows.body_key.duplicated(keep=False)
    rows = (rows.loc[distinct].sort_values(['returned_id_exact', 'id'], ascending=[False, True])
            .drop_duplicates(['source_dataset', 'prompt_canon', 'model']).sort_values('id').reset_index(drop=True))
    rows['length_band'] = np.searchsorted([100, 180], rows.n_words, side='right')
    rows[['gen_route', 'gen_reasoning']] = rows[['gen_route', 'gen_reasoning']].replace('', 'unknown')
    if set(rows.model) != set(labels):
        raise ValueError(f'Missing labels in filtered data: {sorted(set(labels) - set(rows.model))}')
    return rows


def allocate(available, maximum, length_shares):
    count = min(int(available.sum()), maximum)
    quota = np.minimum(available, np.floor(count * np.asarray(length_shares)).astype(int))
    for band in range(3):
        quota[band] += min(int(available[band] - quota[band]), count - int(quota.sum()))
    return quota


def draw(rows, settings):
    available = rows.groupby(STRATA).size().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
    budget = pd.DataFrame([allocate(count, settings['per_class'], settings['length_shares'])
                           for count in available.to_numpy()], index=available.index, columns=available.columns)
    rows = rows.join(budget.stack().rename('quota'), on=STRATA)
    rng = np.random.default_rng(settings['seed'])
    priority = -np.log1p(-rng.random(len(rows)))
    source_keys = STRATA + ['source_dataset']
    source_n = rows.groupby(source_keys).id.transform('size')
    joint_n = rows.groupby(source_keys + ['gen_route', 'gen_reasoning']).id.transform('size')
    adjustment = (source_n + 32.) ** .5 * ((joint_n + 16.) / (source_n + 16.)) ** .25
    samples = {}
    for name, weight in zip(ARMS, (1., adjustment)):
        ordered = rows.assign(priority=priority * weight).sort_values(['priority', 'id'])
        use = ordered.groupby(STRATA, sort=False).cumcount() < ordered.quota
        samples[name] = ordered.loc[use].sort_values('id').reset_index(drop=True)
        realized = samples[name].groupby(STRATA).size().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
        if not realized.equals(budget):
            raise ValueError(f'Quota equality failed for {name}')
    return samples


def provenance_summary(rows):
    keys = STRATA + ['source_dataset', 'gen_route', 'gen_reasoning']
    return dict(rows=len(rows), classes=rows.model.value_counts().to_dict(),
                sources=rows.source_dataset.nunique(), prompts=rows.prompt_canon.nunique(),
                returned_id_exact=int(rows.returned_id_exact.sum()),
                returned_id_exact_by_label=rows.groupby('model').returned_id_exact.sum().to_dict(),
                class_length_counts=rows.groupby(STRATA).size().rename('rows').reset_index().to_dict('records'),
                largest_sources=rows.source_dataset.value_counts().head(12).to_dict(),
                provenance_counts=rows.groupby(keys).size().rename('rows').reset_index().to_dict('records'))


def prepare(config_path, pool, official, public, panels, work, output):
    config = json.loads(config_path.read_text())
    queries = read_inputs(official, public, panels)
    with np.load(public / 'stage_scores.npz', allow_pickle=False) as trace:
        labels = trace['labels'].tolist()
    if len(labels) != 18 or len(set(labels)) != 18 or not queries.model.isin(labels).all():
        raise ValueError('Training and all evaluation sources must share the 18 competition labels')
    available = curate(pool, queries, labels)
    samples = draw(available, config)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(work / 'evaluation.parquet')
    write_json(work / 'labels.json', labels)
    info = dict(available=provenance_summary(available), evaluation=queries.cohort.value_counts().to_dict(),
                training={}, draw_shared_rows=len(set(samples[ARMS[0]].id) & set(samples[ARMS[1]].id)),
                settings=config)
    for arm, rows in samples.items():
        rows[['id', 'model', 'text']].to_parquet(work / f'{arm}.parquet', index=False)
        rows[['id', 'model', 'prompt_canon', 'source_dataset', 'n_words', 'length_band', 'gen_route',
              'gen_reasoning', 'returned_id_exact']].to_parquet(output / f'{arm}_sample.parquet', index=False)
        info['training'][arm] = provenance_summary(rows)
    write_json(output / 'data_report.json', info)
    print('Prepared:', info['evaluation'], {key: info['training'][key]['rows'] for key in ARMS}, flush=True)


def fit(base, config_path, work, output, arm):
    import torch
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    labels = json.loads((work / 'labels.json').read_text())
    rows = pd.read_parquet(work / f'{arm}.parquet')
    queries = read_ids(work / 'evaluation.parquet')
    if torch.cuda.device_count() != 1 or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Each arm requires one assigned T4')
    torch.manual_seed(config['seed'])
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Specialist(base, config, len(labels), device)
    train(model, tokenizer, rows, labels, config, device, focused=False)
    logits = predict(model, tokenizer, queries, labels, config, device)
    pd.DataFrame(logits, index=queries.index, columns=labels).to_parquet(output / f'{arm}_logits.parquet')
    torch.save({name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad},
               output / f'{arm}.pt')


def compare(public_path, work, output, blend):
    evaluation = read_ids(work / 'evaluation.parquet')
    public = read_ids(public_path / 'public_truth.parquet')
    trace, original = read_trace(public_path, public)
    labels = pd.Index(trace['labels'])
    idx = labels.get_indexer(public.model)
    true_idx = labels.get_indexer(evaluation.model)
    rows = evaluation.drop(columns='text')
    applicable = trace['glm_winner'] | trace['fable_guard']
    stage = np.where(applicable[:, None], trace['glm_logp'], trace['sibling_logp'])
    base_top3 = np.argsort(-stage, axis=1, kind='stable')[:, :3]
    old_contains = (base_top3 == idx[:, None]).any(axis=1)
    rows['original'] = pd.Series(original, index=public.index)
    rows['original_contains'] = pd.Series(old_contains, index=public.index, dtype='boolean')
    for arm in ARMS:
        logits = read_ids(output / f'{arm}_logits.parquet')
        same_ids(logits.index, evaluation.index, arm, ordered=True)
        if not logits.columns.equals(labels):
            raise ValueError(f'Label order differs for {arm}')
        ranking = np.argsort(-logits.to_numpy(), axis=1, kind='stable')
        rows[arm] = labels.to_numpy()[ranking[:, 0]]
        rows[f'{arm}_contains'] = (ranking[:, :3] == true_idx[:, None]).any(axis=1)
        consensus = replay_predictions(trace, public.text, external_logits=logits, weight=blend)
        rows[f'{arm}_blend'] = pd.Series(consensus, index=public.index)
    rows.to_parquet(output / 'reader_predictions.parquet')
    results = []
    for cohort, frame in rows.groupby('cohort', sort=True):
        for subset, selected in (('all', frame), ('short', frame[frame.n_words.lt(100)])):
            if selected.empty:
                continue
            comparison = paired_stats(selected, *ARMS, draws=4000)
            by_label = selected.assign(**{f'{arm}_correct': selected[arm].eq(selected.model) for arm in ARMS})
            by_label = by_label.groupby('model').agg(rows=('model', 'size'),
                                                   **{arm: (f'{arm}_correct', 'sum') for arm in ARMS})
            comparison.update(cohort=cohort, subset=subset, candidate_coverage={},
                              per_label_counts=by_label.reset_index().to_dict('records'))
            if selected.original.notna().all():
                wrong = selected.original.ne(selected.model)
                base_has = selected.original_contains.eq(True)
                comparison['baseline_wrong'] = int(wrong.sum())
                comparison['baseline_wrong_covered'] = int((wrong & base_has).sum())
                comparison['fixed_weight_blend'] = {}
                for arm in ARMS:
                    comparison['candidate_coverage'][arm] = {
                        'new_true_top1_on_wrong': int((wrong & ~base_has & selected[arm].eq(selected.model)).sum()),
                        'new_true_top3_on_wrong': int((wrong & ~base_has & selected[f'{arm}_contains']).sum()),
                    }
                    comparison['fixed_weight_blend'][arm] = paired_stats(
                        selected, 'original', f'{arm}_blend', draws=4000)
            results.append(comparison)
    write_json(output / 'evaluation_report.json', dict(diagnostic_blend=blend, results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'config', 'work', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    fit(args.base, args.config, args.work, args.output, args.arm)
