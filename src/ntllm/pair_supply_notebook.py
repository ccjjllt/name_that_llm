"""Inspect matched hard-pair supply under the actual v18 reader filters."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save


READER = Path('runs/provenance_readers_20261125/kaggle/provenance-readers.ipynb')
OUTPUT = Path('runs/pair_supply_20261125/kaggle_audit')


def build(output, reader):
    prepared = json.loads(reader.read_text())['cells'][0]
    run = '''import json

import pandas as pd

from ntllm.provenance_readers import curate, draw, read_inputs

mount = Path('/kaggle/input')

def asset(slug, name):
    path, = [p for p in mount.rglob(name) if slug in p.parts]
    return path

official = asset('name-that-llm', 'train.csv')
historical = asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent
pool = asset('name-that-llm-open-pool', 'ntl_open_pool.parquet')
inputs = read_inputs(official, historical, Path('panels'))
labels = sorted(inputs.model.unique())
rows = curate(pool, inputs, labels)
arms = draw(rows, json.loads(Path('study.json').read_text()))
fields = ['id', 'source_dataset', 'prompt_canon', 'gen_route', 'gen_reasoning', 'returned_id_exact', 'n_words']
key = ['source_dataset', 'prompt_canon']
pairs = [('gemini-3.7-flash', 'gemini-3.8-flash'), ('gemini-3.6-flash', 'gemini-3.8-flash'),
         ('gpt-5.6-luna', 'gpt-5.6-sol'), ('gpt-5.6-luna', 'gpt-5.6-terra'),
         ('gpt-5.5', 'gpt-5.6-terra'), ('kimi-k3', 'glm-5.3-flash')]

def counts(frame):
    return dict(rows=len(frame), prompts=int(frame.prompt_canon.nunique()),
                by_source=frame.source_dataset.value_counts().to_dict())

def near_short(a, b):
    return a.lt(100) & b.lt(100) & (a - b).abs().le(12) & a.le(1.15 * b) & b.le(1.15 * a)

details = []
summary = {}
for left, right in pairs:
    a, b = [rows.loc[rows.model.eq(label), fields] for label in (left, right)]
    found = a.merge(b, on=key, suffixes=('_left', '_right'), validate='one_to_one')
    length_a, length_b = found.n_words_left, found.n_words_right
    short = near_short(length_a, length_b)
    same_route = found.gen_route_left.eq(found.gen_route_right) & found.gen_route_left.ne('unknown')
    same_effort = found.gen_reasoning_left.eq(found.gen_reasoning_right) & found.gen_reasoning_left.ne('unknown')
    returned = found.returned_id_exact_left & found.returned_id_exact_right
    strict = short & same_route & same_effort & returned
    name = left + ' / ' + right
    found = found.assign(pair=name, matched_short=short, matched_protocol_metadata=strict)
    panel = dict(same_source_topic=counts(found), matched_short=counts(found[short]),
                 both_returned_and_route_effort=counts(found[strict]), realized={})
    for arm, sample in arms.items():
        both = found.id_left.isin(sample.id) & found.id_right.isin(sample.id)
        panel['realized'][arm] = dict(matched_short=counts(found[short & both]),
                                     both_returned_and_route_effort=counts(found[strict & both]))
        found['in_' + arm] = both
    details.append(found)
    summary[name] = panel

vertex_fields = ['id', 'arm', 'prompt_id', 'user_message', 'target_words', 'label', 'model_version', 'finish_reason', 'text']
vertex = pd.read_csv(asset('ntl-vertex-gemini-gen', 'passages.csv'), usecols=vertex_fields,
                     dtype={'id': str, 'prompt_id': str})
original_vertex = vertex.label.value_counts().to_dict()
vertex_ok = vertex.model_version.eq(vertex.label) & vertex.finish_reason.eq('STOP')
vertex_ok &= vertex.user_message.notna() & vertex.user_message.ne('') & vertex.target_words.gt(0)
vertex_ok &= vertex.text.notna() & vertex.text.str.strip().ne('')
vertex = vertex.loc[vertex_ok].copy()
vertex['n_words'] = vertex.text.str.split().str.len()
training_vertex = rows.loc[rows.source_dataset.eq('davidpiln/ntl-vertex-gemini-gen'), ['id', 'model', 'text']]
training_vertex = training_vertex.assign(text=training_vertex.text.str.strip()).rename(
    columns={'id': 'curated_id', 'model': 'label'})
if training_vertex.duplicated(['label', 'text']).any():
    raise ValueError('Curated Vertex bodies must identify a unique row per model')
vertex['text'] = vertex.text.str.strip()
vertex = vertex.merge(training_vertex, on=['label', 'text'], how='left', validate='many_to_one')
fields = ['id', 'arm', 'prompt_id', 'user_message', 'target_words', 'n_words', 'curated_id']
old, new = [vertex.loc[vertex.label.eq(label), fields] for label in pairs[0]]
matched = old.merge(new, on=['arm', 'prompt_id'], suffixes=('_left', '_right'), validate='one_to_one')
identical = matched.user_message_left.eq(matched.user_message_right)
identical &= matched.target_words_left.eq(matched.target_words_right)
short = near_short(matched.n_words_left, matched.n_words_right)
both_curated = matched.curated_id_left.notna() & matched.curated_id_right.notna()

def vertex_counts(mask):
    return dict(pairs=int(mask.sum()), arms=matched.loc[mask, 'arm'].value_counts().to_dict())

vertex_report = dict(raw_rows_by_label=original_vertex, admitted_rows_by_label=vertex.label.value_counts().to_dict(),
                     matched_exact_message=vertex_counts(identical), matched_short=vertex_counts(identical & short),
                     matched_short_in_curated_pool=vertex_counts(identical & short & both_curated), selected={})
for name, sample in arms.items():
    both = matched.curated_id_left.isin(sample.id) & matched.curated_id_right.isin(sample.id)
    vertex_report['selected'][name] = vertex_counts(identical & short & both)

report = dict(curated_rows=len(rows), realized_draws={key: len(frame) for key, frame in arms.items()},
              verified_metadata_by_label=rows.groupby('model').returned_id_exact.sum().to_dict(),
              paired_supply=summary, vertex_exact_user_message=vertex_report,
              limitation='Canonical prompt, reported route and effort equality do not establish identical requests')
out = Path('/kaggle/working/pair_supply')
out.mkdir(exist_ok=True)
pd.concat(details, ignore_index=True).to_parquet(out / 'pair_ids.parquet', index=False)
matched.drop(columns=['user_message_left', 'user_message_right']).assign(
    identical_message=identical, matched_short=short, both_curated=both_curated
).to_parquet(out / 'vertex_pairs.parquet', index=False)
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Matched pair supply audit complete', len(rows), flush=True)
'''
    compile(''.join(prepared['source']), '<reader-data-preparation>', 'exec')
    compile(run, '<pair-supply>', 'exec')
    metadata = dict(id='zhuating/ntl-pair-supply-audit', title='NTL Pair Supply Audit', code_file='pair-supply.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/18', 'davidpiln/ntl-vertex-gemini-gen/2'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1'])
    save(output, metadata, dict(cells=[prepared, cell(run)],
                                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                                nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--reader', type=Path, default=READER)
    args = parser.parse_args()
    build(args.output, args.reader)
