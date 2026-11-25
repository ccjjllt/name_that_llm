"""Replay frozen style logits on both exact Sibling Odds bases without refitting."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/style_topic_projection_20261125/kaggle_frozen_transfer')


def build(output, gradient):
    files = ('reader_transfer', 'member_evidence_transfer', 'member_evidence', 'peer_context_report',
             'sibling_odds_replay')
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in files:
        code = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({index!r}))\n"
    run = f"""import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ntllm.member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan
from ntllm.peer_context_report import paired_stats, read_ids
from ntllm.reader_transfer import read_logits
from ntllm.sibling_odds_replay import replay_predictions

mount = Path('/kaggle/input')

def asset(slug, file):
    found, = [path for path in mount.rglob(file) if slug in path.parts]
    return found

sources = {{
    'roberta': ('ntl-styledistance-fp32', 'roberta_logp.parquet'),
    'style_linear': ('ntl-styledistance-fp32', 'style_logp.parquet'),
    'single_gaussian': ('ntl-style-mixture-audit', 'single_logp.parquet'),
    'mixture': ('ntl-style-mixture-audit', 'mixture_logp.parquet'),
    'shuffled_projection': ('ntl-style-topic-projection', 'shuffled_prompt_projection_logp.parquet'),
    'real_projection': ('ntl-style-topic-projection', 'real_prompt_projection_logp.parquet'),
}}
comparisons = [('roberta', 'style_linear'), ('single_gaussian', 'mixture'), ('style_linear', 'mixture'),
               ('shuffled_projection', 'real_projection'), ('style_linear', 'real_projection')]
if {gradient!r}:
    for arm in ('random_13579', 'random_24680', 'random_112525', 'relevance'):
        sources[arm] = ('ntl-gradient-selection-audit', arm + '_logp.parquet')
    comparisons.extend((arm, 'relevance') for arm in ('style_linear', 'random_13579', 'random_24680', 'random_112525'))

historical = read_ids(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'))
holdout = read_prvsiyan(asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent, Path('prior_kimi.parquet'))
historical_trace = asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent
held_trace = asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent
overlap = json.loads((held_trace / 'body_overlap.json').read_text())
exposed = {{row for key in EXPOSURE_KEYS for row in overlap.get(key, [])}}
if not exposed.issubset(holdout.index):
    raise ValueError('Runtime overlap references unknown query IDs')

def replay(truth, folder):
    trace, baseline, *_ = read_capture(folder, truth)
    labels = trace['labels']
    cols = ['model', 'prompt_id', 'n_words', 'cohort']
    rows = truth[cols].assign(baseline=baseline, strict=truth.get('strict_topic_holdout', False))
    rows['runtime_match'] = rows.index.isin(exposed)
    codes = pd.Index(labels).get_indexer(rows.model)
    union = np.zeros(len(rows), dtype=bool)
    for stage in ('stack_logp', 'glm_logp', 'sibling_logp'):
        top3 = np.argsort(-trace[stage], axis=1, kind='stable')[:, :3]
        union |= (top3 == codes[:, None]).any(axis=1)
    rows['base_union_top3'] = union
    for arm, ref in sources.items():
        scores = read_logits(asset(*ref), truth.index, labels)
        rank = np.argsort(-scores.to_numpy(), axis=1, kind='stable')[:, :3]
        rows[arm + '_standalone'] = labels[rank[:, 0]]
        rows[arm + '_top3'] = (rank == codes[:, None]).any(axis=1)
        rows[arm] = replay_predictions(trace, truth.text, external_logits=scores, weight=.15)
    return rows

def measure(rows):
    missing = rows.baseline.ne(rows.model) & ~rows.base_union_top3
    return dict(
        rows=len(rows), missed_by_stages=int(missing.sum()),
        paired=[paired_stats(rows, 'baseline', arm, 2500) for arm in sources],
        incremental=[paired_stats(rows, a, b, 2500) for a, b in comparisons],
        expanded_true_top3={{arm: int((rows[arm + '_top3'] & missing).sum()) for arm in sources}},
    )

with threadpool_limits(limits=2, user_api='blas'):
    old = replay(historical, historical_trace)
    new = replay(holdout, held_trace)
    panels = dict(c2=old.loc[old.cohort.eq('evaluation')], prvsiyan_895=new, prvsiyan_316=new.loc[new.strict])
    if exposed:
        for name in ('prvsiyan_895', 'prvsiyan_316'):
            part = panels[name]
            panels[name + '_no_runtime_match'] = part.loc[~part.runtime_match]
    report = {{name: {{length: measure(part) for length, part in
                      [('all', rows), ('short', rows.loc[rows.n_words.lt(100)])] if len(part)}}
               for name, rows in panels.items() if len(rows)}}

output = Path('/kaggle/working/frozen_style_transfer')
output.mkdir(exist_ok=True)
pd.concat([old, new], verify_integrity=True).to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(dict(fit_steps=0, weight=.15, runtime_overlap=overlap,
                                                panels=report), indent=2, allow_nan=False) + '\\n')
"""
    for code in (setup, run):
        compile(code, '<frozen style transfer notebook>', 'exec')
    sources = ['zhuating/ntl-styledistance-fp32/1', 'zhuating/ntl-style-mixture-audit/1',
               'zhuating/ntl-style-topic-projection/1', 'zhuating/ntl-sibling-odds-member-trace/1',
               'zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-prvsiyan-member-trace/1',
               'zhuating/ntl-heldout-source-audit/2']
    if gradient:
        sources.append('zhuating/ntl-gradient-selection-audit/1')
    metadata = dict(id='zhuating/ntl-frozen-style-transfer', title='NTL Frozen Style Transfer',
                    code_file='frozen-style-transfer.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=sources)
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gradient', action='store_true')
    build(**vars(parser.parse_args()))
