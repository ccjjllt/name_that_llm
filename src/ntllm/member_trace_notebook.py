"""Trace each existing member of the unchanged public Sibling Odds stack."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .atlas_notebook import cell, save
from .gen003_sparse import DENNIS
from .glm_prefix_hybrid import cell_index, replace_once, source
from .terra_holdout_notebook import build as build_frozen_trace


ROOT = Path('runs/member_evidence_20261125')
PUBLIC = Path('runs/atlas_0719_20261123/trace/atlas_baseline/public_truth.parquet')
PINNED = Path('runs/sibling_odds_gemini_dash_20261124/source/pinned-metadata.json')
KERNEL = 'zhuating/ntl-sibling-odds-member-trace'


TRACE_MODULE = """
import json
from pathlib import Path

import numpy as np


SCRATCH = Path('/kaggle/temp/ntl-member-trace')


class MemberTrace:
    def __init__(self, ids, labels):
        self.ids = np.asarray(ids, dtype=str)
        self.labels = np.asarray(labels, dtype=str)
        self.scores = {}
        self.coverage = {}
        self.kinds = {}

    def put(self, name, values, rows=None, columns=None, kind='logp'):
        if name in self.scores:
            raise ValueError(f'Duplicate trace member {name}')
        shape = (len(self.ids), len(self.labels))
        active_rows = np.arange(shape[0]) if rows is None else np.arange(shape[0])[rows]
        active_columns = np.arange(shape[1]) if columns is None else np.arange(shape[1])[columns]
        active = np.ix_(active_rows, active_columns)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(active_rows), len(active_columns)):
            raise ValueError(f'{name}: member shape {values.shape} does not match trace coverage')
        finite = np.isfinite(values)
        scores = np.zeros(shape, dtype=np.float32)
        coverage = np.zeros(shape, dtype=bool)
        scores[active] = np.where(finite, values, 0.)
        coverage[active] = finite
        self.scores[name] = scores
        self.coverage[name] = coverage
        self.kinds[name] = kind

    def save(self, branch):
        SCRATCH.mkdir(exist_ok=True)
        np.savez_compressed(SCRATCH / (branch + '_scores.npz'), ids=self.ids, labels=self.labels, **self.scores)
        np.savez_compressed(SCRATCH / (branch + '_coverage.npz'), ids=self.ids, labels=self.labels, **self.coverage)
        (SCRATCH / (branch + '_kinds.json')).write_text(json.dumps(self.kinds))


def consolidate(output, ids, labels):
    merged = MemberTrace(ids, labels)
    for branch in ('infer', 'comp'):
        score_path = SCRATCH / (branch + '_scores.npz')
        if not score_path.exists():
            if branch == 'infer':
                raise RuntimeError('The main stack did not write per-member traces')
            continue
        with np.load(score_path, allow_pickle=False) as scores, np.load(
            SCRATCH / (branch + '_coverage.npz'), allow_pickle=False
        ) as coverage:
            if any(not np.array_equal(bundle['ids'], merged.ids) for bundle in (scores, coverage)):
                raise ValueError(f'{branch}: member row order differs from the frozen trace')
            order = np.argsort(scores['labels'])
            label_order = order[np.searchsorted(scores['labels'], merged.labels, sorter=order)]
            if not np.array_equal(scores['labels'][label_order], merged.labels):
                raise ValueError(f'{branch}: class order differs from the frozen trace')
            if not np.array_equal(scores['labels'], coverage['labels']):
                raise ValueError(f'{branch}: member classes disagree with their coverage')
            kinds = json.loads((SCRATCH / (branch + '_kinds.json')).read_text())
            for name, kind in kinds.items():
                if name in merged.scores:
                    raise ValueError(f'Duplicate member {name} across branches')
                merged.scores[name] = scores[name][:, label_order]
                merged.coverage[name] = coverage[name][:, label_order]
                merged.kinds[name] = kind
    if not merged.scores:
        raise RuntimeError('No per-member evidence captured')
    np.savez_compressed(output / 'member_scores.npz', ids=merged.ids, labels=merged.labels, **merged.scores)
    np.savez_compressed(output / 'member_coverage.npz', ids=merged.ids, labels=merged.labels, **merged.coverage)
    summary = {name: dict(kind=kind, rows=int(merged.coverage[name].any(1).sum()),
                          label_cells=int(merged.coverage[name].sum()))
               for name, kind in merged.kinds.items()}
    (output / 'member_manifest.json').write_text(json.dumps(summary, indent=2) + '\\n')
    print('Observed ensemble members:', list(summary), flush=True)
"""


def instrument_infer(text):
    text = replace_once(
        text, "failed = []\n", """failed = []
from ntl_member_trace import MemberTrace
_member_trace = MemberTrace(test.id, LABELS)
""",
    )
    text = replace_once(
        text, "tot = sum(LP.values())\n", """tot = sum(LP.values())
for _member_name, _member_values in LP.items():
    if _member_name.endswith(('_twin', '_gauss')):
        _member_rows = (_member_values != 0.).any(axis=1)
        _member_trace.put('infer__' + _member_name, _member_values[_member_rows], rows=_member_rows,
                          kind='weighted_logp')
    elif (_member_values != 0.).any():
        _member_trace.put('infer__' + _member_name, _member_values, kind='weighted_logp')
""",
    )
    text = replace_once(
        text, "\ndef blend(terms):\n", """
if COMP_P is not None:
    _member_trace.put('infer__comp_blend', np.log(np.clip(COMP_P, 1e-15, 1.)))


def blend(terms):
""",
    )
    text = replace_once(
        text, "            write(P, 'blend with GSQ 27B')\n", """            write(P, 'blend with GSQ 27B')
            _member_trace.put('infer__gsq', np.log(np.clip(Q27, 1e-15, 1.)))
""",
    )
    text = replace_once(
        text, "    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)\n",
        """    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)
    _available = z['finite'].astype(bool)
    for _member_fold, _member_lp in zip(folds, lp):
        _member_trace.put('native__fold_' + str(_member_fold), _member_lp[_available], rows=_available)
""",
    )
    return text.rstrip() + "\n\n_member_trace.save('infer')\n"


def instrument_comp(text):
    text = replace_once(
        text, "LOGIT = np.zeros((len(TE), K))\n", """LOGIT = np.zeros((len(TE), K))
from ntl_member_trace import MemberTrace
_member_trace = MemberTrace(test.id, LABELS)
""",
    )
    text = replace_once(
        text, "    lp = np.log(np.clip(prob, 1e-9, 1.0))\n",
        """    lp = np.log(np.clip(prob, 1e-9, 1.0))
    _member_trace.put('comp__' + key, lp, rows=rows)
""",
    )
    text = replace_once(
        text, "    base = LOGIT[:, idx].copy()\n", """    base = LOGIT[:, idx].copy()
    _member_trace.put('comp__gate_' + gate_key, gate_lp, columns=idx, kind='conditional_logp')
""",
    )
    return text.rstrip() + "\n\n_member_trace.save('comp')\n"


def build(queries, output, source_dir, pinned):
    build_frozen_trace(queries, output, source_dir, kernel=KERNEL, title='NTL Sibling Odds Member Trace',
                       probe_name='member evidence', pinned=pinned)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    notebook = json.loads((output / metadata['code_file']).read_text())
    for path, instrument in (('infer.py', instrument_infer), ('comp_stack.py', instrument_comp)):
        target = notebook['cells'][cell_index(notebook, f'%%writefile {path}')]
        traced = instrument(source(target))
        compile(traced.split('\n', 1)[1], path, 'exec')
        target['source'] = traced
    compile(TRACE_MODULE, 'ntl_member_trace.py', 'exec')
    notebook['cells'].insert(1, cell(f"Path('ntl_member_trace.py').write_text({TRACE_MODULE!r})\n"))
    notebook['cells'].append(cell("""
from ntl_member_trace import consolidate

consolidate(proof, trace['ids'], trace['labels'])
"""))
    save(output, metadata, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PUBLIC)
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--pinned', type=Path, default=PINNED)
    parser.add_argument('--out', type=Path, default=ROOT / 'kaggle_trace')
    args = parser.parse_args()
    queries = pd.read_parquet(args.queries, columns=['id', 'text'])
    build(queries, args.out, args.source, json.loads(args.pinned.read_text()))


if __name__ == '__main__':
    main()
