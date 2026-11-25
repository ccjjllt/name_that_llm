"""Measure a checkpoint-repaired Comp branch on two archived unlabeled panels."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .glm_prefix_hybrid import cell_index, replace_once, source
from .member_trace_notebook import DENNIS, PINNED, TRACE_MODULE, instrument_comp


KERNEL = 'zhuating/ntl-sibling-odds-comp-diagnostic'
OUTPUT = Path('runs/member_evidence_20261125/kaggle_comp')
AUDIT = 'zhuating/ntl-heldout-source-audit/2'
PUBLIC = 'zhuating/ntl-0719-public-decision-trace/1'
SOURCE = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'


def pin_qwen8(text):
    return replace_once(text, 'def find_checkpoint(model_type, hidden=None, dir_name=None):\n', '''
def find_checkpoint(model_type, hidden=None, dir_name=None):
    if (model_type, hidden, dir_name) == ('qwen3', 4096, None):
        candidates = [
            cfg.parent for cfg in pathlib.Path('/kaggle/input').rglob('8b-base/1/config.json')
            if 'qwen-3' in cfg.parts and
            (list(cfg.parent.glob('model*.safetensors')) or list(cfg.parent.glob('pytorch_model*.bin')))
        ]
        checkpoint, = candidates
        config = json.loads((checkpoint / 'config.json').read_text())
        if (config.get('model_type'), config.get('hidden_size')) != ('qwen3', 4096):
            raise ValueError('Unexpected model in the pinned Qwen3-8B base')
        print('COMP CHECKPOINT qwen3:4096:', checkpoint, flush=True)
        return str(checkpoint)
''')


def comp_program(text):
    program = pin_qwen8(instrument_comp(text).split('\n', 1)[1])
    program = replace_once(program, '            return os.path.dirname(cfg)\n',
                           "            print('COMP CHECKPOINT', model_type, hidden, dir_name, os.path.dirname(cfg), flush=True)\n"
                           '            return os.path.dirname(cfg)\n')
    program = replace_once(program, "_member_trace = MemberTrace(test.id, LABELS)\n", '''
_member_trace = MemberTrace(test.id, LABELS)

def _save_comp_progress(stage):
    # Save outside the final Comp output path so an interrupted run stays identifiable.
    _member_trace.save('comp')
    folder = pathlib.Path('/kaggle/temp/ntl-member-trace')
    pending = folder / 'partial-new.npz'
    np.savez_compressed(pending, ids=np.asarray(test.id, dtype=str), labels=np.asarray(LABELS, dtype=str),
                        logits=LOGIT.astype(np.float32), stage=stage, elapsed=time.time() - T0)
    pending.replace(folder / 'partial.npz')
''')
    program = replace_once(
        program, '        LOGIT[rows] += w * lp\n',
        "        LOGIT[rows] += w * lp\n    _save_comp_progress('member:' + key)\n",
    )
    program = replace_once(
        program, '    LOGIT[:, idx] = family_mass[:, None] + mixed_lp\n',
        "    LOGIT[:, idx] = family_mass[:, None] + mixed_lp\n    _save_comp_progress('gate:' + gate_key)\n",
    )
    compile(program, 'comp_stack.py', 'exec')
    return program


def build(output, source_dir, pinned, timeout):
    baseline = json.loads((source_dir / 'kernel-metadata.json').read_text())
    original = json.loads((source_dir / baseline['code_file']).read_text())
    stack = comp_program(source(original['cells'][cell_index(original, '%%writefile comp_stack.py')]))
    pipe = source(original['cells'][cell_index(original, '%%writefile pipe.py')]).split('\n', 1)[1]
    setup = f"""from pathlib import Path

Path('/kaggle/working/comp_stack.py').write_text({stack!r})
Path('/kaggle/working/pipe.py').write_text({pipe!r})
Path('/kaggle/working/ntl_member_trace.py').write_text({TRACE_MODULE!r})
"""
    run = f"""source_slug = {SOURCE!r}
timeout = {timeout}
""" + '''import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
    raise RuntimeError('Archived diagnostic only')

root = Path('/kaggle/input')
public, = root.rglob('public_truth.parquet')
candidates, = root.rglob('candidate_queries.parquet')
source_index = pd.read_parquet(candidates.with_name('candidate_truth.parquet'), columns=['id', 'source_dataset'])
source_ids = pd.Index(source_index.loc[source_index.source_dataset.eq(source_slug), 'id'])
source_texts = pd.read_parquet(candidates, columns=['id', 'text']).set_index('id').loc[source_ids]
historical_texts = pd.read_parquet(public, columns=['id', 'text']).set_index('id')
assert len(historical_texts) == 798 and len(source_texts) == 895
combined = pd.concat([historical_texts, source_texts], verify_integrity=True)
assert combined.text.notna().all() and combined.index.name == 'id'

data = Path('/kaggle/temp/comp-diagnostic-data')
data.mkdir(exist_ok=True)
reference = next(p for p in (root / 'competitions/name-that-llm', root / 'name-that-llm')
                 if (p / 'train.csv').is_file())
(data / 'train.csv').symlink_to(reference / 'train.csv')
combined.to_csv(data / 'test.csv')
del source_index, source_ids, source_texts, historical_texts

wheels = glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True)
if wheels:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + wheels,
                   capture_output=True, text=True)
scratch = Path('/kaggle/temp/ntl-member-trace')
working = Path('/kaggle/working')
output = working / 'comp_diagnostic'
output.mkdir(exist_ok=True)
log = output / 'comp.log'
start = time.monotonic()
with log.open('w') as handle:
    try:
        job = subprocess.run([sys.executable, '-u', str(working / 'comp_stack.py')], cwd=working,
                             env=dict(os.environ, NTL_REHEARSAL_DATA=str(data), NTL_DROP_Q35='1',
                                      NTL_PIPE_WORKERS='1'),
                             stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        returncode = job.returncode
    except subprocess.TimeoutExpired:
        returncode = None
elapsed = time.monotonic() - start

for name in ('comp_scores.npz', 'comp_coverage.npz', 'comp_kinds.json'):
    if (scratch / name).is_file():
        shutil.copyfile(scratch / name, output / name)
partial = scratch / 'partial.npz'
if partial.is_file():
    shutil.copyfile(partial, output / 'partial.npz')
score = working / 'comp_logit.npy'
meta = working / 'comp_meta.json'
complete = returncode == 0 and score.is_file() and meta.is_file()
if complete:
    metadata = json.loads(meta.read_text())
    actual_ids = np.asarray(metadata['ids'], dtype=str)
    labels = np.asarray(metadata['labels'], dtype=str)
    raw = np.load(score, allow_pickle=False)
    if (raw.shape != (len(combined), 18) or not np.isfinite(raw).all() or len(set(labels)) != 18
            or actual_ids.tolist() != combined.index.astype(str).tolist()):
        raise ValueError('Comp logits do not match the unlabeled query IDs and classes')
    logp = raw.astype(np.float64) - logsumexp(raw.astype(np.float64), axis=1, keepdims=True)
    np.savez_compressed(output / 'scores.npz', ids=actual_ids, labels=labels,
                        raw_logits=raw, logp=logp.astype(np.float32))
    shutil.copyfile(meta, output / 'comp_meta.json')
    score.unlink()
    meta.unlink()
summary = dict(complete=complete, returncode=returncode, seconds=elapsed, timeout_seconds=timeout,
               rows=len(combined), historical_rows=798, prvsiyan_rows=895,
               ordered_ids_sha256=hashlib.sha256(('\\n'.join(combined.index) + '\\n').encode()).hexdigest(),
               input_body_sha256=[hashlib.sha256(body.encode()).hexdigest() for body in combined.text],
               partial_written=(output / 'partial.npz').is_file())
(output / 'report.json').write_text(json.dumps(summary, indent=2) + '\\n')
print(log.read_text()[-7500:], flush=True)
print(json.dumps(summary, indent=2), flush=True)
'''
    metadata = dict(pinned)
    metadata.pop('id_no', None)
    metadata.update(id=KERNEL, title='NTL Sibling Odds Comp Diagnostic', code_file='comp-diagnostic.ipynb',
                    is_private=True, enable_gpu=True, enable_internet=False, docker_image_pinning_type='original',
                    kernel_sources=[*pinned['kernel_sources'], PUBLIC, AUDIT])
    notebook = {'cells': [cell(setup), cell(run)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--pinned', type=Path, default=PINNED)
    parser.add_argument('--timeout', type=int, default=10800)
    args = parser.parse_args()
    build(args.output, args.source, json.loads(args.pinned.read_text()), args.timeout)
