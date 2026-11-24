"""Stage a private two-T4 context study with pinned panels and model weights."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .atlas_notebook import cell, save


def build(out, config, folds, blind_ids):
    module_dir = Path(__file__).parent
    modules = ('peer_context', 'sibling_odds_replay', 'sibling_specialist', 'paired_attribution')
    files = {'ntllm/' + mod + '.py': (module_dir / (mod + '.py')).read_text() for mod in modules}
    queries = pd.read_parquet(blind_ids).set_index('id')
    hashes = {i: hashlib.sha256(t.encode()).hexdigest() for i, t in queries.text.items()}
    files.update({'ntllm/__init__.py': '', 'study.json': config.read_text(), 'folds.json': folds.read_text(),
                  'blind_ids.json': json.dumps(hashes)})
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / 'blind_ids.json').write_text(json.dumps(hashes))
    stage = cell("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n" +
                 ''.join(f'Path({k!r}).write_text({v!r})\n' for k, v in files.items()))
    run = cell('''import os
import subprocess
import sys

root = Path('/kaggle/input')
base, = [p.parent for p in root.rglob('4b-base/1/config.json')]
trace, = [p.parent for p in root.rglob('stage_scores.npz') if (p.parent/'public_truth.parquet').exists()]
official, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
pool, = root.rglob('ntl_open_pool.parquet')
blind, = root.rglob('ntl_gen003_claude_panel.parquet')
args = [sys.executable, '-u', '-m', 'ntllm.peer_context', '--base', str(base), '--pool', str(pool),
        '--official', str(official), '--trace', str(trace), '--blind', str(blind),
        '--blind-ids', 'blind_ids.json', '--folds', 'folds.json', '--cfg', 'study.json',
        '--out', '/kaggle/working/results']
subprocess.run(args, check=True, env=dict(os.environ, HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2'))
''')
    meta = {'id': 'zhuating/ntl-peer-context-attribution', 'title': 'NTL Peer Context Attribution',
            'code_file': 'study.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_internet': False, 'competition_sources': ['name-that-llm'],
            'dataset_sources': ['crtproduction/name-that-llm-open-pool/15', 'tadeuhayashida/ntl-gen003-claude-panel'],
            'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
            'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1']}
    notebook = {'cells': [stage, run], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(out, meta, notebook)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('out', 'config', 'folds', 'blind-ids'):
        p.add_argument('--' + arg, type=Path, required=True)
    args = p.parse_args()
    build(args.out, args.config, args.folds, args.blind_ids)
