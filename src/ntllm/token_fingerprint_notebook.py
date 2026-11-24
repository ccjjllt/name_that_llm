"""Bundle the token-fingerprint study and its fixed public-cohort comparison for Kaggle."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save


def build(output, config, folds):
    slug = 'ntl-q4-token-forecast-fingerprints'
    metadata = {'id': 'zhuating/' + slug, 'title': 'NTL Q4 Token Forecast Fingerprints',
                'code_file': 'fingerprints.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'competition_sources': ['name-that-llm'],
                'dataset_sources': ['crtproduction/name-that-llm-open-pool/15'],
                'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
                'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1']}
    root = Path(__file__).parent
    modules = ('paired_attribution', 'sibling_odds_replay', 'token_fingerprint')
    files = {'ntllm/' + mod + '.py': (root / (mod + '.py')).read_text() for mod in modules}
    files.update({'ntllm/__init__.py': '', 'config.json': config.read_text(), 'prompt_folds.json': folds.read_text()})
    stage = cell("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n" +
                 ''.join(f'Path({name!r}).write_text({source!r})\n' for name, source in files.items()))
    train = cell("""import os
import subprocess
import sys
root = Path('/kaggle/input')
model, = [p.parent for p in root.rglob('4b-base/1/config.json')]
official, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
trace, = [p.parent for p in root.rglob('stage_scores.npz') if (p.parent / 'public_truth.parquet').is_file()]
pool, = root.rglob('ntl_open_pool.parquet')
subprocess.run([sys.executable, '-u', '-m', 'ntllm.token_fingerprint', '--base', str(model),
                '--official', str(official), '--trace', str(trace), '--pool', str(pool),
                '--config', 'config.json', '--folds', 'prompt_folds.json',
                '--out', '/kaggle/working/fingerprints'],
               env=dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2'),
               check=True)
""")
    notebook = {'cells': [stage, train], 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'config', 'folds'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.config, args.folds)
