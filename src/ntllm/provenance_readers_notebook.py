"""Package the controlled two-reader experiment for private dual-T4 Kaggle."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


KERNEL = 'zhuating/ntl-provenance-readers'


def build(config, gen003, kimi_flash, output):
    stage = "from pathlib import Path\nimport base64\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    root = Path(__file__).parent
    modules = ('provenance_readers', 'peer_context_report', 'paired_attribution',
               'sibling_specialist', 'sibling_odds_replay')
    for module in modules:
        stage += f"Path('ntllm/{module}.py').write_text({(root / f'{module}.py').read_text()!r})\n"
    stage += f"Path('study.json').write_text({config.read_text()!r})\n"
    for name, panel in (('gen003', gen003), ('kimi_flash', kimi_flash)):
        stage += f"Path('panels/{name}').mkdir(parents=True, exist_ok=True)\n"
        for file in ('queries.parquet', 'truth.parquet'):
            payload = base64.b64encode((panel / file).read_bytes()).decode()
            stage += f"Path('panels/{name}/{file}').write_bytes(base64.b64decode({payload!r}))\n"
    run = """import json
import os
import subprocess
import sys

import torch
from ntllm.provenance_readers import ARMS, compare, prepare

if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('Expected a dual T4 instance')
root = Path('/kaggle/input')
config = Path('study.json')
official, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
public, = [p.parent for p in root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
pool, = root.rglob('ntl_open_pool.parquet')
base, = [p.parent for p in root.rglob('4b-base/1/config.json')]
work = Path('/kaggle/temp/provenance-readers')
output = Path('/kaggle/working/results')
prepare(config, pool, official, public, Path('panels'), work, output)

args = [sys.executable, '-u', '-m', 'ntllm.provenance_readers', '--base', str(base),
        '--config', str(config), '--work', str(work), '--output', str(output)]
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
jobs = []
for gpu, arm in enumerate(ARMS):
    log = (output / f'{arm}.log').open('w')
    job = subprocess.Popen(args + ['--arm', arm], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
    jobs.append((arm, job, log))

failed = []
for arm, job, log in jobs:
    status = job.wait()
    log.close()
    print((output / f'{arm}.log').read_text()[-8000:], flush=True)
    if status:
        failed.append((arm, status))
if failed:
    raise RuntimeError(f'Reader jobs failed: {failed}')

compare(public, work, output, json.loads(config.read_text())['diagnostic_blend'])
"""
    metadata = {'id': KERNEL, 'title': 'NTL Provenance Readers', 'code_file': 'provenance-readers.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
                'enable_internet': False, 'dataset_sources': ['crtproduction/name-that-llm-open-pool/18'],
                'competition_sources': ['name-that-llm'],
                'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
                'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1']}
    notebook = {'cells': [cell(stage), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'gen003', 'kimi-flash', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.gen003, args.kimi_flash, args.output)
