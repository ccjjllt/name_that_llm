"""Stage the private two-T4 Kimi/GLM matched-effort ablation."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


def build(config, training, holdout_queries, output):
    root = Path(__file__).parent
    modules = ('effort_pair', 'effort_pair_report', 'peer_context_report', 'sibling_odds_replay',
               'sibling_specialist', 'paired_attribution')
    stage = "from pathlib import Path\nimport base64\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in modules:
        stage += f"Path('ntllm/{module}.py').write_text({(root / (module + '.py')).read_text()!r})\n"
    for name, path in (('study.json', config), ('train_pairs.parquet', training), ('holdout.parquet', holdout_queries)):
        stage += f'Path({name!r}).write_bytes(base64.b64decode({base64.b64encode(path.read_bytes()).decode()!r}))\n'
    run = '''import json
import os
import subprocess
import sys
import torch

if torch.cuda.device_count() != 2:
    raise RuntimeError('The two arms need two T4 GPUs')
input_root = Path('/kaggle/input')
base, = [p.parent for p in input_root.rglob('4b-base/1/config.json')]
public, = [p for p in input_root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
out = Path('/kaggle/working/results')
out.mkdir(exist_ok=True)
args = [sys.executable, '-u', '-m', 'ntllm.effort_pair', '--base', str(base), '--public', str(public),
        '--out', str(out), '--config', 'study.json', '--training', 'train_pairs.parquet', '--holdout', 'holdout.parquet']
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
jobs = []
for gpu, arm in enumerate(('classification', 'consistent')):
    log = (out / (arm + '.log')).open('w')
    worker = subprocess.Popen(args + ['--arm', arm], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
    jobs.append((arm, worker, log))
failures = []
for arm, worker, log in jobs:
    status = worker.wait()
    log.close()
    print((out / (arm + '.log')).read_text()[-10000:], flush=True)
    if status:
        failures.append((arm, status))
if failures:
    raise RuntimeError(f'Training failed: {failures}')
classification, consistent = [json.loads((out / (arm + '.json')).read_text()) for arm, _, _ in jobs]
fields = ('training_file_sha256', 'train_pairs', 'steps', 'public_rows', 'holdout_rows')
if any(classification[key] != consistent[key] for key in fields):
    raise RuntimeError('Arms did not use the same input and step budget')
subprocess.run([sys.executable, '-m', 'ntllm.effort_pair_report', '--scores', str(out),
                '--public-trace', str(public.parent), '--config', 'study.json',
                '--output', '/kaggle/working/audit_public'], env=env, check=True)
'''
    metadata = {
        'id': 'zhuating/ntl-kimi-glm-effort-ablation',
        'title': 'NTL Kimi GLM Effort Ablation',
        'code_file': 'effort-pair.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
    }
    notebook = {'cells': [cell(stage), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('config', 'training', 'holdout-queries', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.training, args.holdout_queries, args.output)
