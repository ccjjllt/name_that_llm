"""Build the private two-T4 Terra/GPT-5.5 training ablation."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


def build(config, training, holdout, out):
    root = Path(__file__).parent
    modules = ('terra_pair', 'sibling_specialist', 'paired_attribution', 'sibling_odds_replay',
               'peer_context_report', 'terra_pair_report')
    stage = "from pathlib import Path\nimport base64\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in modules:
        stage += f"Path('ntllm/{module}.py').write_text({(root / (module + '.py')).read_text()!r})\n"
    for name, path in (('study.json', config), ('train_pairs.parquet', training), ('holdout.parquet', holdout)):
        encoded = base64.b64encode(path.read_bytes()).decode()
        stage += f'Path({name!r}).write_bytes(base64.b64decode({encoded!r}))\n'
    run = '''import json
import os
import subprocess
import sys
import torch

if torch.cuda.device_count() != 2:
    raise RuntimeError('The two arms need two T4 GPUs')
root = Path('/kaggle/input')
base, = [p.parent for p in root.rglob('4b-base/1/config.json')]
public, = [p for p in root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
out = Path('/kaggle/working/results')
out.mkdir(exist_ok=True)
args = [sys.executable, '-u', '-m', 'ntllm.terra_pair', '--base', str(base), '--public', str(public),
        '--out', str(out), '--config', 'study.json', '--training', 'train_pairs.parquet', '--holdout', 'holdout.parquet']
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
jobs = []
for gpu, arm in enumerate(('classification', 'pairwise')):
    log = (out / (arm + '.log')).open('w')
    job = subprocess.Popen(args + ['--arm', arm], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
    jobs.append((arm, job, log))
failed = []
for arm, job, log in jobs:
    status = job.wait()
    log.close()
    print((out / (arm + '.log')).read_text()[-10000:], flush=True)
    if status:
        failed.append((arm, status))
if failed:
    raise RuntimeError(f'Training failed: {failed}')
a, b = [json.loads((out / (arm + '.json')).read_text()) for arm, _, _ in jobs]
fields = ('training_file_sha256', 'train_pairs', 'steps', 'public_rows', 'holdout_rows')
if any(a[k] != b[k] for k in fields):
    raise RuntimeError('Arms did not use the same examples, split or step budget')
subprocess.run([sys.executable, '-m', 'ntllm.terra_pair_report', '--scores', str(out),
                '--public-trace', str(public.parent), '--config', 'study.json',
                '--output', '/kaggle/working/audit_public'], env=env, check=True)
'''
    metadata = {
        'id': 'zhuating/ntl-terra-gpt55-pairwise-ablation',
        'title': 'NTL Terra GPT55 Pairwise Ablation',
        'code_file': 'terra-pair.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
    }
    nb = {'cells': [cell(stage), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(out, metadata, nb)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('config', 'training', 'holdout', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    args = p.parse_args()
    build(args.config, args.training, args.holdout, args.out)
