"""Package the binary specialist with two concurrent T4 seeds."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


KERNEL = 'zhuating/ntl-kimi-flash-specialist'
TRACE = 'zhuating/ntl-0719-public-decision-trace/1'
HOLDOUT_TRACE = 'zhuating/ntl-kimi-flash-holdout-trace/1'
AUDIT = 'zhuating/ntl-kimi-flash-audit'


def stage_inputs(inputs):
    root = Path(__file__).parent
    modules = ('kimi_flash', 'kimi_flash_report', 'effort_pair_report', 'peer_context_report',
               'sibling_odds_replay', 'sibling_specialist', 'paired_attribution')
    stage = "from pathlib import Path\nimport base64\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in modules:
        stage += f"Path('ntllm/{module}.py').write_text({(root / (module + '.py')).read_text()!r})\n"
    for name, path in inputs:
        stage += f'Path({name!r}).write_bytes(base64.b64decode({base64.b64encode(path.read_bytes()).decode()!r}))\n'
    return stage


def notebook_cells(stage, run):
    return {'cells': [cell(stage), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}


def build(config, training, queries, output):
    stage = stage_inputs((('study.json', config), ('training.parquet', training), ('holdout.parquet', queries)))
    run = '''import json
import os
import subprocess
import sys

import torch

seeds = json.loads(Path('study.json').read_text())['seeds']
if len(set(seeds)) != 2 or torch.cuda.device_count() != 2:
    raise RuntimeError('Two distinct seeds and two assigned GPUs are required')
devices = [torch.cuda.get_device_name(i) for i in range(2)]
if any('T4' not in value for value in devices):
    raise RuntimeError(f'Expected dual T4, got {devices}')
print('GPUs:', devices, flush=True)
input_root = Path('/kaggle/input')
base, = [p.parent for p in input_root.rglob('4b-base/1/config.json')]
public, = [p for p in input_root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
out = Path('/kaggle/working/results')
out.mkdir(exist_ok=True)
args = [sys.executable, '-u', '-m', 'ntllm.kimi_flash', '--base', str(base), '--config', 'study.json',
        '--training', 'training.parquet', '--public', str(public), '--holdout', 'holdout.parquet', '--out', str(out)]
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
jobs = []
for gpu, seed in enumerate(seeds):
    log = (out / f'seed_{seed}.log').open('w')
    process = subprocess.Popen(args + ['--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT,
                               env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
    jobs.append((seed, process, log))
failed = []
for seed, process, log in jobs:
    return_code = process.wait()
    log.close()
    print((out / f'seed_{seed}.log').read_text()[-10000:], flush=True)
    if return_code:
        failed.append((seed, return_code))
if failed:
    raise RuntimeError(f'Training failed: {failed}')

subprocess.run([sys.executable, '-m', 'ntllm.kimi_flash_report', '--scores', str(out), '--config', 'study.json',
                '--public-trace', str(public.parent), '--output', '/kaggle/working/audit_public'],
               env=env, check=True)
'''
    metadata = {
        'id': KERNEL,
        'title': 'NTL Kimi Flash Specialist',
        'code_file': 'kimi-flash.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'kernel_sources': [TRACE],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
    }
    save(output, metadata, notebook_cells(stage, run))


def build_audit(config, panel, output):
    stage = stage_inputs((('study.json', config),
                         ('queries.parquet', panel / 'queries.parquet'),
                         ('truth.parquet', panel / 'truth.parquet')))
    run = '''import json
from ntllm.kimi_flash_report import run

input_root = Path('/kaggle/input')
public, = [p.parent for p in input_root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
holdout, = [p.parent for p in input_root.rglob('probe_manifest.json') if (p.parent / 'stage_scores.npz').is_file()]
seeds = json.loads(Path('study.json').read_text())['seeds']
scores, = [p.parent for p in input_root.rglob(f'seed_{seeds[0]}.parquet')
           if (p.parent / f'seed_{seeds[1]}.parquet').is_file()]
print('Frozen input directories:', scores, public, holdout, flush=True)
report = run(scores, Path('study.json'), public, Path('/kaggle/working/audit'), Path('.'), holdout)
for row in report['comparisons']:
    if row['after'] == f"expert_{report['primary_weight']:g}":
        print(row['cohort'], row['subset'], 'n=', row['rows'], 'changed=', row['changed'],
              'fixed=', row['fixed'], 'harmed=', row['harmed'], 'BA gain (pp)=', row['ba_delta_pp'])
'''
    metadata = {
        'id': AUDIT,
        'title': 'NTL Kimi Flash Audit',
        'code_file': 'kimi-flash-audit.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': False,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'kernel_sources': [TRACE, HOLDOUT_TRACE, KERNEL + '/1'],
    }
    save(output, metadata, notebook_cells(stage, run))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'training', 'queries', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.training, args.queries, args.output)
