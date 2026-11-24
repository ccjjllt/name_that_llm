"""Package the private targeted LoRA and its matched GPU control for Kaggle."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


def build(output, config):
    metadata = {'id': 'zhuating/ntl-sibling-error-specialist', 'title': 'NTL Sibling Error Specialist',
                'code_file': 'specialist.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'competition_sources': ['name-that-llm'],
                'dataset_sources': ['crtproduction/name-that-llm-open-pool/15'],
                'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1'],
                'model_sources': ['qwen-lm/qwen-3/Transformers/1.7b-base/1']}
    modules = ('paired_attribution', 'sibling_odds_replay', 'sibling_specialist')
    root = Path(__file__).parent
    files = {'ntllm/' + mod + '.py': (root / (mod + '.py')).read_text() for mod in modules}
    files.update({'ntllm/__init__.py': '', 'config.json': config.read_text()})
    write = cell("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n" +
                 ''.join(f'Path({name!r}).write_text({content!r})\n' for name, content in files.items()))
    train = cell('''import os
import subprocess
import sys
import torch

assert torch.cuda.device_count() == 2, 'The paired run needs its two requested T4 GPUs'
root = Path('/kaggle/input')
model, = [p.parent for p in root.rglob('1.7b-base/1/config.json')]
train, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
trace, = [p.parent for p in root.rglob('stage_scores.npz') if (p.parent / 'public_truth.parquet').is_file()]
pool, = root.rglob('ntl_open_pool.parquet')
args = [sys.executable, '-u', '-m', 'ntllm.sibling_specialist', '--base', str(model),
        '--official', str(train), '--trace', str(trace), '--pool', str(pool), '--config', 'config.json']
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
arms = [('focused', ['--focused']), ('cross_entropy', [])]
jobs = []
for gpu, (name, flags) in enumerate(arms):
    out = '/kaggle/working/' + name
    log = open(out + '.log', 'w')
    job = subprocess.Popen(args + ['--out', out] + flags, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
    jobs.append((name, job, log))
failures = []
for name, job, log in jobs:
    code = job.wait()
    log.close()
    print(Path('/kaggle/working', name + '.log').read_text()[-18000:], flush=True)
    if code:
        failures.append((name, code))
if failures:
    raise RuntimeError(f'Specialist study failed: {failures}')
''')
    notebook = {'cells': [write, train], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.config)
