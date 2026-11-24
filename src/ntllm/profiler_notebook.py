"""Package the private PROFILER study and its frozen evaluation inputs for Kaggle."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


def build(output, config, folds, panel):
    module_dir = Path(__file__).parent
    modules = ('profiler', 'profiler_report', 'peer_context', 'peer_context_report',
               'paired_attribution', 'sibling_specialist', 'sibling_odds_replay')
    files = {'ntllm/' + mod + '.py': (module_dir / (mod + '.py')).read_text() for mod in modules}
    files.update({'ntllm/__init__.py': '', 'study.json': config.read_text(), 'folds.json': folds.read_text()})
    panel_files = ('panel.json', 'queries.parquet', 'truth.parquet', 'visible_ids.parquet')
    stage = (
        "from pathlib import Path\nimport base64\n"
        "Path('ntllm').mkdir(exist_ok=True)\n"
        "Path('panel').mkdir(exist_ok=True)\n"
        + ''.join(f'Path({name!r}).write_text({body!r})\n' for name, body in files.items())
        + ''.join(f"Path('panel', {name!r}).write_bytes(base64.b64decode("
                  f"{base64.b64encode((panel / name).read_bytes()).decode()!r}))\n" for name in panel_files)
    )
    run = '''import os
import subprocess
import sys

root = Path('/kaggle/input')
base, = [p.parent for p in root.rglob('4b-base/1/config.json')]
public_trace, = [p.parent for p in root.rglob('stage_scores.npz')
                 if (p.parent / 'public_truth.parquet').is_file()]
independent_trace, = [p.parent for p in root.rglob('stage_scores.npz')
                      if (p.parent / 'frozen_predictions.parquet').is_file()]
official, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
pool, = root.rglob('ntl_open_pool.parquet')
work = str(Path('/kaggle/working/results'))
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
subprocess.run([sys.executable, '-u', '-m', 'ntllm.profiler', '--base', str(base),
                '--pool', str(pool), '--official', str(official), '--trace', str(public_trace),
                '--blind', 'panel', '--folds', 'folds.json', '--config', 'study.json',
                '--out', work], env=env, check=True)
subprocess.run([sys.executable, '-u', '-m', 'ntllm.profiler_report', '--gpu', work,
                '--public-trace', str(public_trace), '--independent-trace', str(independent_trace),
                '--independent-panel', 'panel', '--config', 'study.json',
                '--output', '/kaggle/working/audit'], env=env, check=True)
'''
    metadata = {
        'id': 'zhuating/ntl-profiler-position-ablation',
        'title': 'NTL Profiler Position Ablation',
        'code_file': 'profiler.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'dataset_sources': ['crtproduction/name-that-llm-open-pool/15'],
        'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-gen003-sparse-trace/1'],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
    }
    notebook = {'cells': [cell(stage), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'config', 'folds', 'panel'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.config, args.folds, args.panel)
