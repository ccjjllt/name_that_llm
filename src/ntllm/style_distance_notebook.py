"""Package the offline Kaggle StyleDistance experiment."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


KERNEL = 'zhuating/ntl-styledistance-study'


def build(output, weights, cache_dataset=None, style_float32=False):
    source = Path(__file__).parent
    modules = ('style_distance', 'provenance_readers', 'peer_context_report',
               'paired_attribution', 'sibling_specialist', 'sibling_odds_replay')
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in modules:
        setup += f"Path('ntllm/{name}.py').write_text({(source / f'{name}.py').read_text()!r})\n"
    run = """import os
import subprocess
import sys

mount = Path('/kaggle/input')
official, = [p for p in mount.rglob('train.csv') if p.parent.name == 'name-that-llm']
public, = [p.parent for p in mount.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').exists()]
pool, = mount.rglob('ntl_open_pool.parquet')
audit, = [p.parent for p in mount.rglob('candidate_truth.parquet') if (p.parent / 'candidate_queries.parquet').exists()]
encoders, = [p for p in mount.rglob('encoders') if (p / 'style/config.json').exists()
             and (p / 'roberta/config.json').exists()]
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='4',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
command = [sys.executable, '-u', '-m', 'ntllm.style_distance', '--official', str(official), '--public',
           str(public), '--audit', str(audit), '--pool', str(pool), '--encoders', str(encoders),
           '--output', '/kaggle/working/evaluation']
"""
    sources = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/1', weights]
    datasets = ['crtproduction/name-that-llm-open-pool/18']
    title, kernel = 'NTL StyleDistance Study', KERNEL
    if cache_dataset is not None:
        slug = cache_dataset.split('/')[1]
        datasets.append(cache_dataset)
        run += f"""cache, = [p.parent for p in mount.rglob('roberta_logp.parquet') if {slug!r} in p.parts]
command += ['--roberta-cache', str(cache)]
"""
        title, kernel = 'NTL StyleDistance Resume', 'zhuating/ntl-styledistance-resume'
    if style_float32:
        run += "command += ['--style-float32']\n"
        title, kernel = 'NTL StyleDistance FP32', 'zhuating/ntl-styledistance-fp32'
    run += 'subprocess.run(command, env=env, check=True)\n'
    for code in (setup, run):
        compile(code, '<style-distance>', 'exec')
    metadata = {
        'id': kernel, 'title': title, 'code_file': 'style-distance.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'competition_sources': ['name-that-llm'],
        'dataset_sources': datasets,
        'kernel_sources': sources,
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--cache-dataset')
    parser.add_argument('--style-float32', action='store_true')
    args = parser.parse_args()
    build(args.output, args.weights, args.cache_dataset, args.style_float32)
