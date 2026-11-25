"""Package the CPU-only StyleDistance mixture diagnostic."""

import argparse
from pathlib import Path
import re

from .atlas_notebook import cell, save


OUTPUT = Path('runs/style_mixture_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-style-mixture-audit'


def pinned_slug(reference):
    match = re.fullmatch(r'[\w-]+/([\w-]+)/[1-9][0-9]*', reference)
    if match is None:
        raise ValueError(f'Expected a pinned Kaggle code source owner/slug/version, got {reference}')
    return match[1]


def build(output, style_ref, held_trace_ref=None):
    style_slug = pinned_slug(style_ref)
    held_trace_slug = pinned_slug(held_trace_ref) if held_trace_ref is not None else None
    modules = ('style_mixture', 'style_distance', 'provenance_readers', 'peer_context_report',
               'paired_attribution', 'sibling_specialist', 'sibling_odds_replay',
               'member_evidence', 'member_evidence_transfer')
    source = Path(__file__).parent
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in modules:
        setup += f"Path('ntllm/{name}.py').write_text({(source / f'{name}.py').read_text()!r})\n"

    run = f"""import os
import subprocess
import sys

mount = Path('/kaggle/input')

def asset(slug, name):
    path, = [path for path in mount.rglob(name) if slug in path.parts]
    return path

official = asset('name-that-llm', 'train.csv')
public = asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent
audit = asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent
pool = asset('name-that-llm-open-pool', 'ntl_open_pool.parquet')
style = asset({style_slug!r}, 'style_embedding.npz').parent
env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
command = [sys.executable, '-u', '-m', 'ntllm.style_mixture', '--official', str(official), '--public', str(public),
           '--audit', str(audit), '--pool', str(pool), '--style', str(style),
           '--output', '/kaggle/working/style_mixture']
"""
    sources = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/1', style_ref]
    if held_trace_ref is not None:
        run += f"held = asset({held_trace_slug!r}, 'probe_manifest.json').parent\n"
        run += "command += ['--held-trace', str(held)]\n"
        sources.append(held_trace_ref)
    run += 'subprocess.run(command, env=env, check=True)\n'
    metadata = {
        'id': KERNEL, 'title': 'NTL Style Mixture Audit', 'code_file': 'style-mixture.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'competition_sources': ['name-that-llm'],
        'dataset_sources': ['crtproduction/name-that-llm-open-pool/18'], 'kernel_sources': sources,
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--style-ref', required=True)
    cli.add_argument('--held-trace-ref')
    cli.add_argument('--output', type=Path, default=OUTPUT)
    build(**vars(cli.parse_args()))
