"""Package the ID-only pair-preserving sample audit on Kaggle CPU."""

import argparse
import hashlib
import json
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/pair_preserving_20261125/kaggle_audit')
ENRICHED_OUTPUT = OUTPUT.with_name('kaggle_audit_v2')
READERS = Path('runs/provenance_readers_20261125')
SUPPLY = Path('runs/pair_supply_20261125/audit_v2_remote/pair_supply')


def build(output, enriched=False):
    prepared = json.loads((READERS / 'kaggle/provenance-readers.ipynb').read_text())['cells'][0]
    code = Path(__file__).with_name('pair_preserving_sample.py').read_text()
    compile(code, '<pair-preserving sample module>', 'exec')
    setup = cell(f"Path('ntllm/pair_preserving_sample.py').write_text({code!r})\n")
    files = {
        'provenance_balanced_sample.parquet': READERS / 'training_gpu/results/provenance_balanced_sample.parquet',
        'vertex_pairs.parquet': SUPPLY / 'vertex_pairs.parquet',
        'report.json': SUPPLY / 'report.json',
    }
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    run = f'expected = {hashes!r}\nenriched = {enriched!r}\n' + '''import hashlib

from ntllm.pair_preserving_sample import run

mount = Path('/kaggle/input')

def asset(slug, name):
    path, = [p for p in mount.rglob(name) if slug in p.parts]
    return path

original = asset('ntl-provenance-readers', 'provenance_balanced_sample.parquet')
vertex = asset('ntl-pair-supply-audit', 'vertex_pairs.parquet')
for path in (original, vertex, vertex.with_name('report.json')):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
        raise ValueError(f'Pinned input artifact differs: {path}')

run(asset('name-that-llm', 'train.csv'),
    asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
    asset('name-that-llm-open-pool', 'ntl_open_pool.parquet'),
    Path('panels'), Path('study.json'), original, vertex.parent,
    Path('/kaggle/working/pair_preserving'), enriched=enriched)
'''
    metadata = dict(id='zhuating/ntl-pair-preserving-sample', title='NTL Pair Preserving Sample',
                    code_file='pair-preserving-sample.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/18'], kernel_sources=[
                        'zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-provenance-readers/1',
                        'zhuating/ntl-pair-supply-audit/2'])
    notebook = dict(cells=[prepared, setup, cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--enriched', action='store_true')
    args = parser.parse_args()
    build(args.output or (ENRICHED_OUTPUT if args.enriched else OUTPUT), enriched=args.enriched)
