"""Package the CPU-only Claude group sample and holdout audit."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .pair_supply_notebook import READER


OUTPUT = Path('runs/claude_group_20261125/kaggle_input')


def build(output):
    setup = ''.join(json.loads(READER.read_text())['cells'][0]['source'])
    source = Path(__file__).with_name('claude_group_sample.py').read_text()
    compile(source, '<claude group sample>', 'exec')
    setup += f"\nPath('ntllm/claude_group_sample.py').write_text({source!r})\n"
    run = """from ntllm.claude_group_sample import build_sample

mount = Path('/kaggle/input')

def asset(slug, name):
    found, = [path for path in mount.rglob(name) if slug in path.parts]
    return found

build_sample(asset('name-that-llm', 'train.csv'),
             asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
             asset('name-that-llm-open-pool', 'ntl_open_pool.parquet'),
             Path('panels'), asset('ntl-provenance-readers', 'provenance_balanced_sample.parquet'),
             asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
             asset('ntl-matched-request-supply', 'groups.parquet'),
             Path('/kaggle/working/claude_group_input'))
"""
    for code in (setup, run):
        compile(code, '<Claude group input notebook>', 'exec')
    metadata = dict(id='zhuating/ntl-claude-group-input', title='NTL Claude Group Input',
                    code_file='claude-group-input.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/18'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1',
                                    'zhuating/ntl-provenance-readers/1',
                                    'zhuating/ntl-heldout-source-audit/2',
                                    'zhuating/ntl-matched-request-supply/2'])
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
