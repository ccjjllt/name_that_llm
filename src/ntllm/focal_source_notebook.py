"""Package a CPU-only audit of independent GLM and Fable candidates."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .atlas_notebook import cell, save
from .focal_source_audit import POOL, SOURCE, STACK_FILES, body_hashes
from .heldout_source_audit import EVAL_FILES, KIMI_EVAL_INDEX


OUTPUT = Path('runs/focal_source_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-focal-source-audit'


def build(output, near_threshold=None):
    prior = pd.read_parquet(KIMI_EVAL_INDEX, columns=['prompt_id'])
    bodies = set().union(*(set(body_hashes(pd.read_parquet(path, columns=['text']).text)) for path in EVAL_FILES))
    keys = json.dumps({'kimi_prompt_ids': sorted(prior.prompt_id.unique()), 'bodies': sorted(bodies)})

    source = Path(__file__).parent
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in ('focal_source_audit', 'heldout_source_audit'):
        setup += f"Path('ntllm/{module}.py').write_text({(source / f'{module}.py').read_text()!r})\n"
    setup += f"Path('prior_keys.json').write_text({keys!r})\n"
    run = f"""from ntllm.focal_source_audit import run

run(Path('/kaggle/input'), Path('prior_keys.json'), Path('/kaggle/working/focal_source'), {near_threshold!r})
"""
    metadata = dict(id=KERNEL, title='NTL Focal Source Audit', code_file='focal-source.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_internet=False,
                    enable_gpu=False, enable_tpu=False, dataset_sources=[SOURCE, POOL, *STACK_FILES],
                    competition_sources=['name-that-llm'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1'], model_sources=[])
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--near-threshold', type=float)
    build(**vars(parser.parse_args()))
