"""Build the offline CPU notebook for the frozen LumiBee decision gate comparison."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .lumibee_pb115_notebook import DATA, KERNEL as COVERAGE_KERNEL


KERNEL = 'zhuating/ntl-lumibee-frozen-focal-gates'
COVERAGE = COVERAGE_KERNEL + '/1'


def build(output):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in ('lumibee_focal_gate', 'peer_context_report', 'sibling_odds_replay'):
        body = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({body!r})\n"
    run = """from threadpoolctl import threadpool_limits
from ntllm.lumibee_focal_gate import run

mount = Path('/kaggle/input')
features, = [path.parent for path in mount.rglob('spec.json')
             if (path.parent / 'paired_token_features.parquet').exists()]
reference, = [path.parent for path in mount.rglob('predictions.parquet') if path.parent.name == 'lumibee_pb115_oof']
with threadpool_limits(limits=2, user_api='blas'):
    run(reference, features, Path('/kaggle/working/lumibee_focal_gate'))
"""
    metadata = {
        'id': KERNEL, 'title': 'NTL LumiBee Frozen Focal Gates', 'code_file': 'lumibee-focal-gates.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False, 'competition_sources': [],
        'dataset_sources': [DATA], 'kernel_sources': [COVERAGE],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
