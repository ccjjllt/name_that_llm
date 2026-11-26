"""Package the official PB115 coverage ablation as a private offline CPU notebook."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .kiro_pb115_notebook import KERNEL as KIRO_KERNEL, NATIVE


KERNEL = 'zhuating/ntl-lumibee-pb115-coverage'
DATA = 'lumibee/ntl-prefixb-full-115/1'
KIRO = KIRO_KERNEL + '/1'


def build(output):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in ('lumibee_pb115_oof', 'kiro_pb115_oof', 'member_comp_replay', 'glm_prefix_response_replay',
                 'peer_context_report', 'sibling_odds_replay'):
        body = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({body!r})\n"
    run = """from threadpoolctl import threadpool_limits
from ntllm.lumibee_pb115_oof import run

mount = Path('/kaggle/input')
official, = [path for path in mount.rglob('train.csv') if path.parent.name == 'name-that-llm']
native, = mount.rglob('glm_dynamics_oof.parquet')
features, = [path.parent for path in mount.rglob('spec.json')
             if (path.parent / 'paired_token_features.parquet').exists()]
reference, = [path for path in mount.rglob('predictions.parquet') if path.parent.name == 'kiro_pb115_oof']
with threadpool_limits(limits=2, user_api='blas'):
    run(official, native, features, reference, Path('/kaggle/working/lumibee_pb115_oof'))
"""
    for code in (setup, run):
        compile(code, '<lumibee-pb115-oof>', 'exec')
    metadata = {
        'id': KERNEL, 'title': 'NTL LumiBee PB115 Coverage', 'code_file': 'lumibee-pb115-coverage.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False, 'competition_sources': ['name-that-llm'],
        'dataset_sources': [DATA], 'kernel_sources': [NATIVE, KIRO],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
