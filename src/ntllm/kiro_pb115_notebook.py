"""Build the Kaggle CPU ablation for the published Kiro PB115 feature asset."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


KERNEL = 'zhuating/ntl-kiro-pb115-oof'
NATIVE = 'zhuating/ntl-glm-likelihood-dynamics/2'


def build(output, native):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in ('kiro_pb115_oof', 'member_comp_replay', 'glm_prefix_response_replay',
                 'peer_context_report', 'sibling_odds_replay'):
        body = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({body!r})\n"
    run = """from threadpoolctl import threadpool_limits
from ntllm.kiro_pb115_oof import run

mount = Path('/kaggle/input')
official, = [path for path in mount.rglob('train.csv') if path.parent.name == 'name-that-llm']
native, = mount.rglob('glm_dynamics_oof.parquet')
features, = [path.parent for path in mount.rglob('spec.json')
             if (path.parent / 'paired_token_features.parquet').exists()]
with threadpool_limits(limits=2, user_api='blas'):
    run(official, native, features, Path('/kaggle/working/kiro_pb115_oof'))
"""
    for code in (setup, run):
        compile(code, '<kiro-pb115-oof>', 'exec')
    metadata = {
        'id': KERNEL, 'title': 'NTL Kiro PB115 OOF', 'code_file': 'kiro-pb115-oof.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['name-that-llm'], 'dataset_sources': ['lumibee/ntl-prefixb-full-115/1'],
        'kernel_sources': [native],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', default=NATIVE)
    args = parser.parse_args()
    build(args.output, args.native)
