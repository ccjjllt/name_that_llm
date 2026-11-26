"""Build the private CPU notebook for equal-budget GLM routing."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .kiro_pb115_notebook import NATIVE


KERNEL = 'zhuating/ntl-glm-route-utility'
OUTPUT = Path('runs/glm_route_utility_20261125/kaggle_audit')


def build(output):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in ('glm_route_utility', 'kiro_pb115_oof', 'member_comp_replay', 'glm_prefix_response_replay',
                 'peer_context_report', 'sibling_odds_replay'):
        source = Path(__file__).with_name(name + '.py').read_text()
        compile(source, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({source!r})\n"
    execute = """from threadpoolctl import threadpool_limits
from ntllm.glm_route_utility import run

root = Path('/kaggle/input')

def asset(slug, filename):
    path, = [path for path in root.rglob(filename) if slug in path.parts]
    return path

with threadpool_limits(limits=2, user_api='blas'):
    run(asset('name-that-llm', 'train.csv'),
        asset('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet'),
        asset('ntl-glm-coverage-oof', 'predictions.parquet'),
        asset('ntl-prefixb-full-115', 'paired_token_features.parquet').parent,
        Path('/kaggle/working/glm_route_utility'))
"""
    compile(execute, '<glm-route-utility>', 'exec')
    metadata = dict(id=KERNEL, title='NTL GLM Route Utility', code_file='glm-route-utility.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=['lumibee/ntl-prefixb-full-115/1'],
                    kernel_sources=[NATIVE, 'zhuating/ntl-glm-coverage-oof/1'])
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
