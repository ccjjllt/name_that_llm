"""Package the native GLM coverage curve on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .kiro_pb115_notebook import KERNEL as KIRO, NATIVE


OUTPUT = Path('runs/glm_coverage_20261125/kaggle_audit')


def build(output):
    modules = ('glm_coverage_oof', 'kiro_pb115_oof', 'member_comp_replay', 'glm_prefix_response_replay',
               'peer_context_report', 'sibling_odds_replay')
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in modules:
        source = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({source!r})\n"
    execute = """
from threadpoolctl import threadpool_limits
from ntllm.glm_coverage_oof import run

mount = Path('/kaggle/input')

def asset(name, filename):
    path, = [item for item in mount.rglob(filename) if name in item.parts]
    return path

with threadpool_limits(limits=2, user_api='blas'):
    run(asset('name-that-llm', 'train.csv'),
        asset('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet'),
        asset('ntl-kiro-pb115-oof', 'predictions.parquet'),
        asset('ntl-prefixb-full-115', 'paired_token_features.parquet').parent,
        Path('/kaggle/working/glm_coverage'))
"""
    for source in (setup, execute):
        compile(source, '<glm-coverage-oof>', 'exec')
    metadata = dict(id='zhuating/ntl-glm-coverage-oof', title='NTL GLM Coverage OOF',
                    code_file='glm-coverage.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'], dataset_sources=['lumibee/ntl-prefixb-full-115/1'],
                    kernel_sources=[NATIVE, f'{KIRO}/1'])
    cells = [cell(setup), cell(execute)]
    save(output, metadata, dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3',
                                                                    'language': 'python', 'name': 'python3'}},
                                nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
