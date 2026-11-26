"""Package the fixed-budget Flash-first route comparison for Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/glm_flash_cascade_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-glm-flash-cascade'
PINS = ('zhuating/ntl-glm-likelihood-dynamics/2', 'zhuating/ntl-kiro-pb115-oof/1',
        'zhuating/ntl-glm-coverage-oof/1')


def build(output=OUTPUT):
    setup = ("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    modules = ('glm_flash_cascade', 'kiro_pb115_oof', 'member_comp_replay', 'glm_prefix_response_replay',
               'peer_context_report', 'sibling_odds_replay')
    for name in modules:
        code = Path(__file__).with_name(f'{name}.py').read_text()
        compile(code, f'<{name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    run = '''from threadpoolctl import threadpool_limits
from ntllm.glm_flash_cascade import run

def asset(slug, name):
    file, = [p for p in Path('/kaggle/input').rglob(name) if slug in p.parts]
    return file

with threadpool_limits(limits=2, user_api='blas'):
    run(
        asset('name-that-llm', 'train.csv'),
        asset('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet'),
        asset('ntl-prefixb-full-115', 'paired_token_features.parquet').parent,
        asset('ntl-glm-coverage-oof', 'predictions.parquet'),
        asset('ntl-kiro-pb115-oof', 'predictions.parquet'),
        Path('/kaggle/working/glm_flash_cascade'),
    )
'''
    for code in (setup, run):
        compile(code, '<Flash cascade>', 'exec')
    metadata = dict(id=KERNEL, title='NTL GLM Flash Cascade', code_file='glm-flash-cascade.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'], dataset_sources=['lumibee/ntl-prefixb-full-115/1'],
                    kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
