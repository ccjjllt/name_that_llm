"""Build the pinned Kaggle CPU Fable source audit."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .fable_source_supply import SEYIT, TADEU, TELLS
from .focal_source_audit import POOL, SOURCE, STACK_FILES


OUTPUT = Path('runs/fable_source_supply_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-fable-independent-source-audit'
V19 = 'crtproduction/name-that-llm-open-pool/19'
PINS = ('zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/2',
        'zhuating/ntl-focal-source-audit/2')


def build(output=OUTPUT, v19_source_version=None):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in ('fable_source_supply', 'focal_source_audit', 'heldout_source_audit'):
        text = Path(__file__).with_name(module + '.py').read_text()
        compile(text, f'<embedded {module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({text!r})\n"
    run = """from threadpoolctl import threadpool_limits
from ntllm.fable_source_supply import run

with threadpool_limits(limits=2):
    run(Path('/kaggle/input'), Path('/kaggle/working/fable_source_supply'))
"""
    metadata = dict(id=KERNEL, title='NTL Fable Independent Source Audit', code_file='fable-sources.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=list(dict.fromkeys([TELLS, SEYIT, TADEU, POOL, SOURCE, *STACK_FILES])),
                    kernel_sources=list(PINS))
    if v19_source_version is not None:
        metadata.update(id='zhuating/ntl-fable-source-v19-screen', title='NTL Fable Source V19 Screen',
                        competition_sources=[], dataset_sources=[V19, TELLS],
                        kernel_sources=[f'{KERNEL}/{v19_source_version}'])
        run = run.replace('import run', 'import check_v19').replace(
            "run(Path('/kaggle/input'), Path('/kaggle/working/fable_source_supply'))",
            "check_v19(Path('/kaggle/input'), Path('/kaggle/working/fable_source_v19'))")
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--v19-source-version', type=int)
    args = parser.parse_args()
    build(args.output, args.v19_source_version)
