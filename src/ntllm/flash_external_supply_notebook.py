"""Build the private Kaggle CPU audit of the pinned Flash harvest."""

from pathlib import Path
import sys

from .atlas_notebook import cell, save


OUTPUT = Path('runs/flash_external_supply_20261125/kaggle')
KERNEL = 'zhuating/ntl-flash-external-supply-audit'
EXPLICIT_KERNEL = 'zhuating/ntl-flash-external-supply-explicit-stop-audit'


def build(explicit=False):
    module = Path(__file__).with_name('flash_external_supply.py').read_text()
    compile(module, '<embedded flash supply audit>', 'exec')
    setup = cell(f'from pathlib import Path\nPath("flash_external_supply.py").write_text({module!r})\n')
    run = cell(f'from flash_external_supply import run\nrun(explicit={explicit!r})\n')
    metadata = dict(id=KERNEL, title='NTL Flash External Supply Audit', code_file='flash-external-supply.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=True, competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/19'])
    notebook = dict(cells=[setup, run], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    output = OUTPUT
    if explicit:
        metadata.update(id=EXPLICIT_KERNEL, title='NTL Flash External Supply Explicit Stop Audit')
        output = OUTPUT.with_name('kaggle_explicit')
    save(output, metadata, notebook)


if __name__ == '__main__':
    build(explicit='--explicit' in sys.argv[1:])
