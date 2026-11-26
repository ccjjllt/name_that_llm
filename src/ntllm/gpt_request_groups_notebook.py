"""Build the private CPU notebook for authenticated GPT request matching."""

from pathlib import Path

from .atlas_notebook import cell, save
from .gpt_request_groups import POOL, PRV


KERNEL = 'zhuating/ntl-gpt-request-groups-audit'
OUTPUT = Path('runs/gpt_request_groups_20261125/kaggle')


def build():
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for name in ('gpt_request_groups', 'heldout_source_audit', 'matched_request_supply',
                 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(f'{name}.py').read_text()
        compile(code, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    execute = 'from ntllm.gpt_request_groups import run\nrun()\n'
    metadata = dict(id=KERNEL, title='NTL GPT Request Groups Audit', code_file='gpt-request-groups.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, dataset_sources=[PRV, POOL], competition_sources=['name-that-llm'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/2'])
    notebook = dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(OUTPUT, metadata, notebook)


if __name__ == '__main__':
    build()
