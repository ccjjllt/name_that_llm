"""Package the GPT and Claude short matched-request supply audit on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/matched_request_supply_20261125/kaggle_audit')


def build(output):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in ('matched_request_supply', 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<{module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    run = """from ntllm.matched_request_supply import run

root = Path('/kaggle/input')

def find(slug, artifact):
    path, = [item for item in root.rglob(artifact) if slug in item.parts]
    return path.parent

run(find('ntl-heldout-source-audit', 'candidate_truth.parquet'),
    find('ntl-openai-generated-responses', 'prompts.jsonl'),
    find('ntl-gen003-claude-panel', 'ntl_gen003_claude_panel.parquet'),
    Path('/kaggle/working/matched_request_supply'))
"""
    for code in (setup, run):
        compile(code, '<matched request supply>', 'exec')
    metadata = dict(id='zhuating/ntl-matched-request-supply', title='NTL Matched Request Supply',
                    code_file='matched-request-supply.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=['enaimaan/ntl-openai-generated-responses/3',
                                     'tadeuhayashida/ntl-gen003-claude-panel/1'],
                    kernel_sources=['zhuating/ntl-heldout-source-audit/2'])
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
