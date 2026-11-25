"""Package the C1-only member ranker transfer audit on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


KERNEL = 'zhuating/ntl-sibling-odds-member-transfer'
OUTPUT = Path('runs/heldout_source_20261125/kaggle_transfer')


def build(output):
    source = Path(__file__).parent
    setup = ("import base64\nfrom pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in ('member_evidence_transfer', 'member_agreement', 'member_evidence',
                 'peer_context_report', 'sibling_odds_replay'):
        code = (source / f'{name}.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    run = """
from ntllm.member_evidence_transfer import run

mount = Path('/kaggle/input')

def source(slug, file):
    matches = [p for p in mount.rglob(file) if slug in p.parts]
    if len(matches) != 1:
        raise ValueError(f'Expected one {file} for {slug}, found {matches}')
    return matches[0].parent

run(
    source('ntl-sibling-odds-member-trace', 'member_scores.npz'),
    source('ntl-0719-public-decision-trace', 'public_truth.parquet'),
    source('ntl-prvsiyan-member-trace', 'member_scores.npz'),
    source('ntl-heldout-source-audit', 'candidate_truth.parquet'),
    Path('kimi_panel_index.parquet'),
    Path('/kaggle/working/evidence_transfer'),
)
"""
    for code in (setup, run):
        compile(code, '<member transfer notebook>', 'exec')
    metadata = {
        'id': KERNEL, 'title': 'NTL Sibling Odds Member Transfer', 'code_file': 'member-transfer.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'kernel_sources': [
            'zhuating/ntl-sibling-odds-member-trace/1',
            'zhuating/ntl-prvsiyan-member-trace/1',
            'zhuating/ntl-0719-public-decision-trace/1',
            'zhuating/ntl-heldout-source-audit/1',
        ],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
