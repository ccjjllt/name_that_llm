"""Package the offline Kaggle CPU analysis of the dual-T4 member trace."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


KERNEL = 'zhuating/ntl-sibling-odds-member-evidence'


def build(output, trace_version):
    source = Path(__file__).parent
    setup = ("from pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in ('member_evidence', 'peer_context_report', 'sibling_odds_replay'):
        code = (source / f'{name}.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    run = """
from ntllm.member_evidence import run

mount = Path('/kaggle/input')
member_trace, = mount.rglob('member_scores.npz')
public_truth, = [p for p in mount.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').exists()]
run(member_trace.parent, public_truth, Path('/kaggle/working/evidence'))
"""
    metadata = {
        'id': KERNEL, 'title': 'NTL Sibling Odds Member Evidence', 'code_file': 'member-evidence.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'kernel_sources': [
            f'zhuating/ntl-sibling-odds-member-trace/{trace_version}',
            'zhuating/ntl-0719-public-decision-trace/1',
        ],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-version', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.trace_version)
