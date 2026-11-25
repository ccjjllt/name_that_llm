"""Package CPU reconciliation for the four already trained reader arms."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/heldout_source_20261125/kaggle_reader_transfer')


def build(output, style_ref):
    package = Path(__file__).parent
    setup = ("import base64\nfrom pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    files = ('reader_transfer', 'member_evidence_transfer', 'member_evidence', 'peer_context_report',
             'sibling_odds_replay')
    for file in files:
        setup += f"Path('ntllm/{file}.py').write_text({(package / (file + '.py')).read_text()!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    style_source = f"find({style_ref.split('/')[1]!r}, 'style_logp.parquet')" if style_ref else 'None'
    execute = f"""
from ntllm.reader_transfer import run

mount = Path('/kaggle/input')

def find(slug, name):
    files = [p for p in mount.rglob(name) if slug in p.parts]
    if len(files) != 1:
        raise ValueError(f'Expected one {{name}} from {{slug}}, found {{files}}')
    return files[0].parent

run(
    find('ntl-sibling-odds-member-trace', 'member_scores.npz'),
    find('ntl-0719-public-decision-trace', 'public_truth.parquet'),
    find('ntl-prvsiyan-member-trace', 'member_scores.npz'),
    find('ntl-heldout-source-audit', 'candidate_truth.parquet'),
    Path('kimi_panel_index.parquet'),
    {style_source},
    find('ntl-provenance-readers', 'provenance_balanced_logits.parquet'),
    find('ntl-provenance-readers-heldout', 'provenance_balanced_logits.parquet'),
    Path('/kaggle/working/reader_transfer'),
)
"""
    for text in (setup, execute):
        compile(text, '<reader transfer notebook>', 'exec')
    name = 'ntl-reader-transfer' if style_ref else 'ntl-provenance-reader-transfer'
    metadata = dict(
        id=f'zhuating/{name}', title=name.replace('-', ' ').title(), code_file='reader-transfer.ipynb',
        language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
        kernel_sources=[
            'zhuating/ntl-sibling-odds-member-trace/1',
            'zhuating/ntl-0719-public-decision-trace/1',
            'zhuating/ntl-prvsiyan-member-trace/1',
            'zhuating/ntl-heldout-source-audit/2',
            *([style_ref] if style_ref else []),
            'zhuating/ntl-provenance-readers/1',
            'zhuating/ntl-provenance-readers-heldout/1',
        ],
    )
    notebook = {'cells': [cell(setup), cell(execute)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--style-ref')
    args = parser.parse_args()
    build(args.output, args.style_ref)
