"""Package the frozen member candidate coverage audit on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/member_candidate_frontier_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-member-candidate-frontier'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2',
        'zhuating/ntl-styledistance-fp32/1', 'zhuating/ntl-provenance-readers/1',
        'zhuating/ntl-provenance-readers-heldout/1')


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    modules = ('member_candidate_frontier', 'counterfactual_selection', 'glm_prefix_response_replay',
               'member_evidence', 'member_evidence_transfer', 'peer_context_report', 'reader_transfer',
               'sibling_odds_replay')
    for name in modules:
        code = Path(__file__).with_name(f'{name}.py').read_text()
        compile(code, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    encoded = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({encoded!r}))\n"
    run = '''from ntllm.member_candidate_frontier import run

def find(slug, name):
    file, = [path for path in Path('/kaggle/input').rglob(name) if slug in path.parts]
    return file.parent

run(
    find('ntl-sibling-odds-member-trace', 'member_scores.npz'),
    find('ntl-0719-public-decision-trace', 'public_truth.parquet'),
    find('ntl-prvsiyan-member-trace', 'member_scores.npz'),
    find('ntl-heldout-source-audit', 'candidate_truth.parquet'),
    Path('prior_kimi.parquet'),
    find('ntl-styledistance-fp32', 'style_logp.parquet'),
    find('ntl-provenance-readers', 'provenance_balanced_logits.parquet'),
    find('ntl-provenance-readers-heldout', 'provenance_balanced_logits.parquet'),
    Path('/kaggle/working/member_candidate_frontier'),
)
'''
    for code in (setup, run):
        compile(code, '<member candidate frontier>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Member Candidate Frontier', code_file='candidate-frontier.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
