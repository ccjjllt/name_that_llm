"""Package exact consensus-versus-typography replay on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/unanimous_rule_replay_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-unanimous-rule-replay'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2')


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    modules = ('unanimous_rule_replay', 'member_evidence', 'member_evidence_transfer', 'peer_context_report',
               'sibling_odds_replay')
    for name in modules:
        code = Path(__file__).with_name(f'{name}.py').read_text()
        compile(code, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    encoded = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({encoded!r}))\n"
    run = '''from ntllm.unanimous_rule_replay import run

def find(slug, name):
    file, = [path for path in Path('/kaggle/input').rglob(name) if slug in path.parts]
    return file.parent

run(
    find('ntl-sibling-odds-member-trace', 'member_scores.npz'),
    find('ntl-0719-public-decision-trace', 'public_truth.parquet'),
    find('ntl-prvsiyan-member-trace', 'member_scores.npz'),
    find('ntl-heldout-source-audit', 'candidate_truth.parquet'),
    Path('prior_kimi.parquet'),
    Path('/kaggle/working/unanimous_rule_replay'),
)
'''
    for code in (setup, run):
        compile(code, '<unanimous rule replay>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Unanimous Rule Replay', code_file='unanimous-rule-replay.ipynb',
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
