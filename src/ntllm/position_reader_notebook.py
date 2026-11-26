"""Build the Kaggle CPU equal-window attribution comparison."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


KERNEL = 'zhuating/ntl-position-window-reader'
OUTPUT = Path('runs/position_reader_20261125/kaggle_audit')
PINS = ('zhuating/ntl-claude-group-input/1', 'zhuating/ntl-sibling-odds-member-trace/1',
        'zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-prvsiyan-member-trace/1',
        'zhuating/ntl-heldout-source-audit/2')


def build(output):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for name in ('position_reader', 'member_evidence_transfer', 'member_evidence',
                 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(name + '.py').read_text()
        compile(code, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    prior = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({prior!r}))\n"
    execute = """from threadpoolctl import threadpool_limits
from ntllm.member_evidence_transfer import read_prvsiyan, verify_manifest
from ntllm.peer_context_report import read_ids
from ntllm.position_reader import runs

def asset(slug, name):
    path, = [path for path in Path('/kaggle/input').rglob(name) if slug in path.parts]
    return path

train = asset('ntl-claude-group-input', 'training.parquet')
public = read_ids(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'))
held = read_prvsiyan(asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent, Path('prior_kimi.parquet'))
public_capture = asset('ntl-sibling-odds-member-trace', 'stage_scores.npz').parent
held_capture = asset('ntl-prvsiyan-member-trace', 'stage_scores.npz').parent
verify_manifest(public_capture / 'probe_manifest.json', public)
verify_manifest(held_capture / 'probe_manifest.json', held)
with threadpool_limits(limits=2):
    runs(train, public, public_capture, held, held_capture, Path('/kaggle/working/position_reader'))
"""
    compile(execute, '<position-reader notebook>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Position Window Reader', code_file='position-reader.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
