"""Package the official LumiBee v2 typography comparisons on Kaggle CPU."""

import argparse
import base64
import hashlib
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


KERNEL = 'zhuating/ntl-lumibee-typography-transfer'
OUTPUT = Path('runs/lumibee_typography_20261125/kaggle')
PUBLIC = Path('runs/research_frontier_20261125/lumibee_0724/0-724-pb-115-likelihood-unlocked-glm-coverage.ipynb')
PUBLIC_SHA256 = '9ed68f82c13bd1a6c5b5337b019e35b234f6f2e58f7cf721fe0756abb6123d8e'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-heldout-source-audit/2', 'zhuating/ntl-prvsiyan-member-trace/1',
        'zhuating/ntl-candidate-choice-transfer/1')


def build(output):
    if hashlib.sha256(PUBLIC.read_bytes()).hexdigest() != PUBLIC_SHA256:
        raise ValueError('Expected the fixed LumiBee v2 public source')
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in ('lumibee_typography', 'counterfactual_native_guard', 'counterfactual_selection',
                 'member_evidence_transfer', 'member_evidence', 'peer_context_report', 'glm_prefix_response_replay',
                 'sibling_odds_replay'):
        source = Path(__file__).with_name(name + '.py').read_text()
        compile(source, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({source!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    execute = """from ntllm.lumibee_typography import run

def asset(slug, basename):
    path, = [candidate for candidate in Path('/kaggle/input').rglob(basename) if slug in candidate.parts]
    return path

run(asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent,
    asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
    asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
    asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent,
    Path('kimi_panel_index.parquet'),
    asset('ntl-candidate-choice-transfer', 'predictions.parquet'),
    Path('/kaggle/working/lumibee_typography'))
"""
    for source in (setup, execute):
        compile(source, '<lumibee-typography notebook>', 'exec')
    metadata = dict(id=KERNEL, title='NTL LumiBee Typography Transfer', code_file='lumibee-typography.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=list(PINS))
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
