"""Build a Kaggle CPU notebook that uses only the frozen Sibling Odds archives."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


KERNEL = 'zhuating/ntl-glm-late-transfer'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-heldout-source-audit/2', 'zhuating/ntl-prvsiyan-member-trace/1')


def build(output):
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in ('glm_late_transfer', 'counterfactual_native_guard', 'counterfactual_selection',
                 'member_evidence_transfer', 'member_evidence', 'peer_context_report', 'glm_prefix_response_replay',
                 'sibling_odds_replay'):
        source = Path(__file__).with_name(name + '.py').read_text()
        compile(source, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({source!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    execute = """from threadpoolctl import threadpool_limits
from ntllm.glm_late_transfer import run

def asset(slug, basename):
    path, = [candidate for candidate in Path('/kaggle/input').rglob(basename) if slug in candidate.parts]
    return path.parent

with threadpool_limits(limits=2, user_api='blas'):
    run(asset('ntl-sibling-odds-member-trace', 'member_scores.npz'),
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet'),
        asset('ntl-prvsiyan-member-trace', 'member_scores.npz'),
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet'), Path('kimi_panel_index.parquet'),
        Path('/kaggle/working/glm_late_transfer'))
"""
    metadata = dict(id=KERNEL, title='NTL GLM Late Transfer', code_file='glm-late-transfer.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=list(PINS))
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
