"""Build the fixed candidate confidence transfer study on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .provenance_family_replay_notebook import bytes_payloads, first_cell


OUTPUT = Path('runs/consensus_transfer_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-consensus-transfer'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-gen003-sparse-trace/1', 'zhuating/ntl-kimi-flash-holdout-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2',
        'zhuating/counterfactual-deployment-audit/1')


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    modules = ('consensus_transfer', 'counterfactual_native_guard', 'counterfactual_selection',
               'counterfactual_selection_audit', 'glm_prefix_response_replay', 'heldout_source_audit',
               'member_evidence', 'member_evidence_transfer', 'peer_context_report', 'provenance_family_replay',
               'sibling_odds_replay', 'support_censored_selector')
    for name in modules:
        code = Path(__file__).with_name(f'{name}.py').read_text()
        compile(code, f'<{name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    gen003 = Path('runs/peer_retrieval_20261124/independent')
    kimi = Path('runs/kimi_flash_20261125/data/holdout')
    frozen = {f'gen003/{name}': (gen003 / name).read_bytes()
              for name in ('panel.json', 'visible_ids.parquet', 'queries.parquet', 'truth.parquet')}
    frozen.update({f'kimi_flash/{name}': (kimi / name).read_bytes() for name in ('queries.parquet', 'truth.parquet')})
    frozen['gen003_trace_input.csv'] = bytes_payloads(first_cell(gen003 / 'kaggle'))['/kaggle/working/gen003_queries.csv']
    for name, value in frozen.items():
        target = 'frozen/' + name
        setup += f"Path({target!r}).parent.mkdir(parents=True, exist_ok=True)\n"
        setup += f"Path({target!r}).write_bytes(base64.b64decode({base64.b64encode(value).decode()!r}))\n"
    run = '''from threadpoolctl import threadpool_limits
from ntllm.consensus_transfer import run

def asset(slug, filename):
    file, = [p for p in Path('/kaggle/input').rglob(filename) if slug in p.parts]
    return file

with threadpool_limits(limits=2, user_api='blas'):
    run(
        asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent,
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        asset('ntl-gen003-sparse-trace', 'frozen_predictions.parquet').parent,
        asset('ntl-kimi-flash-holdout-trace', 'probe_manifest.json').parent,
        Path('frozen'),
        asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent,
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
        asset('counterfactual-deployment-audit', 'stage_multisource.json'),
        Path('/kaggle/working/consensus_transfer'),
    )
'''
    for code in (setup, run):
        compile(code, '<consensus transfer notebook>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Consensus Transfer', code_file='consensus-transfer.ipynb',
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
