"""Package the candidate selection comparison for Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .provenance_family_replay_notebook import bytes_payloads, first_cell


OUTPUT = Path('runs/counterfactual_selection_20261125/kaggle_audit')
BASE = ['zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-gen003-sparse-trace/1', 'zhuating/ntl-kimi-flash-holdout-trace/1',
        'zhuating/ntl-heldout-source-audit/2']


def setup_cell(modules):
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    gen003 = Path('runs/peer_retrieval_20261124/independent')
    kimi = Path('runs/kimi_flash_20261125/data/holdout')
    frozen = {f'gen003/{name}': (gen003 / name).read_bytes()
              for name in ('panel.json', 'visible_ids.parquet', 'queries.parquet', 'truth.parquet')}
    frozen.update({f'kimi_flash/{name}': (kimi / name).read_bytes() for name in ('queries.parquet', 'truth.parquet')})
    original = bytes_payloads(first_cell(gen003 / 'kaggle'))
    frozen['gen003_trace_input.csv'] = original['/kaggle/working/gen003_queries.csv']
    for name, value in frozen.items():
        target = 'frozen/' + name
        payload = base64.b64encode(value).decode()
        setup += f"Path({target!r}).parent.mkdir(parents=True, exist_ok=True)\n"
        setup += f"Path({target!r}).write_bytes(base64.b64decode({payload!r}))\n"
    return setup


def build(output, heldout, guard=False, deployment=False, typography=False, readers=False, family_factored=False,
          support_censored=False):
    score_study = ('family_factored_audit' if family_factored
                   else 'support_censored_selector' if support_censored else '')
    modules = ('counterfactual_selection', 'counterfactual_selection_audit', 'peer_context_report',
               'heldout_source_audit', 'member_evidence', 'member_evidence_transfer',
               'sibling_odds_replay', 'provenance_family_replay')
    if guard or deployment or readers or score_study:
        modules += ('counterfactual_native_guard', 'glm_prefix_response_replay')
    if guard:
        modules += ('counterfactual_guard_audit',)
    if deployment:
        modules += ('counterfactual_deployment', 'counterfactual_deployment_audit')
    if typography:
        modules += ('typography_transfer',)
    if readers:
        modules += ('reader_candidate_audit', 'reader_transfer')
    if score_study:
        modules += ('counterfactual_deployment', score_study)
    if family_factored:
        modules += ('family_factored_stage',)
    setup = setup_cell(modules)
    audit_module = 'counterfactual_guard_audit' if guard else 'counterfactual_selection_audit'
    working = 'counterfactual_guard' if guard else 'counterfactual_selection'
    base_reference = 'counterfactual-selection' if guard else None
    reference_file = 'predictions.parquet'
    if deployment:
        audit_module = 'counterfactual_deployment_audit'
        working = 'counterfactual_deployment'
        base_reference = 'counterfactual-native-guard'
    extras = ''
    if typography:
        audit_module = 'typography_transfer'
        working = 'typography_transfer'
        base_reference = 'counterfactual-native-guard-transfer' if heldout else 'counterfactual-native-guard'
        extras = """        pool=source('name-that-llm-open-pool', 'ntl_open_pool.parquet') / 'ntl_open_pool.parquet',
        focal=source('ntl-focal-source-audit', 'screened_rows.parquet'),
"""
    if readers:
        audit_module = 'reader_candidate_audit'
        working = 'reader_candidates'
        base_reference = 'counterfactual-native-guard'
        extras = """        training=source('ntl-provenance-readers', 'provenance_balanced_logits.parquet'),
"""
        if heldout:
            extras += """        heldout=source('ntl-provenance-readers-heldout', 'provenance_balanced_logits.parquet'),
"""
    if score_study:
        audit_module = working = score_study
        base_reference = 'counterfactual-deployment-audit'
        reference_file = 'stage_multisource.json'
    reference = (f"        reference=source({base_reference!r}, {reference_file!r}) / {reference_file!r},\n"
                 if base_reference else '')
    run = f"""from threadpoolctl import threadpool_limits
from ntllm.{audit_module} import run

mount = Path('/kaggle/input')

def source(slug, name):
    found, = [path for path in mount.rglob(name) if slug in path.parts]
    return found.parent

with threadpool_limits(limits=2, user_api='blas'):
    run(
        source('ntl-sibling-odds-member-trace', 'member_scores.npz'),
        source('ntl-0719-public-decision-trace', 'public_truth.parquet'),
        source('ntl-gen003-sparse-trace', 'frozen_predictions.parquet'),
        source('ntl-kimi-flash-holdout-trace', 'probe_manifest.json'),
        Path('frozen'), Path('/kaggle/working/{working}'),
{reference}        independent=source('ntl-prvsiyan-member-trace', 'member_scores.npz') if {heldout!r} else None,
        source=source('ntl-heldout-source-audit', 'candidate_truth.parquet'),
{extras}    )
"""
    for code in (setup, run):
        compile(code, '<counterfactual-selection>', 'exec')
    pins = BASE + (['zhuating/ntl-prvsiyan-member-trace/1'] if heldout else [])
    name = 'counterfactual-native-guard' if guard else 'counterfactual-selection'
    if deployment:
        name = 'counterfactual-deployment-audit'
    if typography:
        name = 'counterfactual-typography'
        pins.append('zhuating/ntl-focal-source-audit/2')
    if readers:
        name = 'counterfactual-reader-candidates'
        pins.append('zhuating/ntl-provenance-readers/1')
        if heldout:
            pins.append('zhuating/ntl-provenance-readers-heldout/1')
    if family_factored:
        name = 'ntl-family-factored-stage'
    if support_censored:
        name = 'ntl-support-censored-selector'
    if base_reference:
        pins += [f'zhuating/{base_reference}/1']
    if heldout:
        name += '-transfer'
    metadata = dict(id=f'zhuating/{name}', title=name.replace('-', ' ').title(),
                    code_file='counterfactual-selection.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=pins)
    if typography:
        metadata['dataset_sources'] = ['crtproduction/name-that-llm-open-pool/18']
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--heldout', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--guard', action='store_true')
    mode.add_argument('--deployment', action='store_true')
    mode.add_argument('--typography', action='store_true')
    mode.add_argument('--readers', action='store_true')
    mode.add_argument('--family-factored', action='store_true')
    mode.add_argument('--support-censored', action='store_true')
    args = parser.parse_args()
    build(args.output, args.heldout, args.guard, args.deployment, args.typography, args.readers, args.family_factored,
          args.support_censored)
