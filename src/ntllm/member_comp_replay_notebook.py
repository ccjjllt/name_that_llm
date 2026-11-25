"""Package the repaired Comp branch replay over both original-ensemble traces."""

import argparse
import base64
import io
import json
from pathlib import Path
import re
import tarfile

from .atlas_notebook import cell, save
from .glm_prefix_hybrid import cell_index, source
from .heldout_source_audit import KIMI_EVAL_INDEX
from .member_trace_notebook import PINNED


KERNEL = 'zhuating/ntl-sibling-odds-comp-replay'
DEFAULT_OUTPUT = Path('runs/member_evidence_20261125/kaggle_comp_replay')


def build(output, comp_version, *, gsq=False, heldout=False):
    parent = json.loads(PINNED.read_text())
    original = json.loads((PINNED.parent / parent['code_file']).read_text())
    residual_cell = original['cells'][cell_index(original, '%%writefile pngy_residual.py')]
    residual = source(residual_cell).split('\n', 1)[1]
    capsules = [m[1] for c in original['cells']
                if (m := re.search(r'^V1_PAYLOAD = "([A-Za-z0-9+/=]+)"$', source(c), re.MULTILINE))]
    encoded, = capsules
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded, validate=True)), mode='r:gz') as archive:
        ll_config = archive.extractfile('ll_model.json').read().decode()
    prior = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
             f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({prior!r}))\n"
             f"Path('pngy_residual.py').write_text({residual!r})\n"
             f"Path('ll_config.json').write_text({ll_config!r})\n")
    modules = ('member_comp_replay', 'member_comp_frontend', 'member_evidence_transfer', 'member_evidence',
               'glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay')
    if gsq:
        modules += ('gsq_member_replay',)
    for name in modules:
        code = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    run = """import json
from ntllm.member_comp_replay import run

mount = Path('/kaggle/input')

def asset(slug, relative):
    folder, = [path for path in mount.rglob(slug) if path.is_dir() and (path / relative).is_file()]
    return folder / relative

historical = asset('ntl-sibling-odds-member-trace', 'atlas_baseline/stage_scores.npz').parent
original = asset('ntl-0719-public-decision-trace', 'atlas_baseline/public_truth.parquet').parent
held = asset('ntl-prvsiyan-member-trace', 'atlas_baseline/stage_scores.npz').parent
audit = asset('ntl-heldout-source-audit', 'source_audit/candidate_queries.parquet').parent
comp = asset('ntl-sibling-odds-comp-diagnostic', 'comp_diagnostic/report.json').parent
official, = [p for p in mount.rglob('train.csv') if p.parent.name == 'name-that-llm']
run(historical, original, held, audit, Path('prior_kimi.parquet'), comp, official,
    json.loads(Path('ll_config.json').read_text()), Path('/kaggle/working/comp_replay'))
"""
    metadata = {
        'id': KERNEL, 'title': 'NTL Sibling Odds Comp Replay', 'code_file': 'comp-replay.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False, 'competition_sources': ['name-that-llm'],
        'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-sibling-odds-member-trace/1',
                           'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/1',
                           f'zhuating/ntl-sibling-odds-comp-diagnostic/{comp_version}'],
    }
    if gsq:
        identifier = 'ntl-sibling-odds-gsq-replay' + ('-transfer' if heldout else '')
        sources = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-sibling-odds-member-trace/1']
        held_args = ''
        if heldout:
            sources += ['zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2']
            held_args = """
    independent=asset('ntl-prvsiyan-member-trace', 'atlas_baseline/stage_scores.npz').parent,
    source=asset('ntl-heldout-source-audit', 'source_audit/candidate_queries.parquet').parent,
    prior=Path('prior_kimi.parquet'),
"""
        run = run[:run.index("held = asset(")].replace('ntllm.member_comp_replay', 'ntllm.gsq_member_replay')
        run += """official, = [p for p in mount.rglob('train.csv') if p.parent.name == 'name-that-llm']
run(historical, original, official, json.loads(Path('ll_config.json').read_text()),
    Path('/kaggle/working/gsq_member_replay'),
""" + held_args + ")\n"
        metadata.update(id=f'zhuating/{identifier}', title=identifier.replace('-', ' ').title(),
                        code_file='gsq-replay.ipynb', kernel_sources=sources)
    for code in (residual, setup, run):
        compile(code, '<member-replay>', 'exec')
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--comp-version', type=int, default=1)
    parser.add_argument('--gsq', action='store_true')
    parser.add_argument('--heldout', action='store_true')
    args = parser.parse_args()
    build(args.output, args.comp_version, gsq=args.gsq, heldout=args.heldout)
