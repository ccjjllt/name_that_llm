"""Run original-route identity checks or matched-prefix response replay on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import save
from .glm_prefix_response_notebook import AUDIT, PLAN, notebook


ARCHIVED_TRACE = 'zhuating/ntl-0719-public-decision-trace/1'
SOURCE_TRACE = 'zhuating/ntl-prvsiyan-member-trace/1'
ANALYSIS = 'zhuating/ntl-glm-prefix-response-analysis/1'


def build(output, audit):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    modules = ('glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay')
    for name in modules:
        code = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    if audit:
        name = 'NTL GLM Prefix Route Audit'
        kernel = 'zhuating/ntl-glm-prefix-route-audit'
        inputs = [ARCHIVED_TRACE]
        run = """import json

from ntllm.glm_prefix_response_replay import FABLE, GLM, restored_route
from ntllm.peer_context_report import read_ids, read_trace

truth_path, = Path('/kaggle/input').rglob('public_truth.parquet')
folder = truth_path.parent
truth = read_ids(truth_path)
trace, baseline = read_trace(folder, truth)
report = json.loads((folder / 'glm_report.json').read_text())
routed, weight = restored_route(trace, truth.text, baseline, report)
scopes = {}
for name, group in truth.assign(routed=routed).groupby('cohort'):
    focal = group.model.isin((*GLM, FABLE))
    scopes[name] = dict(rows=len(group), routed=int(group.routed.sum()), focal=int(focal.sum()),
                        routed_focal=int((focal & group.routed).sum()))
result = dict(identity=True, total_rows=len(truth), recovered_route=int(routed.sum()),
              archived_report_rows=report['n_blended'], incumbent_weight=weight, scopes=scopes)
out = Path('/kaggle/working/prefix_route_audit')
out.mkdir(exist_ok=True)
(out / 'report.json').write_text(json.dumps(result, indent=2) + '\\n')
print(result, flush=True)
"""
    else:
        name = 'NTL GLM Prefix Response Replay'
        kernel = 'zhuating/ntl-glm-prefix-response-replay'
        inputs = [SOURCE_TRACE, AUDIT, f'{PLAN}/1', ANALYSIS]
        run = """from ntllm.glm_prefix_response_replay import run

mount = Path('/kaggle/input')
trace, = [p.parent for p in mount.rglob('stage_scores.npz') if (p.parent / 'probe_manifest.json').exists()]
source, = [p.parent for p in mount.rglob('candidate_truth.parquet') if (p.parent / 'candidate_queries.parquet').exists()]
panel, = [p.parent for p in mount.rglob('truth.parquet') if p.parent.name == 'prefix_response_panel']
analysis, = [p.parent for p in mount.rglob('head_logp.npz') if p.parent.name == 'prefix_response_analysis']
run(trace, source, panel, analysis, Path('/kaggle/working/prefix_response_replay'))
"""
    for code in (setup, run):
        compile(code, '<prefix-response-replay>', 'exec')
    metadata = {
        'id': kernel, 'title': name, 'code_file': 'prefix-response-replay.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'kernel_sources': inputs,
    }
    save(output, metadata, notebook(setup, run))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    build(args.output, args.audit)
