"""Replay the two Kiro post-processing changes on the published trace inputs."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


def build(output, source_version):
    setup = "from pathlib import Path\nimport base64\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    modules = ['kiro_decision_oof', 'glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay']
    if source_version is not None:
        modules.extend(['member_evidence_transfer', 'member_evidence'])
    for name in modules:
        body = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({body!r})\n"
    setup += "Path('kimi').mkdir(exist_ok=True)\n"
    for name in ('queries.parquet', 'truth.parquet'):
        content = base64.b64encode(Path(KIMI_EVAL_INDEX).with_name(name).read_bytes()).decode()
        setup += f"Path('kimi/{name}').write_bytes(base64.b64decode({content!r}))\n"
    run = """from ntllm.kiro_decision_oof import run

mount = Path('/kaggle/input')

def asset(slug, relative):
    folder, = [path for path in mount.rglob(slug) if path.is_dir() and (path / relative).is_file()]
    return (folder / relative).parent

public = asset('ntl-0719-public-decision-trace', 'atlas_baseline/public_truth.parquet')
kimi = asset('ntl-kimi-flash-holdout-trace', 'atlas_baseline/stage_scores.npz')
independent = None
"""
    if source_version is not None:
        run += """independent = (asset('ntl-heldout-source-audit', 'source_audit/candidate_truth.parquet'),
               asset('ntl-prvsiyan-member-trace', 'atlas_baseline/stage_scores.npz'))
"""
    run += "run(public, kimi, Path('kimi'), Path('/kaggle/working/kiro_decision_oof'), independent=independent)\n"
    name = 'ntl-kiro-decision-oof' if source_version is None else 'ntl-kiro-decision-transfer'
    pins = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-kimi-flash-holdout-trace/1']
    if source_version is not None:
        pins.extend(['zhuating/ntl-heldout-source-audit/1', f'zhuating/ntl-prvsiyan-member-trace/{source_version}'])
    for code in (setup, run):
        compile(code, '<kiro-decision-oof>', 'exec')
    metadata = {
        'id': 'zhuating/' + name, 'title': name.replace('-', ' ').title(), 'code_file': 'kiro-decision-oof.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'kernel_sources': pins,
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-version', type=int)
    args = parser.parse_args()
    build(args.output, args.source_version)
