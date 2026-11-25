"""Package train-only topic projection on the frozen StyleDistance embeddings."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX
from .style_mixture_notebook import pinned_slug


OUTPUT = Path('runs/style_topic_projection_20261125/kaggle_audit')


def build(output, style_ref, independent):
    style_slug = pinned_slug(style_ref)
    modules = ('style_topic_projection', 'style_mixture', 'style_distance', 'provenance_readers',
               'peer_context_report', 'paired_attribution', 'sibling_specialist', 'sibling_odds_replay',
               'member_evidence', 'member_evidence_transfer')
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<{module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    prior = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({prior!r}))\n"

    run = f"""from threadpoolctl import threadpool_limits
from ntllm.style_topic_projection import run

root = Path('/kaggle/input')

def asset(slug, name):
    found, = [path for path in root.rglob(name) if slug in path.parts]
    return found

with threadpool_limits(limits=4):
    run(asset('name-that-llm', 'train.csv'),
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
        asset('name-that-llm-open-pool', 'ntl_open_pool.parquet'),
        asset({style_slug!r}, 'style_embedding.npz').parent,
        Path('prior_kimi.parquet'), Path('/kaggle/working/style_topic_projection'),
        independent=asset('ntl-prvsiyan-member-trace', 'stage_scores.npz').parent if {independent!r} else None,
    )
"""
    for content in (setup, run):
        compile(content, '<style topic projection notebook>', 'exec')
    name = 'ntl-style-topic-projection'
    sources = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/2', style_ref]
    if independent:
        name += '-transfer'
        sources.append('zhuating/ntl-prvsiyan-member-trace/1')
    metadata = dict(id='zhuating/' + name, title=name.replace('-', ' ').title(),
                    code_file='style-topic-projection.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/18'], kernel_sources=sources)
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--style-ref', required=True)
    parser.add_argument('--independent', action='store_true')
    build(**vars(parser.parse_args()))
