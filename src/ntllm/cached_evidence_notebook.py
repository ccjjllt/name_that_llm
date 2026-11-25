"""Run broad cached evidence cross-fitting on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/cached_evidence_20261125/kaggle_audit')


def build(output):
    source = Path(__file__).parent
    modules = ('cached_evidence_oof', 'kiro_pb115_oof', 'member_comp_replay',
               'glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay')
    setup = ("from pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in modules:
        setup += f"Path('ntllm/{name}.py').write_text({(source / (name + '.py')).read_text()!r})\n"
    run = """
from threadpoolctl import threadpool_limits

from ntllm.cached_evidence_oof import run

mount = Path('/kaggle/input')

def pinned_file(slug, name):
    matches = [path for path in mount.rglob(name) if slug in path.parts]
    if len(matches) != 1:
        raise ValueError(f'Expected one fixed {name} for {slug}, found {matches}')
    return matches[0]

with threadpool_limits(limits=2, user_api='blas'):
    run(
        pinned_file('name-that-llm', 'train.csv'),
        pinned_file('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet'),
        pinned_file('ntl-kiro-pb115-oof', 'predictions.parquet'),
        pinned_file('ntl-q4-token-forecast-fingerprints', 'states_only_external_only.npz').parent,
        pinned_file('ntl-roberta-study-v1-cache', 'roberta_external_logp.parquet'),
        pinned_file('ntl-prefixb-full-115', 'paired_token_features.parquet').parent,
        pinned_file('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        Path('/kaggle/working/cached_evidence'),
    )
"""
    for code in (setup, run):
        compile(code, '<cached-evidence>', 'exec')
    metadata = dict(id='zhuating/ntl-cached-evidence-oof', title='NTL Cached Evidence OOF',
                    code_file='cached-evidence.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=['lumibee/ntl-prefixb-full-115/1', 'zhuating/ntl-roberta-study-v1-cache/1'],
                    kernel_sources=['zhuating/ntl-glm-likelihood-dynamics/2', 'zhuating/ntl-kiro-pb115-oof/1',
                                    'zhuating/ntl-q4-token-forecast-fingerprints/1',
                                    'zhuating/ntl-0719-public-decision-trace/1'])
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
