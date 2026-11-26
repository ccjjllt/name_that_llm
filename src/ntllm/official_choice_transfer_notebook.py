"""Package official prompt-OOF calibration with exact held-out decision traces."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .candidate_choice_notebook import MODULES, PINS
from .counterfactual_selection_notebook import setup_cell
from .lightning_pb115_notebook import FEATURES, KIRO, NATIVE, SOURCE, extract_residual


OUTPUT = Path('runs/official_choice_transfer_20261125/kaggle')
KERNEL = 'zhuating/ntl-official-choice-transfer'


def build(output):
    modules = (*MODULES, 'official_choice_transfer', 'cached_evidence_oof', 'member_comp_replay', 'kiro_pb115_oof')
    for name in modules:
        compile(Path(__file__).with_name(name + '.py').read_text(), f'<embedded {name}>', 'exec')
    setup = setup_cell(modules) + f"Path('pngy_residual.py').write_text({extract_residual(SOURCE)!r})\n"
    execute = """from threadpoolctl import threadpool_limits
from ntllm.official_choice_transfer import run

def asset(slug, name):
    path, = [candidate for candidate in Path('/kaggle/input').rglob(name) if slug in candidate.parts]
    return path

with threadpool_limits(limits=2, user_api='blas'):
    run(asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent,
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        asset('ntl-gen003-sparse-trace', 'frozen_predictions.parquet').parent,
        asset('ntl-kimi-flash-holdout-trace', 'probe_manifest.json').parent,
        Path('frozen'),
        asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent,
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
        asset('name-that-llm', 'train.csv'),
        asset('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet'),
        asset('ntl-kiro-pb115-oof', 'predictions.parquet'),
        asset('ntl-prefixb-full-115', 'paired_token_features.parquet').parent,
        asset('ntl-candidate-choice-transfer', 'predictions.parquet'),
        asset('ntl-lumibee-typography-transfer', 'predictions.parquet'),
        Path('/kaggle/working/official_choice_transfer'))
"""
    for code in (setup, execute):
        compile(code, '<official-choice transfer notebook>', 'exec')
    pins = [*PINS[:-1], KIRO, NATIVE, 'zhuating/ntl-candidate-choice-transfer/1',
            'zhuating/ntl-lumibee-typography-transfer/1']
    metadata = dict(id=KERNEL, title='NTL Official Choice Transfer', code_file='official-choice.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=[FEATURES], kernel_sources=pins)
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
