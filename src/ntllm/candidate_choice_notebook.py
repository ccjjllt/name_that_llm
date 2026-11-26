"""Package the fixed candidate-choice comparison for a Kaggle CPU run."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save
from .counterfactual_selection_notebook import BASE, setup_cell


KERNEL = 'zhuating/ntl-candidate-choice-transfer'
OUTPUT = Path('runs/candidate_choice_20261125/kaggle')
LEGACY = 'zhuating/ntl-support-censored-selector-transfer/1'
PINS = [*BASE, 'zhuating/ntl-prvsiyan-member-trace/1', LEGACY]
MODULES = ('candidate_choice', 'counterfactual_selection', 'counterfactual_selection_audit',
           'counterfactual_native_guard', 'support_censored_selector', 'peer_context_report',
           'heldout_source_audit', 'member_evidence', 'member_evidence_transfer', 'glm_prefix_response_replay',
           'sibling_odds_replay', 'provenance_family_replay')


def build(output, parity=False):
    for name in MODULES:
        compile(Path(__file__).with_name(name + '.py').read_text(), f'<embedded {name}>', 'exec')
    setup = setup_cell(MODULES)
    execute = """from threadpoolctl import threadpool_limits
from ntllm.candidate_choice import run

def asset(slug, basename):
    found, = [path for path in Path('/kaggle/input').rglob(basename) if slug in path.parts]
    return found

with threadpool_limits(limits=2):
    run(asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent,
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        asset('ntl-gen003-sparse-trace', 'frozen_predictions.parquet').parent,
        asset('ntl-kimi-flash-holdout-trace', 'probe_manifest.json').parent,
        Path('frozen'), Path('/kaggle/working/candidate_choice'),
        asset('ntl-support-censored-selector-transfer', 'predictions.parquet'),
        asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent,
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent)
"""
    kernel, title, pins = KERNEL, 'NTL Candidate Choice Transfer', PINS
    if parity:
        kernel, title = 'zhuating/ntl-candidate-capsule-parity', 'NTL Candidate Capsule Parity'
        pins = [PINS[i] for i in (0, 1, 4, 5)] + [KERNEL + '/1']
        execute = """import json

import numpy as np
import pandas as pd

from ntllm.candidate_choice import score_capsule
from ntllm.member_evidence_transfer import read_capture, read_prvsiyan
from ntllm.peer_context_report import read_ids, same_ids

def asset(slug, basename):
    found, = [path for path in Path('/kaggle/input').rglob(basename) if slug in path.parts]
    return found

public = read_ids(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'))
held = read_prvsiyan(asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
                    Path('frozen/kimi_flash/truth.parquet'))
recorded = asset('ntl-candidate-choice-transfer', 'predictions.parquet')
model = json.loads(recorded.with_name('capsule.json').read_text())
blocks = []
for truth, slug in ((public, 'ntl-sibling-odds-member-trace'), (held, 'ntl-prvsiyan-member-trace')):
    capture = asset(slug, 'member_scores.npz').parent
    trace, baseline, _, _, _ = read_capture(capture, truth)
    receipt = json.loads((capture / 'glm_report.json').read_text())
    blocks.append(score_capsule(truth[['text']].reset_index(), trace, baseline, receipt, model))
expected = read_ids(recorded)
actual = pd.concat(blocks, verify_integrity=True).loc[expected.index]
same_ids(actual.index, expected.index, 'Fixed candidate capsule', ordered=True)
if len(actual) != 1195 or len(actual.columns) != 8 or not np.array_equal(actual, expected[actual.columns]):
    raise ValueError('Unlabeled capsule replay differs from the fitted comparison')
output = Path('/kaggle/working/candidate_capsule_parity')
output.mkdir(exist_ok=True)
(output / 'report.json').write_text(json.dumps(dict(rows=len(actual), arms=list(actual.columns),
    identical_predictions=actual.size), indent=2))
print('Identical capsule predictions:', actual.size, flush=True)
"""
    for code in (setup, execute):
        compile(code, '<candidate-choice notebook>', 'exec')
    metadata = dict(id=kernel, title=title, code_file='candidate-choice.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=pins)
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parity', action='store_true')
    args = parser.parse_args()
    build(args.output or (OUTPUT.parent / 'kaggle_parity' if args.parity else OUTPUT), args.parity)
