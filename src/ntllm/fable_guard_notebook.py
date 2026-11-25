"""Diagnose Fable incumbent protection using the frozen selector proposals."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/counterfactual_selection_20261125/kaggle_fable_audit')
PIN = 'zhuating/counterfactual-native-guard-transfer/1'
RUN = """import json

import pandas as pd

from ntllm.peer_context_report import paired_stats

FABLE = 'claude-fable-5'
mount = Path('/kaggle/input')
source, = [path for path in mount.rglob('predictions.parquet') if 'counterfactual-native-guard-transfer' in path.parts]
rows = pd.read_parquet(source)
if 'id' in rows:
    rows = rows.set_index('id')
rows = rows.loc[rows.arm.eq('stage_multisource')].copy()
if not rows.index.is_unique or rows.cohort.eq('prvsiyan').sum() != 895:
    raise ValueError('Expected the frozen multisource proposals and full held-source frame')

old_fable = rows.baseline.eq(FABLE)
rows['keep_incumbent_fable'] = rows.native_guard.where(~old_fable, rows.baseline)
direct_veto = old_fable | rows.explicit_fable | rows.native_veto
rows['actual_constraints'] = rows.open.where(~(direct_veto | rows.proposed_typography_violation), rows.baseline)
rows['native_and_fable'] = rows.open.where(~direct_veto, rows.baseline)
arms = ('open', 'native_guard', 'keep_incumbent_fable', 'actual_constraints', 'native_and_fable')
scopes = {'c2': rows.cohort.eq('evaluation'), 'stress': rows.cohort.eq('reconstructed_stress'),
          'prvsiyan_895': rows.cohort.eq('prvsiyan'), 'prvsiyan_316': rows.cohort.eq('prvsiyan') & rows.strict}
panels = {}
for name, mask in scopes.items():
    for length, selector in (('full', mask), ('short', mask & rows.n_words.lt(100))):
        subset = rows.loc[selector]
        if subset.empty:
            continue
        evaluated = {}
        for arm in arms:
            adjusted = paired_stats(subset, 'baseline', arm, 2500)
            fixed = subset[arm].eq(subset.model) & subset.baseline.ne(subset.model)
            harmed = subset[arm].ne(subset.model) & subset.baseline.eq(subset.model)
            adjusted['by_true_class'] = {
                model: dict(fixed=int(fixed.loc[group.index].sum()), harmed=int(harmed.loc[group.index].sum()))
                for model, group in subset.groupby('model')}
            evaluated[arm] = adjusted
        panels[f'{name}_{length}'] = evaluated
output = Path('/kaggle/working/fable_incumbent')
output.mkdir(parents=True, exist_ok=True)
rows.to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(dict(
    scopes=panels, diagnosis='Prvsiyan revealed the Fable failure and is reused for diagnosis.',
    candidate_set='Reuses the frozen stage_multisource proposal and 0.05 decision threshold.'
), indent=2, allow_nan=False) + '\\n')
print('Wrote Fable diagnosis', len(rows), flush=True)
"""


def build(output=OUTPUT):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    for module in ('peer_context_report', 'sibling_odds_replay'):
        text = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({text!r})\n"
    meta = dict(id='zhuating/ntl-fable-incumbent-diagnosis', title='NTL Fable Incumbent Diagnosis',
                code_file='fable-incumbent.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=[PIN])
    notebook = dict(cells=[cell(setup), cell(RUN)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, meta, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
