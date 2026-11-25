"""Run the actual modified final cell with privately mounted frozen upstream scores."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .counterfactual_candidate_notebook import FINAL_CELL
from .glm_prefix_hybrid import cell_index, source


def build(output, candidate_folder, original_metadata, arm, policy):
    metadata = json.loads(original_metadata.read_text())
    original = json.loads((original_metadata.parent / metadata['code_file']).read_text())
    candidate_meta = json.loads((candidate_folder / 'kernel-metadata.json').read_text())
    candidate = json.loads((candidate_folder / candidate_meta['code_file']).read_text())
    base_cell = source(original['cells'][FINAL_CELL])
    candidate_cell = source(candidate['cells'][FINAL_CELL])
    if '_typo_fixed = 0' not in base_cell or 'score_predictions' not in candidate_cell:
        raise ValueError('The final-cell fixture does not contain the original and modified decisions')
    setup = """from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
"""
    for name in ('pngy_residual.py', 'prefix_decisions.py'):
        module = source(original['cells'][cell_index(original, f'%%writefile {name}')]).split('\n', 1)[1]
        compile(module, name, 'exec')
        setup += f"Path({name!r}).write_text({module!r})\n"
    inputs = f"""
def mounted(slug, name):
    found, = [path for path in Path('/kaggle/input').rglob(name) if slug in path.parts]
    return found

capture = mounted('ntl-sibling-odds-member-trace', 'stage_scores.npz').parent
archive = mounted('ntl-0719-public-decision-trace', 'public_truth.parquet')
truth = pd.read_parquet(archive).set_index('id', verify_integrity=True)
columns = pd.read_parquet(capture / 'frozen_predictions.parquet').set_index('id', verify_integrity=True)
with np.load(capture / 'stage_scores.npz', allow_pickle=False) as saved:
    trace = dict(saved)
ids = pd.Index(trace['ids'], name='id')
if not ids.equals(truth.index) or not ids.equals(columns.index):
    raise ValueError('Archived runtime IDs differ')
DATA = str(mounted('name-that-llm', 'train.csv').parent)
V1_OUT = Path('/kaggle/working/cell_inputs')
V1_OUT.mkdir(exist_ok=True)
scratch = Path('/kaggle/temp')
scratch.mkdir(exist_ok=True)
V1_TEST_CSV = V1_OUT / 'input.csv'
truth.text.reset_index().to_csv(V1_TEST_CSV, index=False)
FINAL_LOGITS = V1_OUT / 'final-logits.npz'
np.savez_compressed(FINAL_LOGITS, ids=trace['ids'], labels=trace['labels'], logits=trace['stack_logp'])
np.savez_compressed(scratch / 'prefix-votes.npz', ids=trace['ids'], labels=trace['labels'], logits=trace['glm_logp'])
(V1_OUT / 'report.json').write_bytes((capture / 'glm_report.json').read_bytes())
OUT = V1_OUT / 'predictions.parquet'
def reset():
    pd.DataFrame(dict(id=trace['ids'], model=trace['glm'])).to_parquet(OUT, index=False)

FAST_COMMIT = False
reset()
with threadpool_limits(limits=2, user_api='blas'):
    exec({base_cell!r})
produced = pd.read_parquet(OUT).set_index('id', verify_integrity=True)
if not produced.index.equals(ids) or not np.array_equal(produced.model, columns.sibling_odds):
    raise ValueError('The exact final cell does not reproduce the archived original decisions')

reset()
with threadpool_limits(limits=2, user_api='blas'):
    exec({candidate_cell!r})
replayed = pd.read_parquet(OUT).set_index('id', verify_integrity=True)
if not replayed.index.equals(ids) or list(replayed) != ['model']:
    raise ValueError('The modified cell changed the prediction layout')
frozen = pd.read_parquet(mounted('counterfactual-native-guard', 'predictions.parquet'))
expected = frozen.loc[frozen.arm.eq({arm!r}), {policy!r}]
if not expected.index.is_unique or not expected.index.isin(ids).all():
    raise ValueError('The remotely frozen candidate record has invalid IDs')
observed = replayed.loc[expected.index, 'model']
if not np.array_equal(observed, expected):
    raise ValueError('The actual modified final cell differs from the selected policy')
metrics = dict(original_matched=int(len(ids)), candidate_matched=int(len(expected)),
               observed_changes=int(observed.ne(produced.loc[expected.index, 'model']).sum()),
               input_count=798, arm={arm!r}, policy={policy!r})
output = Path('/kaggle/working/final_cell_parity')
output.mkdir(exist_ok=True)
(output / 'report.json').write_text(json.dumps(metrics, indent=2) + '\\n')
print(metrics, flush=True)
"""
    for content in (setup, inputs):
        compile(content, '<fixed final-cell replay>', 'exec')
    name = 'counterfactual-final-cell-parity'
    pins = dict(id=f'zhuating/{name}', title='Counterfactual Final Cell Parity',
                code_file='cell-parity.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_internet=False,
                competition_sources=['name-that-llm'],
                docker_image=metadata['docker_image'], docker_image_pinning_type='original',
                kernel_sources=['zhuating/ntl-sibling-odds-member-trace/1',
                                'zhuating/ntl-0719-public-decision-trace/1',
                                'zhuating/counterfactual-native-guard/1'])
    notebook = dict(cells=[cell(setup), cell(inputs)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, pins, notebook)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    for option in ('output', 'candidate-folder', 'original-metadata'):
        args.add_argument('--' + option, type=Path, required=True)
    args.add_argument('--arm', required=True)
    args.add_argument('--policy', required=True)
    build(**vars(args.parse_args()))
