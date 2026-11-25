"""Stage unlabeled Kimi/Flash short passages for a private Sibling Odds trace."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .atlas_notebook import save
from .gen003_sparse import DENNIS
from .glm_prefix_hybrid import cell_index, replace_once, source
from .terra_holdout_notebook import build


STUDY = Path('runs/kimi_flash_20261125')
PINNED_INPUTS = Path('runs/sibling_odds_gemini_dash_20261124/source/pinned-metadata.json')
KERNEL = 'zhuating/ntl-kimi-flash-holdout-trace'


def add_exposure_audit(output):
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    notebook = json.loads((output / metadata['code_file']).read_text())
    setup = notebook['cells'][cell_index(notebook, 'import glob\n')]
    text = replace_once(
        source(setup), 'open_pool = open_pool[~open_pool.flag_train_prompt_regen.astype(bool)]',
        '_unfiltered_pool = open_pool\nopen_pool = open_pool[~open_pool.flag_train_prompt_regen.astype(bool)]',
    )
    text = replace_once(text, "    active_pool_rows=len(open_pool), checkpoint_training_histories='unknown'), indent=2))\n", """
    full_pool_ids=probe.id[_keys.isin(_body_keys(_unfiltered_pool.text))].tolist(),
    full_pool_rows=len(_unfiltered_pool), active_pool_rows=len(open_pool),
    checkpoint_training_histories='unknown'), indent=2))
del _unfiltered_pool
""")
    compile(text, '<Kimi/Flash trace setup>', 'exec')
    setup['source'] = text

    stack = notebook['cells'][cell_index(notebook, '%%writefile comp_stack.py')]
    text = replace_once(source(stack), 'EXTRA = load_extra_passages()\n', r"""EXTRA = load_extra_passages()

def _audit_text_keys(values):
    return (values.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\*\*(.*)\*\*$', r'\1', regex=True).str.replace(r'\s+', ' ', regex=True))

_audit_file = pathlib.Path('/kaggle/working/atlas_baseline/body_overlap.json')
_audit = json.loads(_audit_file.read_text())
_audit_overlap = _audit_text_keys(test.text).isin(_audit_text_keys(EXTRA.text))
_audit['stack_extra_ids'] = test.id[_audit_overlap].astype(str).tolist()
_audit['stack_extra_rows'] = len(EXTRA)
_audit_file.write_text(json.dumps(_audit, indent=2) + '\n')

""")
    compile(text.split('\n', 1)[1], '<Kimi/Flash comp_stack.py>', 'exec')
    stack['source'] = text
    save(output, metadata, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, default=STUDY / 'data/holdout')
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--pin-metadata', type=Path, default=PINNED_INPUTS)
    parser.add_argument('--out', type=Path, default=STUDY / 'kaggle_trace')
    args = parser.parse_args()
    build(pd.read_parquet(args.holdout / 'queries.parquet'), args.out, args.source,
          kernel=KERNEL, title='NTL Kimi Flash Holdout Trace', probe_name='Kimi/Flash short',
          pinned=json.loads(args.pin_metadata.read_text()))
    add_exposure_audit(args.out)


if __name__ == '__main__':
    main()
