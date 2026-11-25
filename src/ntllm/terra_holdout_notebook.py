"""Stage the frozen Terra/GPT-5.5 holdout for a private baseline trace."""

import argparse
from pathlib import Path

import pandas as pd

from .atlas_notebook import cell, save
from .gen003_sparse import DENNIS, prepare_trace
from .glm_prefix_hybrid import cell_index, replace_once, source


KERNEL = 'zhuating/ntl-terra-gpt55-holdout-trace'
DEFAULT_HOLDOUT = Path('runs/terra_pair_20261124/data/holdout')
DEFAULT_OUT = Path('runs/terra_pair_20261124/kaggle_trace')


def build(queries, output, source_dir=DENNIS, *, kernel=KERNEL,
          title='NTL Terra GPT55 Holdout Trace', probe_name='Terra/GPT-5.5', pinned=None):
    if list(queries) != ['id', 'text'] or queries.empty:
        raise ValueError('Holdout input must contain id and text only')
    if queries.isna().any().any() or not queries.id.is_unique:
        raise ValueError('Holdout query IDs and texts must be present and IDs unique')
    if not queries.id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Holdout query IDs must be nonempty strings')
    if not queries.text.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Holdout query texts must be nonempty strings')

    metadata, notebook = prepare_trace(queries, source_dir=source_dir)
    setup = notebook['cells'][cell_index(notebook, 'import glob\n')]
    setup['source'] = replace_once(
        source(setup), "assert list(probe) == ['id', 'text'] and probe.id.is_unique\n",
        "assert list(probe) == ['id', 'text'] and probe.id.is_unique\n"
        f"print(f'Frozen {probe_name} probe: {{len(probe)}} unlabeled passages', flush=True)\n",
    )
    inference = notebook['cells'][cell_index(notebook, 'LAYERS = [')]
    inference['source'] = replace_once(
        source(inference), "OUT = '/kaggle/working/submission.parquet'",
        "OUT = '/kaggle/working/atlas_baseline/probe_predictions.parquet'",
    )
    notebook['cells'].append(cell("""
import hashlib

assert len(trace['labels']) == 18
report = dict(
    rows=len(probe),
    ordered_ids_sha256=hashlib.sha256(('\\n'.join(probe.id) + '\\n').encode()).hexdigest(),
    input_body_sha256=[
        hashlib.sha256(text.encode()).hexdigest() for text in probe.text
    ],
    prediction_classes=len(trace['labels']),
    stage_changes=dict(
        stack_to_glm=int((trace['stack'] != trace['glm']).sum()),
        glm_to_sibling_residual=int((trace['glm'] != trace['siblings']).sum()),
        dennis_to_sibling_odds=int((trace['final'] != sibling).sum()),
    ),
)
(proof / 'probe_manifest.json').write_text(json.dumps(report, indent=2))
print(f"Holdout trace complete, {report['prediction_classes']} classes", report['stage_changes'], flush=True)
"""))
    metadata.pop('id_no', None)
    if pinned is not None:
        def unversioned(name):
            stem, _, tail = name.rpartition('/')
            return stem if tail.isdigit() else name

        for field in ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources'):
            versions = {unversioned(name): name for name in pinned[field]}
            names = [unversioned(name) for name in metadata[field]]
            if set(versions) != set(names):
                raise ValueError(f'Pinned inputs do not match the underlying trace for {field}')
            metadata[field] = [versions[name] for name in names]
    metadata.update(
        id=kernel, title=title, is_private=True,
        enable_gpu=True, enable_internet=False, docker_image_pinning_type='original',
    )
    save(output, metadata, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(pd.read_parquet(args.holdout / 'queries.parquet'), args.out, args.source)


if __name__ == '__main__':
    main()
