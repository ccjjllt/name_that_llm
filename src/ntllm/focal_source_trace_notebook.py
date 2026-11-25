"""Package the frozen DRJ passages for a private offline Sibling Odds trace."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .atlas_notebook import cell, save
from .kimi_flash_holdout_notebook import add_exposure_audit
from .member_trace_notebook import DENNIS, PINNED, build as build_member_trace


OUTPUT = Path('runs/focal_source_20261125/kaggle_trace')
AUDIT = 'zhuating/ntl-focal-source-audit/2'
KERNEL = 'zhuating/ntl-focal-source-member-trace'


PAYLOAD = f"""from pathlib import Path
import hashlib
import json

import pandas as pd

root = Path('/kaggle/input')
query_path, = [p for p in root.rglob('focal_source/queries.parquet')
               if {AUDIT.split('/')[1]!r} in p.parts]
expected_rows = json.loads(query_path.with_name('report.json').read_text())['retained']['rows']
queries = pd.read_parquet(query_path)
if (list(queries) != ['id', 'text'] or len(queries) != expected_rows or queries.empty
        or queries.isna().any().any() or not queries.id.is_unique
        or not queries.id.str.fullmatch(r'focal-[0-9a-f]{{24}}').all()
        or not queries.id.is_monotonic_increasing or not queries.text.str.strip().ne('').all()):
    raise ValueError('The frozen focal query schema, count, or order differs from the audit')

focal_manifest = dict(
    source_audit={AUDIT!r},
    queries_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),
    rows=len(queries),
    ordered_ids_sha256=hashlib.sha256(('\\n'.join(queries.id) + '\\n').encode()).hexdigest(),
    input_body_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in queries.text],
)
queries.to_csv('/kaggle/working/frozen_queries.csv', index=False, lineterminator='\\r\\n')
del queries
print('Frozen DRJ focal audit:', focal_manifest['source_audit'], focal_manifest['rows'],
      focal_manifest['queries_sha256'], flush=True)
"""


FINISH = """manifest_path = proof / 'probe_manifest.json'
trace_manifest = json.loads(manifest_path.read_text())
for key in ('rows', 'ordered_ids_sha256', 'input_body_sha256'):
    if trace_manifest[key] != focal_manifest[key]:
        raise ValueError(f'DRJ query {key} changed in the inference input')

overlap = json.loads((proof / 'body_overlap.json').read_text())
trace_manifest.update(source_audit=focal_manifest['source_audit'],
                      queries_sha256=focal_manifest['queries_sha256'],
                      runtime_exact_overlap={
                          key: len(overlap[key]) if key in overlap else None
                          for key in ('official_train_ids', 'active_pool_ids', 'full_pool_ids', 'stack_extra_ids')
                      })
manifest_path.write_text(json.dumps(trace_manifest, indent=2) + '\\n')
print('DRJ runtime exact overlaps:', trace_manifest['runtime_exact_overlap'], flush=True)
"""


def build(output, source_dir=DENNIS, pinned=PINNED):
    # The shared notebook builder replaces this placeholder cell before saving.
    placeholder = pd.DataFrame({'id': ['placeholder'], 'text': ['placeholder']})
    build_member_trace(placeholder, output, source_dir, json.loads(pinned.read_text()),
                       kernel=KERNEL, title='NTL Focal Source Member Trace',
                       probe_name='DRJ focal source', payload=PAYLOAD, extra_kernel_sources=[AUDIT])
    add_exposure_audit(output)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    notebook = json.loads((output / metadata['code_file']).read_text())
    notebook['cells'].append(cell(FINISH))
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
