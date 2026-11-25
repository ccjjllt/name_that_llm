"""Replay unmodified Sibling Odds on all audited 895 Prvsiyan passages."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .heldout_source_audit import DATASETS
from .kimi_flash_holdout_notebook import add_exposure_audit
from .member_trace_notebook import DENNIS, PINNED, PUBLIC, build


OUTPUT = Path('runs/heldout_source_20261125/kaggle_trace')
AUDIT = 'zhuating/ntl-heldout-source-audit/1'
KERNEL = 'zhuating/ntl-prvsiyan-member-trace'


PAYLOAD = f"""from pathlib import Path

import pandas as pd

root = Path('/kaggle/input')
queried, = root.rglob('candidate_queries.parquet')
source_index = pd.read_parquet(queried.with_name('candidate_truth.parquet'), columns=['id', 'source_dataset'])
selected = source_index.loc[source_index.source_dataset.eq({DATASETS[4]!r}), 'id']
queries = pd.read_parquet(queried, columns=['id', 'text'])
frozen = queries.loc[queries.id.isin(selected)]
assert len(frozen) == 895 and frozen.id.is_unique and frozen.id.tolist() == selected.tolist()
assert list(frozen) == ['id', 'text'] and frozen.text.notna().all()
frozen.to_csv(Path('/kaggle/working/frozen_queries.csv'), index=False)
del frozen, queries, selected, source_index
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    # Reuse the verified packaging layout. The remote mount supplies the real queries.
    known_queries = pd.read_parquet(PUBLIC, columns=['id', 'text']).head(1)
    build(known_queries, args.out, DENNIS, json.loads(PINNED.read_text()), kernel=KERNEL,
          title='NTL Prvsiyan Member Trace', probe_name='Prvsiyan member evidence',
          payload=PAYLOAD, extra_kernel_sources=[AUDIT])
    add_exposure_audit(args.out)


if __name__ == '__main__':
    main()
