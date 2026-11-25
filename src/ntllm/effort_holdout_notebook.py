"""Stage the Kimi/GLM reasoning-effort holdout for a private baseline trace."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .gen003_sparse import DENNIS
from .terra_holdout_notebook import build


STUDY = Path('runs/effort_pair_20261124')
PINNED_INPUTS = Path('runs/sibling_odds_gemini_dash_20261124/source/pinned-metadata.json')
KERNEL = 'zhuating/ntl-kimi-glm-effort-holdout-trace'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, default=STUDY / 'data/holdout')
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--pin-metadata', type=Path, default=PINNED_INPUTS)
    parser.add_argument('--out', type=Path, default=STUDY / 'kaggle_trace')
    args = parser.parse_args()
    build(pd.read_parquet(args.holdout / 'queries.parquet'), args.out, args.source,
          kernel=KERNEL, title='NTL Kimi GLM Effort Holdout Trace', probe_name='Kimi/GLM reasoning effort',
          pinned=json.loads(args.pin_metadata.read_text()))


if __name__ == '__main__':
    main()
