"""Build an offline Kaggle smoke and throughput run for a completed reader."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save


BASELINE = Path('runs/sibling_odds_gemini_dash_20261124/source/pinned-metadata.json')


def build(output, version, arm, baseline=BASELINE):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    modules = ('provenance_reader_inference', 'sibling_specialist', 'paired_attribution', 'sibling_odds_replay')
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    run = '''import json
import time

import pandas as pd
from ntllm.provenance_reader_inference import predict_reader

root = Path('/kaggle/input')
base, = [p.parent for p in root.rglob('4b-base/1/config.json')]
artifacts, = [p.parent for p in root.rglob('data_report.json') if (p.parent / (arm + '.pt')).is_file()]
official = root / 'competitions/name-that-llm/test.csv'
if not official.exists():
    official = root / 'name-that-llm/test.csv'
public, = [p for p in root.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
scratch = Path('/kaggle/temp')
output = Path('/kaggle/working/reader_preflight')
output.mkdir(exist_ok=True)

# Exercise the official entry point and independently measure a full unlabeled batch.
preview, preview_time = predict_reader(official, base, artifacts, arm, time.time() + 480.)
sample = scratch / 'throughput_queries.csv'
pd.read_parquet(public, columns=['id', 'text']).to_csv(sample, index=False)
timed, timing = predict_reader(sample, base, artifacts, arm, time.time() + 480.)
expected = pd.read_parquet(artifacts / (arm + '_logits.parquet')).loc[timed.index, timed.columns]
delta = (timed - expected).abs().to_numpy()
report = dict(arm=arm, preview=preview_time, throughput=timing, labels=list(timed.columns),
              archived_logits_mean_absolute_delta=float(delta.mean()),
              archived_logits_max_absolute_delta=float(delta.max()),
              archived_logits_same_argmax=int((timed.idxmax(axis=1) == expected.idxmax(axis=1)).sum()))
(output / 'report.json').write_text(json.dumps(report, indent=2) + '\\n')
print(json.dumps(report, indent=2))
'''
    metadata = {
        'id': 'zhuating/ntl-provenance-reader-preflight',
        'title': 'NTL Provenance Reader Preflight',
        'code_file': 'reader-preflight.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'competition_sources': ['name-that-llm'],
        'kernel_sources': ['zhuating/ntl-0719-public-decision-trace/1',
                           f'zhuating/ntl-provenance-readers/{version}'],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
    }
    baseline_metadata = json.loads(baseline.read_text())
    metadata.update(docker_image=baseline_metadata['docker_image'], docker_image_pinning_type='original',
                    machine_shape=baseline_metadata['machine_shape'])
    notebook = {'cells': [cell(setup), cell(f'arm = {arm!r}\n' + run)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--arm', choices=('length_matched', 'provenance_balanced'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    args = parser.parse_args()
    build(args.output, args.version, args.arm, args.baseline)
