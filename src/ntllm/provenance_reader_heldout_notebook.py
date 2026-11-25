"""Apply both trained provenance readers to the frozen Prvsiyan holdout."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .provenance_reader_preflight_notebook import BASELINE


KERNEL = 'zhuating/ntl-provenance-readers-heldout'


def build(output, reader_version):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    modules = ('provenance_reader_inference', 'sibling_specialist', 'paired_attribution', 'sibling_odds_replay')
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    run = '''from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time

import pandas as pd
from ntllm.provenance_reader_inference import predict_reader

mount = Path('/kaggle/input')
arms = ('length_matched', 'provenance_balanced')
base, = [p.parent for p in mount.rglob('4b-base/1/config.json')]
artifacts, = [p.parent for p in mount.rglob('data_report.json')
              if all((p.parent / (arm + '.pt')).is_file() for arm in arms)]
public, = [p for p in mount.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
audit, = [p.parent for p in mount.rglob('candidate_truth.parquet')
          if (p.parent / 'candidate_queries.parquet').is_file()]

source = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
index = pd.read_parquet(audit / 'candidate_truth.parquet', columns=['id', 'source_dataset'])
queries = pd.read_parquet(audit / 'candidate_queries.parquet', columns=['id', 'text'])
if not queries.id.is_unique or not index.id.equals(queries.id):
    raise ValueError('Source audit ID order differs')
held = queries.loc[index.source_dataset.eq(source)].copy()
archived = pd.read_parquet(public, columns=['id', 'text'])
all_queries = pd.concat([held, archived], ignore_index=True)
if len(held) != 895 or len(archived) != 798 or not all_queries.id.is_unique:
    raise ValueError('Expected the complete original blinded input panels')

scratch = Path('/kaggle/temp')
scratch.mkdir(exist_ok=True)
query_file = scratch / 'readers_heldout_queries.csv'
all_queries.to_csv(query_file, index=False)
output = Path('/kaggle/working/heldout')
output.mkdir(exist_ok=True)
id_hash = hashlib.sha256(('\\n'.join(held.id) + '\\n').encode()).hexdigest()
manifest = dict(rows=len(held), ordered_ids_sha256=id_hash,
                input_body_sha256=[hashlib.sha256(body.encode()).hexdigest() for body in held.text])
deadline = time.time() + 540

def run_arm(gpu, arm):
    logits, timing = predict_reader(query_file, base, artifacts, arm, deadline, devices=(gpu,), scratch=scratch)
    if not logits.index.equals(pd.Index(all_queries.id, name='id')):
        raise ValueError(f'{arm} changed the raw query ID order')
    expected = pd.read_parquet(artifacts / f'{arm}_logits.parquet').loc[archived.id, logits.columns]
    observed = logits.loc[archived.id]
    delta = (observed - expected).abs().to_numpy()
    timing.update(archived_rows=len(expected), same_archived_argmax=int(expected.idxmax(1).eq(observed.idxmax(1)).sum()),
                  archived_mean_abs_delta=float(delta.mean()), archived_max_abs_delta=float(delta.max()))
    logits.loc[held.id].to_parquet(output / f'{arm}_logits.parquet')
    return timing

with ThreadPoolExecutor(max_workers=2) as workers:
    jobs = {arm: workers.submit(run_arm, gpu, arm) for gpu, arm in enumerate(arms)}
    manifests = {arm: job.result() for arm, job in jobs.items()}
manifest['timing'] = manifests
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(manifests, flush=True)
'''
    parent = json.loads(BASELINE.read_text())
    metadata = {
        'id': KERNEL, 'title': 'NTL Provenance Readers Heldout', 'code_file': 'readers-heldout.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': False,
        'docker_image': parent['docker_image'], 'docker_image_pinning_type': 'original',
        'machine_shape': parent['machine_shape'],
        'model_sources': ['qwen-lm/qwen-3/Transformers/4b-base/1'],
        'kernel_sources': [f'zhuating/ntl-provenance-readers/{reader_version}',
                           'zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-heldout-source-audit/1'],
    }
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reader-version', type=int, required=True)
    args = parser.parse_args()
    build(args.output, args.reader_version)
