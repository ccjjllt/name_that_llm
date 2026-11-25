"""Package two full 18-class readers with fixed sampling marginals on dual T4s."""

import argparse
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .pair_supply_notebook import READER


OUTPUT = Path('runs/pair_preserving_20261125/kaggle_train')


def build(output, sampler, prepare_only=False):
    stage = json.loads(READER.read_text())['cells'][0]
    source = Path(__file__).with_name('pair_preserving_train.py').read_text()
    setup = ''.join(stage['source']) + f"\nPath('ntllm/pair_preserving_train.py').write_text({source!r})\n"
    if not prepare_only:
        exact = Path(__file__).with_name('exact_pair_train.py').read_text()
        setup += f"Path('ntllm/exact_pair_train.py').write_text({exact!r})\n"
    run = '''from ntllm.pair_preserving_train import ARMS, evaluate, prepare

mount = Path('/kaggle/input')

def asset(slug, filename):
    match, = [p for p in mount.rglob(filename) if slug in p.parts]
    return match

official = asset('name-that-llm', 'train.csv')
public = asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent
pool = asset('name-that-llm-open-pool', 'ntl_open_pool.parquet')
archived = asset('ntl-provenance-readers', 'provenance_balanced_sample.parquet')
selected = asset('ntl-pair-preserving-sample', 'enriched_pair_ids.parquet')
independent = asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent
provider_audit = asset('ntl-matched-provider-audit', 'row_audit.parquet')
provider_raw = asset('ntl-matched-passages-four-models', 'passages.csv')
config = Path('study.json')
work = Path('/kaggle/temp/pair-preserving-reader')
outputs = Path('/kaggle/working/pair_preserving_train')
prepare(official, public, pool, Path('panels'), archived, selected, independent, provider_audit,
        provider_raw, work, outputs, config)
'''
    if prepare_only:
        run += "import shutil\nshutil.copyfile(outputs / 'data_report.json', outputs / 'report.json')\n"
    else:
        run += '''from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import torch

if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('Expected two T4 devices for the matched readers')
base, = [p.parent for p in mount.rglob('4b-base/1/config.json')]

command = [sys.executable, '-u', '-m', 'ntllm.pair_preserving_train', '--base', str(base),
           '--config', str(config), '--work', str(work), '--output', str(outputs)]
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')

def launch(arm, gpu):
    with (outputs / f'{arm}.log').open('w') as log:
        with subprocess.Popen(command + ['--arm', arm], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))) as process:
            for line in process.stdout:
                log.write(line)
                print(arm, line.rstrip(), flush=True)
            if process.wait():
                raise RuntimeError(f'{arm} training failed')

with ThreadPoolExecutor(max_workers=2) as workers:
    tasks = [workers.submit(launch, arm, gpu) for gpu, arm in enumerate(ARMS)]
    for task in tasks:
        task.result()
a, b = [json.loads((outputs / f'{arm}.json').read_text()) for arm in ARMS]
keys = ('schedule_sha256', 'scheduled_batches', 'row_exposures', 'pair_exposures')
if any(a[key] != b[key] for key in keys):
    raise ValueError('The training arms used different batches or exposures')
evaluate(public, work, outputs)
'''
    for code in (setup, run):
        compile(code, '<pair-preserving-reader>', 'exec')
    name = 'Exact Pair Loss Input Audit' if prepare_only else 'Exact Pair Loss Readers'
    slug = 'ntl-' + name.lower().replace(' ', '-')
    metadata = dict(id='zhuating/' + slug, title='NTL ' + name,
                    code_file='pair-preserving-train.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=not prepare_only, enable_internet=False,
                    competition_sources=['name-that-llm'],
                    dataset_sources=['crtproduction/name-that-llm-open-pool/18',
                                     'markbarney/ntl-matched-passages-four-models/2'],
                    kernel_sources=['zhuating/ntl-0719-public-decision-trace/1',
                                    'zhuating/ntl-provenance-readers/1',
                                    'zhuating/ntl-heldout-source-audit/2',
                                    'zhuating/ntl-matched-provider-audit/1', sampler])
    if not prepare_only:
        metadata['model_sources'] = ['qwen-lm/qwen-3/Transformers/4b-base/1']
    save(output, metadata, dict(cells=[cell(setup), cell(run)],
                                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                        'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--sampler', default='zhuating/ntl-pair-preserving-sample/2')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    build(args.output, args.sampler, args.prepare_only)
