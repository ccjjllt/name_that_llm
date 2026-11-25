"""Package the frozen Claude matched-group study for a single Colab GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from .claude_group_reader_notebook import WORKER
from .glm_prefix_hybrid import replace_once


REPO = 'Qwen/Qwen3-4B-Base'
REVISION = '906bfd4'
REMOTE_ROOT = '/content/ntl-claude-group'
RUNS = Path('runs/claude_group_20261125')
VERIFY = """import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
manifest = json.loads((root / 'bundle.json').read_text())
for name, digest in manifest['sha256'].items():
    with (root / name).open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
            raise ValueError(f'Bundle hash differs: {name}')
packages = ['transformers==4.57.3', 'bitsandbytes==0.48.2', 'accelerate']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *packages], check=True)
subprocess.run([sys.executable, '-u', 'study.py'], check=True)
"""


def build(notebook, assets, output, accelerator):
    candidate = json.loads(notebook.read_text())
    setup, run = [''.join(cell['source']) for cell in candidate['cells']]
    worker = WORKER.replace("'T4'", repr(accelerator)).replace('one assigned T4', f'one assigned {accelerator}')
    receipt_line = "write_json(args.output / f'{args.arm}.json', receipt)"
    checkpoint = """torch.save({name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if parameter.requires_grad}, args.output / f'{args.arm}.pt')"""
    worker = replace_once(worker, receipt_line, checkpoint + '\n' + receipt_line)
    compile(worker, '<Claude group Colab arm>', 'exec')
    setup = replace_once(setup, repr(WORKER), repr(worker))
    run = replace_once(run, "mount = Path('/kaggle/input')", f"mount = Path({(REMOTE_ROOT + '/assets')!r})")
    run = replace_once(run, "if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):\n"
                       "    raise RuntimeError('Expected two T4 devices for the matched study')",
                       f"if torch.cuda.device_count() != 1 or {accelerator!r} not in torch.cuda.get_device_name(0):\n"
                       f"    raise RuntimeError('Expected one {accelerator} for both matched arms')")
    base = ("from huggingface_hub import snapshot_download\n"
            f"base = Path(snapshot_download({REPO!r}, revision={REVISION!r}, local_dir='/content/ntl-qwen3-4b-base',\n"
            "                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja']))\n"
            "print('Ready', torch.cuda.get_device_name(0), 'with', sum(p.stat().st_size for p in base.glob('*.safetensors')),\n"
            "      'checkpoint bytes', flush=True)")
    run = replace_once(run, "base, = [p.parent for p in mount.rglob('4b-base/1/config.json')]", base)
    run = run.replace("'/kaggle/temp/claude_group_reader'", "'work'").replace("'/kaggle/working/claude_group_readers'", "'results'")
    run = replace_once(run, "with ThreadPoolExecutor(max_workers=2) as pool:\n"
                       "    tasks = [pool.submit(launch, arm, index) for index, arm in enumerate(ARMS)]\n"
                       "    for task in tasks:\n"
                       "        task.result()",
                       "for arm in ARMS:\n    launch(arm, 0)")
    run = run.replace('from concurrent.futures import ThreadPoolExecutor\n', '')
    run = replace_once(run, "report = dict(weight=WEIGHT, input_manifest=manifest, training=receipts, audited_group_sizes=actual_sizes,",
                       f"report = dict(checkpoint={{'repository': {REPO!r}, 'revision': {REVISION!r}}},\n"
                       "              accelerator=torch.cuda.get_device_name(0), weight=WEIGHT, input_manifest=manifest,\n"
                       "              training=receipts, audited_group_sizes=actual_sizes,")
    study = setup + '\n' + run
    compile(study, '<Claude group Colab>', 'exec')
    compile(VERIFY, '<Claude group launcher>', 'exec')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'study.py').write_text(study)
    (output / 'start.py').write_text(VERIFY)
    fingerprint = {}
    with tarfile.open(output / 'claude_group.tar', 'w') as bundle:
        for path in sorted(assets.rglob('*')):
            if path.suffix not in ('.json', '.parquet', '.npz') or not path.is_file():
                continue
            relative = f'assets/{path.relative_to(assets)}'
            bundle.add(path, arcname=relative, recursive=False)
            with path.open('rb') as handle:
                fingerprint[relative] = hashlib.file_digest(handle, 'sha256').hexdigest()
        if len(fingerprint) != 18:
            raise ValueError(f'Unexpected frozen asset count {len(fingerprint)}')
        for name in ('study.py', 'start.py'):
            path = output / name
            bundle.add(path, arcname=name, recursive=False)
            fingerprint[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = output / 'bundle.json'
        report.write_text(json.dumps({'checkpoint': [REPO, REVISION], 'accelerator': accelerator,
                                      'sha256': fingerprint}, indent=2) + '\n')
        bundle.add(report, arcname=report.name, recursive=False)
    return output / 'claude_group.tar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=RUNS / 'kaggle_train/claude-group-readers.ipynb')
    parser.add_argument('--assets', type=Path, default=RUNS / 'colab_input')
    parser.add_argument('--output', type=Path, default=RUNS / 'colab_bundle')
    parser.add_argument('--accelerator', choices=('A100', 'L4', 'T4'), default='A100')
    args = parser.parse_args()
    print(build(args.notebook, args.assets, args.output, args.accelerator))
