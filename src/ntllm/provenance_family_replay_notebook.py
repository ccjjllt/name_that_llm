"""Package the locked provenance decisions and identity checks for Kaggle CPU."""

import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/provenance_readers_20261125/kaggle_family_replay')
PINS = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-gen003-sparse-trace/1',
        'zhuating/ntl-kimi-flash-holdout-trace/1', 'zhuating/ntl-provenance-readers/1']


def first_cell(folder):
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    notebook = json.loads((folder / metadata['code_file']).read_text())
    return ast.parse(''.join(notebook['cells'][0]['source']))


def bytes_payloads(tree):
    result = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == 'write_bytes':
            path = ast.literal_eval(call.func.value.args[0])
            value = call.args[0]
            compressed = value.func.attr == 'decompress'
            encoded = value.args[0].args[0] if compressed else value.args[0]
            payload = base64.b64decode(ast.literal_eval(encoded))
            result[path] = gzip.decompress(payload) if compressed else payload
    return result


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def build(output):
    runs = Path('runs')
    study = runs / 'provenance_readers_20261125'
    gen003 = runs / 'peer_retrieval_20261124/independent'
    kimi = runs / 'kimi_flash_20261125'
    reader_inputs = bytes_payloads(first_cell(study / 'kaggle'))
    trace_inputs = bytes_payloads(first_cell(gen003 / 'kaggle'))
    frozen = {'gen003_trace_input.csv': trace_inputs['/kaggle/working/gen003_queries.csv']}
    for name, folder in [('gen003', gen003), ('kimi_flash', kimi / 'data/holdout')]:
        for file in ('queries.parquet', 'truth.parquet'):
            relative = f'{name}/{file}'
            payload = reader_inputs[f'panels/{relative}']
            if payload != (folder / file).read_bytes():
                raise ValueError(f'Frozen panel bytes differ from the reader input: {relative}')
            frozen[relative] = payload
    for file in ('panel.json', 'visible_ids.parquet'):
        frozen[f'gen003/{file}'] = (gen003 / file).read_bytes()
    remotes = {
        'historical': (runs / 'atlas_0719_20261123/trace/atlas_baseline', ('public_truth.parquet', 'stage_scores.npz')),
        'gen003': (gen003 / 'gpu/atlas_baseline', ('stage_scores.npz', 'frozen_predictions.parquet')),
        'kimi': (kimi / 'trace_gpu/atlas_baseline', ('stage_scores.npz', 'probe_manifest.json', 'frozen_predictions.parquet')),
        'reader': (study / 'training_gpu/results', ('reader_predictions.parquet', 'length_matched_logits.parquet',
                                                  'provenance_balanced_logits.parquet')),
    }
    hashes = {key: {file: sha((folder / file).read_bytes()) for file in files}
              for key, (folder, files) in remotes.items()}
    hashes['embedded'] = {path: sha(payload) for path, payload in frozen.items()}
    setup = ("from pathlib import Path\nimport base64\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for module in ('provenance_family_replay', 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    for path, value in frozen.items():
        target = 'frozen/' + path
        setup += f"Path({target!r}).parent.mkdir(parents=True, exist_ok=True)\n"
        setup += f"Path({target!r}).write_bytes(base64.b64decode({base64.b64encode(value).decode()!r}))\n"
    setup += f"Path('frozen/hashes.json').write_text({json.dumps(hashes, indent=2)!r})\n"
    payload = """from ntllm.provenance_family_replay import run

mount = Path('/kaggle/input')

def asset(slug, file):
    path, = [p for p in mount.rglob(file) if slug in p.parts]
    return path.parent

run(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'),
    asset('ntl-gen003-sparse-trace', 'frozen_predictions.parquet'),
    asset('ntl-kimi-flash-holdout-trace', 'probe_manifest.json'),
    asset('ntl-provenance-readers', 'reader_predictions.parquet'),
    Path('frozen'), Path('/kaggle/working/provenance_family_replay'))
"""
    metadata = dict(id='zhuating/ntl-provenance-family-replay', title='NTL Provenance Family Replay',
                    code_file='provenance-family-replay.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=PINS)
    notebook = dict(cells=[cell(setup), cell(payload)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
