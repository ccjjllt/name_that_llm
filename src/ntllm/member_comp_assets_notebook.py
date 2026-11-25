"""Audit the Comp capsule's offline checkpoint resolution and shard inventory."""

import argparse
import ast
import json
from pathlib import Path

from .atlas_notebook import cell, save
from .comp_checkpoint_notebook import repaired_notebook
from .glm_prefix_hybrid import cell_index, source
from .member_comp_notebook import DENNIS, PINNED, comp_program


KERNEL = 'zhuating/ntl-sibling-odds-comp-assets'
OUTPUT = Path('runs/member_evidence_20261125/kaggle_comp_assets')
MOUNTS = {
    'dataset_sources': (
        'lightningv08/ntl-olmo2-native24-final-inference-assets',
        'zbigniewkirchner/ntl-qwen3-8b-lora-claude-head',
        'dhruvtibarewal/nlm-enc-v5',
        'pjmathematician/ultrabert-380m-ntl',
    ),
    'kernel_sources': (
        'lucasboesen/name-that-llm-glm4-9b-trunc25',
        'lucasboesen/name-that-llm-extract-encoders2',
        'lucasboesen/name-that-llm-electra-large-weights',
        'lucasboesen/name-that-llm-desklib-weights',
        'lucasboesen/name-that-llm-train-lora24-s7',
        'lucasboesen/name-that-llm-train-tokattn2',
        'vnton37nguyn/ntl-train-lora-qwen25-instyle',
    ),
    'model_sources': (
        'qwen-lm/qwen-3/Transformers/8b-base',
        'qwen-lm/qwen-3/Transformers/4b-base',
        'qwen-lm/qwen2.5/Transformers/7b',
    ),
}


def lookup_code(text):
    parsed = ast.parse(text)
    return next(ast.get_source_segment(text, node) for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name == 'find_checkpoint')


def declared_requests(text):
    parsed = ast.parse(text)
    names = {'GROUP_LAYER', 'LM_HEADS', 'ENC_HEADS', 'LORA_RUNS', 'TOKATTN_HEADS'}
    fragments = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id in names for target in node.targets for name in ast.walk(target)
        ):
            fragments.append(ast.get_source_segment(text, node))
        if isinstance(node, ast.If) and 'NTL_DROP_Q35' in ast.unparse(node.test):
            fragments.append("declared_token_heads = TOKATTN_HEADS.copy()")
            fragments.append(ast.get_source_segment(text, node))
    return '\n'.join(fragments)


def build(output, source_dir, pinned):
    original_meta = json.loads((source_dir / 'kernel-metadata.json').read_text())
    original = json.loads((source_dir / original_meta['code_file']).read_text())
    original_code = source(original['cells'][cell_index(original, '%%writefile comp_stack.py')]).split('\n', 1)[1]
    repaired = repaired_notebook(original)
    repaired_code = comp_program(source(repaired['cells'][cell_index(repaired, '%%writefile comp_stack.py')]))
    before = lookup_code(original_code)
    after = lookup_code(repaired_code)
    setup = ("import glob, json, os, pathlib\nfrom pathlib import Path\n"
             "os.environ['NTL_DROP_Q35'] = '1'\n"
             + before + '\noriginal_find_checkpoint = find_checkpoint\n'
             + after + '\n' + declared_requests(repaired_code))
    run = '''import hashlib

root = Path('/kaggle/input')

def inventory(directory):
    if directory is None:
        return dict(path=None, ready=False)
    directory = Path(directory)
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    fmt = ('model.safetensors', 'pytorch_model.bin', 'adapter_model.safetensors', 'adapter_model.bin')
    standalone = {path.name: path.stat().st_size for name in fmt if (path := directory / name).is_file()}
    indexes = {}
    for path in sorted(directory.glob('*.index.json')):
        weight_map = json.loads(path.read_text()).get('weight_map')
        if not weight_map:
            continue
        files = sorted(set(weight_map.values()))
        sizes = {name: (directory / name).stat().st_size for name in files if (directory / name).is_file()}
        indexes[path.name] = dict(shards=len(files), present_bytes=sizes,
                                  missing=[name for name in files if not sizes.get(name)])
    complete = (any(size > 0 for size in standalone.values())
                or any(not data['missing'] for data in indexes.values()))
    return dict(path=str(directory), ready=complete, model_type=config.get('model_type'),
                hidden_size=config.get('hidden_size'), layers=config.get('num_hidden_layers'),
                tokenizer_assets=[name for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                                                    'vocab.txt', 'spm.model', 'tokenizer.model')
                                  if (directory / name).is_file()],
                standalone_bytes=standalone, indexes=indexes)

requests = []
for name, kind, hidden, parts, extra, _ in LM_HEADS:
    requests.append((name, kind, hidden, None, max(layer for layer, _ in parts + extra) + 1))
for name, dirname, parts, _ in ENC_HEADS:
    requests.append((name, None, None, dirname, max(layer for layer, _ in parts)))
for (kind, hidden, keep), _, adapters in LORA_RUNS:
    requests.append((' + '.join(item[0] for item in adapters), kind, hidden, None, keep))

loaders = []
for name, kind, hidden, dirname, depth in requests:
    old = original_find_checkpoint(kind, hidden, dirname)
    try:
        new = find_checkpoint(kind, hidden, dirname)
    except Exception as exc:
        loaders.append(dict(module=name, query=[kind, hidden, dirname], layer_request=depth,
                            original=inventory(old), selection_error=f'{type(exc).__name__}: {exc}'))
        continue
    loaders.append(dict(module=name, query=[kind, hidden, dirname], layer_request=depth,
                        original=inventory(old), repaired=inventory(new)))

adapters = {}
for _, _, specs in LORA_RUNS:
    for name, marker, norms, anchor in specs:
        hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
        if not hits:
            adapters[name] = dict(will_execute=False, reason='original optional marker not found')
            continue
        directory = Path(hits[0]).parent
        adapters[name] = dict(will_execute=True, matching_markers=hits, weights=inventory(directory / 'adapter'),
                              norms=(directory / norms).is_file(), tokenizer=(directory / 'tokenizer').is_dir(),
                              readouts=(directory / 'readouts.npz').is_file(), anchor=(directory / anchor).is_file())

active = {item[0] for item in TOKATTN_HEADS}
token_heads = {}
for name, head, backbone in declared_token_heads:
    if name not in active:
        token_heads[name] = dict(will_execute=False, reason='NTL_DROP_Q35=1 excludes the head before lookup')
        continue
    hits = sorted(glob.glob('/kaggle/input/**/' + head, recursive=True))
    base = find_checkpoint('qwen3_5') if backbone == 'qwen3_5' else find_checkpoint(None, dir_name='deberta-v3-large')
    token_heads[name] = dict(will_execute=bool(hits and base), head_file_hits=hits, backbone=inventory(base))

folds = []
for directory in glob.glob('/kaggle/input/**/ultrabert-380m-ntl', recursive=True):
    candidates = [inventory(path) for path in sorted(Path(directory).glob('fold*'))]
    selected = [item for item in candidates if item.get('model_type') == 'ultrabert']
    if len(selected) == 5:
        folds = selected
        break

critical = [entry['module'] for entry in loaders if not entry.get('repaired', {}).get('ready')]
critical += [name for name, entry in adapters.items() if entry['will_execute']
             and not (entry['weights']['ready'] and entry['norms'] and entry['tokenizer'] and entry['readouts'])]
critical += [name for name, entry in token_heads.items() if entry['will_execute']
             and not entry['backbone']['ready']]
if len(folds) != 5 or not all(item['ready'] for item in folds):
    critical.append('ultrabert')
qwen = next(entry for entry in loaders if entry['module'] == 'q3_8b')
expected_bad = [str(cfg.parent) for cfg in root.rglob('base_config/config.json')
                if 'ntl-lora24-fivefold-training' in cfg.parts]
report = dict(env=dict(NTL_DROP_Q35=os.environ['NTL_DROP_Q35']),
              audit_scope='read only config/index JSON and file metadata; no tensor load',
              repaired_finder_sha256=finder_sha256, original_shadow_directories=expected_bad,
              qwen8_distinct=qwen.get('repaired', {}).get('path') != qwen['original']['path'],
              loaders=loaders, adapters=adapters, token_attention=token_heads, ultrabert_folds=folds,
              unresolved_or_incomplete=critical)
out = Path('/kaggle/working/comp_assets')
out.mkdir(exist_ok=True)
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\\n')
print('Unresolved or incomplete:', critical, flush=True)
for entry in loaders:
    print(entry['module'], entry.get('repaired', {}).get('path'), entry.get('selection_error', ''), flush=True)
'''
    import hashlib

    setup += f'\nfinder_sha256 = {hashlib.sha256(after.encode()).hexdigest()!r}\n'
    metadata = dict(id=KERNEL, title='NTL Sibling Odds Comp Assets', code_file='comp-assets.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_tpu=False, enable_internet=False)
    for kind, names in MOUNTS.items():
        metadata[kind] = [next(pin for pin in pinned[kind] if pin.rsplit('/', 1)[0] == name) for name in names]
    notebook = {'cells': [cell(setup), cell(run)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=DENNIS)
    parser.add_argument('--pinned', type=Path, default=PINNED)
    args = parser.parse_args()
    build(args.output, args.source, json.loads(args.pinned.read_text()))
