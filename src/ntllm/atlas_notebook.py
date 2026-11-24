"""Build the private ATLAS training and Dennis-plus-ATLAS Kaggle notebooks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .glm_prefix_hybrid import replace_once, source


TRACE = 'zhuating/ntl-0719-public-decision-trace/1'
TRAINING = 'zhuating/ntl-0719-atlas-likelihood'
VARIANT = 'zhuating/ntl-0719-atlas-review'
MODULE = 'atlas_likelihood_runtime.py'
PROXY_MODEL = 'qwen-lm/qwen-3/Transformers/1.7b-base/1'


def cell(text):
    compile(text, '<atlas notebook cell>', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': text}


def module_cell():
    path = Path(__file__).with_name('atlas_likelihood.py')
    return cell('from pathlib import Path\nPath(' + repr(MODULE) + ').write_text(' + repr(path.read_text()) + ')\n')


def save(output, metadata, notebook):
    output.mkdir(parents=True, exist_ok=True)
    (output / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + '\n')


def training_notebook(output, config):
    setup = cell('from pathlib import Path\nPath("atlas_config.json").write_text(' + repr(config.read_text()) + ')\n')
    run = cell('''import os
import subprocess
import sys

root = Path('/kaggle/input')
trace, = [p.parent for p in root.rglob('stage_scores.npz') if (p.parent / 'public_truth.parquet').is_file()]
model, = [p.parent for p in root.rglob('1.7b-base/1/config.json')]
train, = [p for p in root.rglob('train.csv') if p.parent.name == 'name-that-llm']
pool, = root.rglob('ntl_open_pool.parquet')
subprocess.run([sys.executable, '-u', 'atlas_likelihood_runtime.py', '--model', str(model),
                '--trace', str(trace), '--pool', str(pool), '--official', str(train),
                '--config', 'atlas_config.json', '--out', '/kaggle/working/atlas'],
                env=dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'), check=True)
''')
    notebook = {'cells': [module_cell(), setup, run], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    metadata = {'id': TRAINING, 'title': 'NTL 0719 ATLAS Likelihood', 'code_file': 'atlas_training.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
                'enable_internet': False, 'competition_sources': ['name-that-llm'],
                'dataset_sources': ['crtproduction/name-that-llm-open-pool/18'], 'kernel_sources': [TRACE],
                'model_sources': [PROXY_MODEL]}
    save(output, metadata, notebook)


def review_stage(method, weight_override):
    return f'''\n# Apply just the prespecified ATLAS review after Dennis's final decision.\nATLAS_METHOD = {method!r}
ATLAS_WEIGHT_OVERRIDE = {weight_override!r}
from atlas_likelihood_runtime import (AttributionLM, apply_scores, load_tokenizer,
                                      route_candidates, route_scores)
_atlas_root = Path('/kaggle/input')
_atlas_assets, = [p.parent for p in _atlas_root.rglob('atlas_readout.pt')
                  if (p.parent / 'atlas_report.json').is_file()]
_atlas_model, = [p.parent for p in _atlas_root.rglob('1.7b-base/1/config.json')]
_atlas_report = json.loads((_atlas_assets / 'atlas_report.json').read_text())
_atlas_config = _atlas_report['config']
_atlas_calibration = _atlas_report['calibration']
if ATLAS_WEIGHT_OVERRIDE is not None:
    _atlas_calibration[ATLAS_METHOD]['weight'] = ATLAS_WEIGHT_OVERRIDE
if not FAST_COMMIT:
    import torch
    clock_deadline = NB_T0 + 8.95 * 3600
    before_review = current_preds.copy()
    routes = route_candidates(query_df.text.fillna('').tolist(), labels, P_final, before_review, _atlas_config)
    print('ATLAS routed', len(routes), 'seconds available', round(clock_deadline-time.time()), flush=True)
    if len(routes) and clock_deadline > time.time() + 20:
        _adapter = torch.load(_atlas_assets / 'atlas_readout.pt', weights_only=True, map_location='cpu')
        assert _adapter['labels'] == _atlas_report['labels']
        assert _adapter['max_tokens'] == _atlas_config['max_tokens']
        _device = 'cuda:0'
        _tokenizer = load_tokenizer(_atlas_model)
        _model = AttributionLM(_atlas_model, len(_adapter['labels']), _device, _adapter['rank'])
        _weights = {{name for name, param in _model.named_parameters() if param.requires_grad}}
        assert _weights == set(_adapter['parameters'])
        _model.load_state_dict(_adapter['parameters'], strict=False)
        _deadline = time.monotonic() + clock_deadline - time.time()
        _raw = route_scores(_model, _tokenizer, query_df, routes, _adapter['labels'],
                            _atlas_config, _device, deadline=_deadline)
        revised = apply_scores(before_review, routes, _raw, ATLAS_METHOD, _atlas_calibration)
        current_sub['model'] = revised
        current_sub.to_parquet(OUT, index=False)
        print('ATLAS complete', int(np.isfinite(_raw[ATLAS_METHOD]).all(1).sum()),
              'changed', int((revised != before_review).sum()), flush=True)
'''


def variant_notebook(source_dir, output, method, weight_override=None):
    metadata = json.loads((source_dir / 'kernel-metadata.json').read_text())
    notebook = deepcopy(json.loads((source_dir / metadata['code_file']).read_text()))
    apex = next(c for c in notebook['cells'] if '_protect = fable_return(' in source(c))
    notebook['cells'].insert(notebook['cells'].index(apex) + 1, cell(review_stage(method, weight_override)))
    notebook['cells'].insert(0, module_cell())
    metadata.pop('id_no', None)
    metadata.update(id=VARIANT, title='NTL 0719 ATLAS Review', is_private=True, enable_gpu=True,
                    enable_internet=False, docker_image_pinning_type='original')
    metadata['kernel_sources'].append(TRAINING + '/1')
    metadata['model_sources'].append(PROXY_MODEL)
    save(output, metadata, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    variant = sub.add_parser('variant')
    variant.add_argument('--source', type=Path, required=True)
    variant.add_argument('--out', type=Path, required=True)
    variant.add_argument('--method', choices=('mean', 'clipped'), required=True)
    variant.add_argument('--weight-override', type=float)
    args = parser.parse_args()
    if args.kind == 'train':
        training_notebook(args.out, args.config)
    else:
        variant_notebook(args.source, args.out, args.method, args.weight_override)


if __name__ == '__main__':
    main()
