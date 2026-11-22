"""Move the public q14v5/q8v5 Gaussian support fit out of scoring."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from textwrap import indent
from zipfile import BadZipFile

import numpy as np


MEMBERS = ('q14v5', 'q8v5')
PREFIT_ASSET = 'zhuating/ntl-l24-gaussian-heads'
ARRAYS = ('mx', 'sx', 'W', 'm', 'lam')
FIT_CONFIG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=0.0001,
                  n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)


def fit_support(features, labels, prompts):
    import gauss

    params = gauss.fit(features, labels, prompts, FIT_CONFIG)
    params['T1'] = params['Tg'] = 3.0
    return params


def support_identity(spec, train, labels):
    adapters = spec.get('adapters') or [spec['adapter']]
    support = train[['id', 'text', 'model', 'prompt_id']].to_csv(index=False).encode()
    options = ('path', 'keep', 'layers', 'max_len', 'bs', 'dp', 'no_final_norm', 'eos_id')
    return {'member': spec['name'], 'adapter_order': list(adapters),
            'encoder': {key: spec.get(key) for key in options},
            'support_sha256': hashlib.sha256(support).hexdigest(), 'labels': list(labels)}


def save_support(root, identity, params, redo_batches, feature_seconds, fit_seconds):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{identity['member']}.npz"
    meta = {'identity': identity, 'bf16_redo_batches': redo_batches,
            'feature_seconds': feature_seconds, 'fit_seconds': fit_seconds,
            'cfg': params['cfg'], 'T1': params['T1'], 'Tg': params['Tg']}
    np.savez(path, metadata=np.array(json.dumps(meta, sort_keys=True)),
             **{key: params[key] for key in ARRAYS})
    return path


def read_support(root, identity):
    path = Path(root) / f"{identity['member']}.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive['metadata'].item())
            if meta['identity'] != identity or meta['cfg'] != FIT_CONFIG or (meta['T1'], meta['Tg']) != (3.0, 3.0):
                raise ValueError(f'Gaussian support in {path} differs from the current inputs')
            params = {key: archive[key] for key in ARRAYS}
    except (BadZipFile, TypeError) as error:
        raise ValueError(f'Invalid Gaussian support in {path}') from error
    d = FIT_CONFIG['d']
    width = params['mx'].size
    shapes = {'mx': (width,), 'sx': (width,), 'W': (width, d), 'm': (len(identity['labels']), d), 'lam': (d,)}
    if not all(a.shape == shapes[key] and a.dtype.kind == 'f' and np.isfinite(a).all() for key, a in params.items()):
        raise ValueError(f'Invalid Gaussian parameters in {path}')
    return {**params, 'cfg': meta['cfg'], 'T1': meta['T1'], 'Tg': meta['Tg']}, meta['bf16_redo_batches']


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Expected one occurrence in the public infer.py: {before.splitlines()[0]}')
    return source.replace(before, after, 1)


def patch_infer(source):
    """Add train-only support preparation and loading to the public L24 worker."""
    source = source.replace('    if train_texts is not None: return P_acc, (C_acc if groups is not None else None), FEATS\n',
                            '    if train_texts is not None:\n        return (P_acc, C_acc if groups is not None else None, FEATS)\n')
    source = source.replace("sp_shrink='lw', lam_floor=1e-4, n_exact=3",
                            "sp_shrink='lw', lam_floor=0.0001, n_exact=3", 1)
    source = replace_once(source, 'a = p.parse_args()\n', '''p.add_argument('--gauss_prefit_out', default='')
p.add_argument('--gauss_prefit_in', default='')
p.add_argument('--gauss_prefit_members', default='q14v5,q8v5')
a = p.parse_args()
if a.gauss_prefit_out and a.gauss_prefit_in:
    p.error('choose either Gaussian support preparation or loading')
from prefit_gauss import MEMBERS, fit_support, read_support, save_support, support_identity
''')
    source = replace_once(source, "test = pd.read_csv(os.path.join(a.data, 'test.csv'))\n",
                          "test = train.iloc[:0].copy() if a.gauss_prefit_out else pd.read_csv(os.path.join(a.data, 'test.csv'))\n")
    start = source.index('from sklearn.feature_extraction.text import TfidfVectorizer\ntry:\n')
    end = source.index('from transformers import AutoTokenizer, AutoModel\n', start)
    source = source[:start] + 'if not a.gauss_prefit_out:\n' + indent(source[start:end], '    ') + source[end:]
    source = replace_once(source, 'def l24_predict(spec, texts, groups=None, train_texts=None):\n',
                          'def l24_predict(spec, texts, groups=None, train_texts=None, support_only=False, use_prefit=False):\n')
    source = replace_once(source, '    if train_texts is not None:\n        enc_tr = _tk(train_texts)\n',
                          '    if train_texts is not None and not use_prefit:\n        enc_tr = _tk(train_texts)\n')
    source = replace_once(source, '''        Z = featurize(enc, norms, hid)
        if train_texts is not None and ai == len(adapters) - 1:
            FEATS = (Z.copy(), featurize(enc_tr, norms, hid))
''', '''        last_adapter = ai == len(adapters) - 1
        if support_only and last_adapter:
            feature_started = time.monotonic()
        Z = featurize(enc, norms, hid) if not support_only or last_adapter else None
        if support_only and last_adapter:
            FEATS = Z
            feature_seconds = round(time.monotonic() - feature_started, 1)
        if train_texts is not None and last_adapter:
            same_copies = not spec.get('dp') or len(_ms) == 2
            if use_prefit and not sum(OVF) and same_copies:
                FEATS = (Z.copy(), None)
            else:
                if use_prefit:
                    log(f"  l24 {spec['name']}: test redo or device placement changed; refitting Gaussian support")
                    enc_tr = _tk(train_texts)
                FEATS = (Z.copy(), featurize(enc_tr, norms, hid))
''')
    source = replace_once(source, "        hd = lambda p, X: lsm(X @ H[f'{p}.weight'].float().numpy().T + H[f'{p}.bias'].float().numpy())\n",
                          "        if support_only:\n            continue\n"
                          "        hd = lambda p, X: lsm(X @ H[f'{p}.weight'].float().numpy().T + H[f'{p}.bias'].float().numpy())\n")
    source = replace_once(source, '    if train_texts is not None:\n        return (P_acc, C_acc if groups is not None else None, FEATS)\n',
                          '    if support_only:\n        return FEATS, sum(OVF), feature_seconds\n'
                          '    if train_texts is not None:\n        return (P_acc, C_acc if groups is not None else None, FEATS)\n')
    source = replace_once(source, 'specs = json.loads(a.models)\nfailed = []\n', '''specs = json.loads(a.models)
if a.gauss_prefit_out:
    selected = [spec for spec in specs if spec['name'] in MEMBERS and spec.get('gauss', 0) > 0]
    if {spec['name'] for spec in selected} != set(MEMBERS):
        raise ValueError('Both Gaussian members q14v5 and q8v5 are required')
    for spec in selected:
        Ztr, redos, feature_seconds = l24_predict(spec, train.text.values, support_only=True)
        started = time.monotonic()
        params = fit_support(Ztr, y, Gtr)
        fit_seconds = round(time.monotonic() - started, 1)
        path = save_support(a.gauss_prefit_out, support_identity(spec, train, LABELS), params, redos,
                            feature_seconds, fit_seconds)
        log(f"Gaussian {spec['name']}: {len(Ztr)} train rows, {feature_seconds:.1f}s forward, "
            f"{fit_seconds:.1f}s fit, {redos} bf16 redo batches, {path}")
        del Ztr, params
    sys.exit(0)
failed = []
''')
    source = replace_once(source, '''        plain, cent, (Zte_g, Ztr_g) = l24_predict(spec, test.text.values, Gte if a.adapt_w > 0 else None, train_texts=train.text.values)
        import gauss as _ga
        prm = _ga.fit(Ztr_g, y, Gtr, dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=0.0001, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
        prm['T1'] = prm['Tg'] = 3.0
''', '''        import gauss as _ga
        prm = None
        if a.gauss_prefit_in and spec['name'] in MEMBERS and spec['name'] in a.gauss_prefit_members.split(','):
            try:
                prm, support_redos = read_support(a.gauss_prefit_in, support_identity(spec, train, LABELS))
            except (OSError, ValueError, KeyError) as error:
                log(f"  gaussian {spec['name']}: refitting support ({error})")
            else:
                log(f"  gaussian prefit {spec['name']} (support bf16 redo batches {support_redos})")
        plain, cent, (Zte_g, Ztr_g) = l24_predict(
            spec, test.text.values, Gte if a.adapt_w > 0 else None,
            train_texts=train.text.values, use_prefit=prm is not None)
        if Ztr_g is not None:
            prm = fit_support(Ztr_g, y, Gtr)
''')
    return replace_once(source, '''        _q = _ga.scores(Ztr_g[[0, 1]], np.zeros(2, dtype=int), prm)
        assert np.allclose(np.exp(_q).sum(1), 1)
''', '''        if Ztr_g is not None:
            _q = _ga.scores(Ztr_g[[0, 1]], np.zeros(2, dtype=int), prm)
            assert np.allclose(np.exp(_q).sum(1), 1)
''')


def deployment_notebook(notebook, members=MEMBERS, skip_config_only_comp=False):
    """Add optional support files to the original estopadilla v5 inference cell."""
    result = deepcopy(notebook)
    cells = result['cells']
    worker = next(i for i, cell in enumerate(cells) if ''.join(cell['source']).startswith('%%writefile infer.py\n'))
    command = next(cell for cell in cells if ''.join(cell['source']).startswith('LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]'))
    marker, original = ''.join(cells[worker]['source']).split('\n', 1)
    cells[worker]['source'] = marker + '\n' + patch_infer(original)
    asset = PREFIT_ASSET.split('/')[-1]
    find_cache = f'''from pathlib import Path
gauss_prefit_members = {tuple(members)!r}
gauss_prefit_roots = (Path('/kaggle/input/notebooks/{PREFIT_ASSET}/prefit_gauss'),
                      Path('/kaggle/input/{asset}/prefit_gauss'))
gauss_prefit = next((root for root in gauss_prefit_roots
                    if any((root / f'{{member}}.npz').is_file() for member in gauss_prefit_members)), None)
if gauss_prefit is not None:
    cmd += ['--gauss_prefit_in', str(gauss_prefit), '--gauss_prefit_members', ','.join(gauss_prefit_members)]
    print('Gaussian support input:', gauss_prefit, flush=True)
'''
    command['source'] = replace_once(''.join(command['source']), 'if FAST_COMMIT:\n', find_cache + 'if FAST_COMMIT:\n')
    cells.insert(worker, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                          'source': '%%writefile prefit_gauss.py\n' + Path(__file__).read_text()})
    if skip_config_only_comp:
        comp = next(cell for cell in cells if ''.join(cell['source']).startswith('%%writefile comp_stack.py\n'))
        comp['source'] = replace_once(''.join(comp['source']), '\ndef load_truncated_decoder(', '''
if not SMOKE:
    qwen_path = find_checkpoint('qwen3', 4096)
    if (qwen_path and qwen_path.endswith('/lora24/final/base_config')
            and os.listdir(qwen_path) == ['config.json']):
        print(f'Comp cannot load the selected Qwen3 checkpoint: {qwen_path}', flush=True)
        raise SystemExit(1)


def load_truncated_decoder(''')
    for cell in cells:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f'Attach {PREFIT_ASSET} v1 in Kaggle.')
    parser.add_argument('--public-0716', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--members', nargs='+', choices=MEMBERS, default=MEMBERS)
    parser.add_argument('--skip-config-only-comp', action='store_true',
                        help='Skip Comp if its original Qwen3 selector finds the known config-only checkpoint.')
    args = parser.parse_args()
    notebook = json.loads(args.public_0716.read_text())
    patched = deployment_notebook(notebook, args.members, args.skip_config_only_comp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(patched, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
