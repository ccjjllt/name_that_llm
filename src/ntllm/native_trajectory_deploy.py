"""Deploy the frozen sibling heads inside the public Native Q8 forward."""

import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax

if __package__:
    from .glm_prefix_hybrid import PREFIX_ASSET, cell_index, hybrid_notebook, replace_once, source
else:
    from glm_prefix_hybrid import PREFIX_ASSET, cell_index, hybrid_notebook, replace_once, source


HEADS_ASSET = 'zhuating/ntl-frozen-native-sibling-heads'
HEADS_FILE = 'native_frozen_sibling_heads.npz'
REPORT_FILE = 'native_frozen_siblings.json'
TRAJECTORY_DEPTHS = (6, 18, 30)
GROUPS = {'gemini_flash': ('gemini-3.7-flash', 'gemini-3.8-flash'),
          'gpt56': ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra')}
PAIR_ORDER = (
    ('gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-luna', 'gpt-5.6-sol'),
    ('gpt-5.6-luna', 'gpt-5.6-terra'),
    ('gpt-5.6-sol', 'gpt-5.6-terra'),
)


def find_heads():
    mount = Path('/kaggle/input')
    paths = [mount / 'notebooks' / HEADS_ASSET / HEADS_FILE,
             mount / HEADS_ASSET.split('/')[-1] / HEADS_FILE]
    matches = [path for path in paths if path.is_file()] or sorted(mount.rglob(HEADS_FILE))
    if len(matches) != 1 or not matches[0].with_name(REPORT_FILE).is_file():
        raise ValueError(f'Expected the head archive and layout report from {HEADS_ASSET}: {matches}')
    return matches[0]


def load_heads(path, labels, depths, hidden, weights):
    layout = json.loads(path.with_name(REPORT_FILE).read_text())['head_layout']
    if (set(layout['labels']) != set(labels) or tuple(layout['last_token_depth_order']) != tuple(depths)
            or tuple(map(tuple, layout['pair_order'])) != PAIR_ORDER
            or tuple(layout['trajectory_depth_order']) != TRAJECTORY_DEPTHS
            or layout['trajectory_parts_per_depth'] != ['middle_minus_early', 'late_minus_middle']):
        raise ValueError('Frozen Native head layout differs from the inference features')
    enabled = {label for group, siblings in GROUPS.items() if weights[group] for label in siblings}
    heads = {}
    with np.load(path, allow_pickle=False) as archive:
        for fold in range(5):
            heads[fold] = []
            for pair, (left, right) in enumerate(PAIR_ORDER):
                if left not in enabled:
                    continue
                stem = f'fold{fold}_trajectory_pair{pair}'
                arrays = tuple(archive[f'{stem}_{part}'] for part in ('direction', 'midpoint', 'scale'))
                if any(x.shape != (2 * len(TRAJECTORY_DEPTHS) * hidden,) or not np.isfinite(x).all() for x in arrays):
                    raise ValueError(f'Invalid frozen Native array {stem}')
                size = next(len(group) for group in GROUPS.values() if left in group)
                heads[fold].append((labels.index(left), labels.index(right), size, arrays))
    return heads


def record_trajectory(expert, out, pos, idx, depths, capture):
    """Read the same hidden states as the original Native classifier."""
    torch = expert.torch
    totals, owners, weights, prompt_tokens = capture
    try:
        parts = []
        for depth in TRAJECTORY_DEPTHS:
            state = out.hidden_states[depth]
            position = pos.to(state.device)
            token = torch.arange(state.shape[1], device=state.device)[None]
            n_body = (position - prompt_tokens).clamp_min(1)
            body = (token >= prompt_tokens) & (token < position[:, None])
            segment = ((token - prompt_tokens) * 3 // n_body[:, None]).clamp(0, 2)
            mask = torch.stack([body & segment.eq(i) for i in range(3)], dim=1).float()
            means = torch.bmm(mask, state.float()) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
            means = expert.norms[depths.index(depth)](means.to(expert.head_device))
            parts.extend((means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]))
        features = torch.cat(parts, dim=1).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        # Keep the Native result from this forward. This window supplies no sibling evidence.
        totals[owners[idx]] = np.nan
        return
    np.add.at(totals, owners[idx], features * weights[idx, None].astype(np.float32))


def pair_effects(features, heads, width):
    effects = np.zeros((len(features), width), dtype=np.float32)
    for start in range(0, len(features), 256):
        rows = slice(start, start + 256)
        for left, right, size, (direction, midpoint, scale) in heads:
            odds = np.clip((features[rows] - midpoint) / scale, -6, 6) @ direction
            evidence = np.clip(odds, -6, 6) / size
            effects[rows, left] += evidence
            effects[rows, right] -= evidence
    return effects


def aligned_effects(path, ids, labels):
    with np.load(path, allow_pickle=False) as archive:
        index = archive['labels'].astype(str).tolist()
        effects = archive['effects']
        if (archive['ids'].astype(str).tolist() != ids or set(index) != set(labels)
                or effects.shape != (len(ids), len(labels))):
            raise ValueError('Native trajectory effects do not align with the Apex rows and labels')
        return effects[:, [index.index(label) for label in labels]]


def rerank_siblings(probs, current, labels, effects, weights):
    top = np.argsort(-probs, axis=1, kind='stable')[:, :3]
    names = np.asarray(labels)
    revised = current.copy()
    for group, siblings in GROUPS.items():
        if not weights[group]:
            continue
        inside = np.isin(names[top], siblings)
        eligible = (names[top[:, 0]] == current) & inside[:, 0] & (inside.sum(axis=1) >= 2)
        for row in np.flatnonzero(eligible):
            allowed = top[row, inside[row]]
            score = probs[row, allowed]
            effect = effects[row, allowed]
            if not np.isfinite(effect).all() or (score <= 0).any():
                continue
            candidate = probs[row].copy()
            candidate[allowed] = score.sum() * softmax(np.log(score) + weights[group] * effect)
            winner = candidate.argmax()
            if winner in allowed:
                revised[row] = names[winner]
    return revised


def runtime_source():
    header = '%%writefile native_trajectory.py\nimport json\nfrom pathlib import Path\nimport numpy as np\nfrom scipy.special import softmax\n'
    literals = ('HEADS_ASSET', 'HEADS_FILE', 'REPORT_FILE', 'TRAJECTORY_DEPTHS', 'GROUPS', 'PAIR_ORDER')
    functions = (find_heads, load_heads, record_trajectory, pair_effects, aligned_effects, rerank_siblings)
    return header + '\n'.join(f'{name} = {globals()[name]!r}' for name in literals) + '\n\n' + '\n\n'.join(
        inspect.getsource(function) for function in functions)


def patch_native(text):
    text = replace_once(text, '        return torch.cat(parts, dim=-1)\n', '''        features = torch.cat(parts, dim=-1)
        if (capture := getattr(self, 'trajectory_capture', None)) is not None:
            from native_trajectory import record_trajectory
            record_trajectory(self, out, pos, idx, DEPTHS, capture)
        return features
''')
    text = replace_once(text, 'on_fold=None):', 'on_fold=None, trajectory_heads=None):')
    text = replace_once(text, '        finite = np.ones(n, dtype=bool)\n', '''        finite = np.ones(n, dtype=bool)
        self.trajectory_capture = None
        if trajectory_heads is not None:
            from native_trajectory import TRAJECTORY_DEPTHS, pair_effects
            prefix = len(self.tokenizer(INSTRUCTION, add_special_tokens=False)['input_ids'])
            res['trajectory_effects'] = []
''')
    text = replace_once(text, '            t0 = time.time()\n', '''            t0 = time.time()
            if trajectory_heads is not None:
                totals = np.zeros((n, 2 * len(TRAJECTORY_DEPTHS) * self.hidden), dtype=np.float32)
                self.trajectory_capture = totals, owners, weights, prefix
''')
    text = replace_once(text, "            res['folds'].append(int(fold))\n", '''            res['folds'].append(int(fold))
            if trajectory_heads is not None:
                totals /= denom.astype(np.float32)
                res['trajectory_effects'].append(pair_effects(totals, trajectory_heads[int(fold)], len(self.labels)))
                self.trajectory_capture = None
                del totals
''')
    text = replace_once(text, "    tmp = str(path) + '.tmp.npz'\n", '''    if res['folds'].tolist() != list(range(5)) and 'trajectory_effects' in res:
        res = {key: value for key, value in res.items() if key != 'trajectory_effects'}
    tmp = str(path) + '.tmp.npz'
''')
    text = replace_once(text, '    res, finite = model.predict(', '''    trajectory_heads = None
    if head_path := os.environ.get('NTL_NATIVE_TRAJECTORY_HEADS'):
        from native_trajectory import load_heads
        try:
            trajectory_heads = load_heads(Path(head_path), model.labels, DEPTHS, model.hidden,
                                          json.loads(os.environ['NTL_NATIVE_TRAJECTORY_WEIGHTS']))
        except (OSError, ValueError, KeyError) as error:
            log('native sibling heads skipped:', repr(error))
    res, finite = model.predict(''')
    return replace_once(text, 'on_fold=lambda r, f: save(a.out, frame.id.values, model.labels, r, f))',
                        'on_fold=lambda r, f: save(a.out, frame.id.values, model.labels, r, f),\n'
                        '                                trajectory_heads=trajectory_heads)')


def patch_infer(text):
    text = replace_once(text, '    return probs\n', '''    effects = None
    if folds == list(range(5)) and 'trajectory_effects' in z:
        effects = z['trajectory_effects'][:, :, order].mean(axis=0)
        effects[~z['finite']] = np.nan
    return probs, effects
''')
    text = replace_once(text, 'probs = native_stage(P)\n', 'probs, trajectory = native_stage(P)\n')
    save = next(line for line in text.splitlines() if line.strip() == 'save_final_logits(np.log(np.maximum(probs, 1e-15)))')
    indent = save[:len(save) - len(save.lstrip())]
    publish = '''if trajectory is not None and (sidecar := os.environ.get('NTL_NATIVE_TRAJECTORY_EFFECTS')):
    try:
        np.savez(sidecar, ids=test['id'].astype(str).to_numpy(dtype=str), labels=np.asarray(LABELS), effects=trajectory)
    except OSError as error:
        log('native trajectory sidecar skipped:', repr(error))
'''
    return replace_once(text, save + '\n', save + '\n' + ''.join(indent + line + '\n' for line in publish.splitlines()))


def patch_apex(text):
    text = replace_once(text, 'penalty=0.,post=None):', 'penalty=0.,post=None,trajectory=None,trajectory_weights=None):')
    text = replace_once(text, '    return out,refined\n', '''    if trajectory is not None:
        from native_trajectory import GROUPS, rerank_siblings
        routed = rerank_siblings(probs, out, labels, trajectory, trajectory_weights)
        changed = routed != out
        changes = {group: int((changed & np.isin(out, members)).sum()) for group, members in GROUPS.items()}
        print(json.dumps({'stage': 'native_sibling_trajectory', 'changes': changes, 'weights': trajectory_weights}), flush=True)
        out = routed
    return out,refined
''')
    text = replace_once(text, 'corrected=False,penalty=0.):',
                        'corrected=False,penalty=0.,trajectory_path=None,trajectory_weights=None):')
    text = replace_once(text, '    predictions,_=apply_arrays(', '''    trajectory = None
    if trajectory_path is not None and trajectory_path.is_file():
        from native_trajectory import aligned_effects
        trajectory = aligned_effects(trajectory_path, ids, labels)
    predictions,_=apply_arrays(''')
    return replace_once(text, 'strength,corrected,penalty,post=post)\n',
                        'strength,corrected,penalty,post=post,trajectory=trajectory,trajectory_weights=trajectory_weights)\n')


def patch_public_0716_apex(notebook):
    marker = ('# ============================================================\n'
              '# 🏆 13. APEX GRANDMASTER V2: SIBLING RESIDUAL & TYPOGRAPHIC INVARIANTS\n')
    cell = notebook['cells'][cell_index(notebook, marker)]
    anchor = "        current_sub['model'] = current_preds\n"
    cell['source'] = replace_once(source(cell), anchor, '''        if NATIVE_TRAJECTORY_EFFECTS.is_file():
            try:
                from native_trajectory import GROUPS, aligned_effects, rerank_siblings
                trajectory = aligned_effects(NATIVE_TRAJECTORY_EFFECTS, current_sub.id.astype(str).tolist(), labels)
                routed = rerank_siblings(P_final, current_preds, labels, trajectory, NATIVE_TRAJECTORY_WEIGHTS)
                changed = routed != current_preds
                changes = {group: int((changed & np.isin(current_preds, members)).sum())
                           for group, members in GROUPS.items()}
                print(json.dumps({'stage': 'native_sibling_trajectory', 'changes': changes,
                                  'weights': NATIVE_TRAJECTORY_WEIGHTS}), flush=True)
                current_preds = routed
            except (OSError, ValueError, KeyError) as error:
                print(f'Native sibling trajectories skipped: {error}', flush=True)

''' + anchor)


def add_native_trajectory(notebook, gpt56_weight=0.15, gemini_weight=0.0, public_0716=False):
    weights = {'gpt56': gpt56_weight, 'gemini_flash': gemini_weight}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError('Sibling weights must be finite and nonnegative')
    result = deepcopy(notebook)
    if not any(weights.values()):
        return result
    for marker, patch in (('%%writefile native_q8.py', patch_native), ('%%writefile infer.py', patch_infer)):
        cell = result['cells'][cell_index(result, marker)]
        cell['source'] = patch(source(cell))
    if public_0716:
        patch_public_0716_apex(result)
    else:
        cell = result['cells'][cell_index(result, '%%writefile apex_postprocess.py')]
        cell['source'] = patch_apex(source(cell))
        apex = result['cells'][cell_index(result, 'if not FAST_COMMIT:\n    from apex_postprocess import apply_postprocessing')]
        apex['source'] = replace_once(source(apex), 'strength=0.1, corrected=False, penalty=0.15)',
                                      'strength=0.1, corrected=False, penalty=0.15,\n'
                                      '                              trajectory_path=NATIVE_TRAJECTORY_EFFECTS,\n'
                                      '                              trajectory_weights=NATIVE_TRAJECTORY_WEIGHTS)')
    first = cell_index(result, 'LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]')
    settings = f'''from pathlib import Path
from native_trajectory import find_heads
NATIVE_TRAJECTORY_WEIGHTS = {weights!r}
NATIVE_TRAJECTORY_EFFECTS = Path('/kaggle/temp/ntl-native-sibling-effects.npz')
NATIVE_TRAJECTORY_EFFECTS.unlink(missing_ok=True)
os.environ.pop('NTL_NATIVE_TRAJECTORY_HEADS', None)
try:
    os.environ['NTL_NATIVE_TRAJECTORY_HEADS'] = str(find_heads())
except (OSError, ValueError) as error:
    print(f'Native sibling heads unavailable: {{error}}', flush=True)
else:
    os.environ['NTL_NATIVE_TRAJECTORY_WEIGHTS'] = json.dumps(NATIVE_TRAJECTORY_WEIGHTS)
    os.environ['NTL_NATIVE_TRAJECTORY_EFFECTS'] = str(NATIVE_TRAJECTORY_EFFECTS)
'''
    result['cells'][first:first] = [
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': content}
        for content in (runtime_source(), settings)
    ]
    for cell in result['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f'Attach {HEADS_ASSET} v1 and {PREFIX_ASSET} in Kaggle.')
    for flag in ('base', 'donor', 'hybrid', 'public-0716'):
        parser.add_argument(f'--{flag}', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpt56-weight', type=float, default=0.15)
    parser.add_argument('--gemini-weight', type=float, default=0.0)
    args = parser.parse_args()
    if bool(args.hybrid) + bool(args.public_0716) == 1 and not (args.base or args.donor):
        notebook = json.loads((args.hybrid or args.public_0716).read_text())
    elif args.base and args.donor and not (args.hybrid or args.public_0716):
        notebook = hybrid_notebook(json.loads(args.base.read_text()), json.loads(args.donor.read_text()))
    else:
        parser.error('provide --public-0716, --hybrid, or both --base and --donor')
    patched = add_native_trajectory(notebook, args.gpt56_weight, args.gemini_weight, public_0716=bool(args.public_0716))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(patched, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
