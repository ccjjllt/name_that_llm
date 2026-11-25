"""Embed a fitted stage-only selector in the exact scored Sibling Odds notebook."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from textwrap import indent

from .atlas_notebook import save
from .counterfactual_deployment import POLICIES
from .glm_prefix_hybrid import replace_once, source


PINNED = Path('runs/sibling_odds_gemini_dash_20261124/source/pinned-metadata.json')
OUTPUT = Path('runs/counterfactual_selection_20261125/kaggle_candidate')
KERNEL = 'zhuating/ntl-sibling-odds-counterfactual'
PARENT = 'ranjeet258/name-that-llm-sibling-odds'
ORIGINAL_SHA256 = '3d5ffd92697dd1f19f229ae05370ebcf55efba61dddfadfc40acca625cbe65c6'
METADATA_SHA256 = 'e5086329d0e4e0a8b54b748bdefff96f5130b5507eb930335ad37ee5e45e8d65'
FINAL_CELL = 52
FINAL_SHA256 = 'd9ad64ffc76fcda1fbdfce650c9364386f8ba272aa84b99c996c7e8883fb32b7'
VISUAL_CELL = 2
MODULES = ('counterfactual_deployment', 'counterfactual_selection', 'counterfactual_native_guard',
           'glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay')
FEATURES = tuple('feature_' + name for name in (
    'same_family', 'short', 'stage_votes', 'incumbent_votes', 'glm_winner', 'fable_guard', 'glm_incumbent',
    'fable_incumbent', 'glm_candidate', 'fable_candidate', 'original_typography_change', 'member_only',
    'member_top1_groups', 'member_top2_groups', 'member_covered_groups', 'incumbent_member_top1_groups',
    'margin_stack_logp', 'margin_glm_logp', 'margin_sibling_logp',
))


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(values, width):
    return isinstance(values, list) and len(values) == width and all(map(numeric, values))


def validate_payload(payload):
    required = {'features', 'threshold', 'mean', 'scale', 'coefficient', 'intercept', 'outcomes',
                'label_order', 'arm', 'version'}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError('Embed only the exported parameter JSON and label schema')
    if payload['arm'] not in ('stage_c1', 'stage_multisource') or payload['version'] != 1:
        raise ValueError('Expected an exported stage-only candidate selector')
    if payload['features'] != list(FEATURES):
        raise ValueError('The exported candidate features do not match the stage-only runtime')
    labels = payload['label_order']
    if (not isinstance(labels, list) or len(labels) != 18
            or any(not isinstance(label, str) or not re.fullmatch(r'[a-z0-9][a-z0-9.\-]{0,60}', label)
                   for label in labels) or labels != sorted(set(labels))):
        raise ValueError('Expected the sorted, unique 18-label class schema')
    threshold = payload['threshold']
    outcomes = payload['outcomes']
    if (not numeric(threshold) or not 0. <= threshold <= 1. or not isinstance(outcomes, list)
            or len(outcomes) not in (2, 3) or any(type(value) is not int or value not in (-1, 0, 1) for value in outcomes)
            or outcomes != sorted(set(outcomes))):
        raise ValueError('Invalid fixed threshold or fitted outcome schema')
    width = len(FEATURES)
    categories = 1 if len(outcomes) == 2 else 3
    coefficient = payload['coefficient']
    if (not vector(payload['mean'], width) or not vector(payload['scale'], width)
            or min(payload['scale']) <= 0. or not vector(payload['intercept'], categories)
            or not isinstance(coefficient, list) or len(coefficient) != categories
            or not all(vector(row, width) for row in coefficient)):
        raise ValueError('Invalid fitted scaling or logit coefficient matrix')
    return payload


def runtime_patch(payload, policy):
    # Only this final decision cell changes. Source reads here are local packaging.
    package = '''_cf_root = Path('/kaggle/temp/ntl-counterfactual-candidate')
_cf_package = _cf_root / 'ntllm'
_cf_package.mkdir(parents=True, exist_ok=True)
(_cf_package / '__init__.py').touch()
'''
    for module in MODULES:
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<embedded {module}>', 'exec')
        package += f"(_cf_package / {module + '.py'!r}).write_text({code!r})\n"
    model_json = json.dumps(payload, separators=(',', ':'), sort_keys=True, allow_nan=False)
    package += f'_cf_model = json.loads({model_json!r})\n_cf_policy = {policy!r}\n'
    prefix = '''# Score fixed alternatives only after the original decision is complete.
if _report.get('status') == 'applied':
    try:
        from pathlib import Path
        import sys
        if len(query_df) != len(current_sub) or not query_df.id.is_unique:
            raise ValueError('Invalid counterfactual query IDs')
        _cf_ids = query_df.id.astype(str).to_numpy(dtype=str)
        if not np.array_equal(_cf_ids, current_sub.id.astype(str).to_numpy(dtype=str)):
            raise ValueError('Counterfactual query and submission ID orders differ')
        if not os.path.isfile(FINAL_LOGITS):
            raise ValueError('The original stack scores are unavailable')
        if 'ids' not in z_raw or not np.array_equal(z_raw['ids'].astype(str), _cf_ids):
            raise ValueError('The original stack score IDs are unavailable or differ')
'''
    suffix = '''sys.path.insert(0, str(_cf_root))
from ntllm.counterfactual_deployment import score_predictions
_cf_glm = np.asarray(labels)[_post.argmax(1)]
if not np.array_equal(_cf_glm, current_sub.model.to_numpy()):
    raise ValueError('Original GLM prediction order differs')
_cf_trace = dict(ids=_cf_ids, labels=np.asarray(labels, dtype=str), stack_logp=base_lp,
                 glm_logp=_post, sibling_logp=refined_lp, glm=_cf_glm,
                 glm_winner=np.isin(_cf_glm, ('glm-5.3', 'glm-5.3-flash')), fable_guard=_protect)
_cf_result = score_predictions(query_df[['id', 'text']], _cf_trace, current_preds.copy(),
                               _report, _cf_model, _cf_policy)
if _cf_result.shape != current_preds.shape or not np.isin(_cf_result, labels).all():
    raise ValueError('Invalid counterfactual prediction labels or shape')
print('Counterfactual selector', _cf_model['arm'], _cf_policy, 'changed',
      int((current_preds != _cf_result).sum()), flush=True)
current_preds = _cf_result
'''
    fallback = '''    except Exception as _cf_error:
        print('Counterfactual selector kept original calls:', type(_cf_error).__name__, flush=True)
else:
    print('Counterfactual selector kept original native-GLM fallback', flush=True)
'''
    return indent(prefix + indent(package + suffix, '        ') + fallback, '        ')


def apply_postprocess(base, payload, policy):
    if policy not in POLICIES:
        raise ValueError('Choose an explicit audited counterfactual policy')
    anchor = "        current_sub['model'] = current_preds\n"
    patched = replace_once(base, anchor, runtime_patch(validate_payload(payload), policy) + anchor)
    compile(patched, '<fixed Sibling Odds counterfactual final cell>', 'exec')
    return patched


def build(output, pinned_path, weights_path, policy):
    metadata_bytes = pinned_path.read_bytes()
    meta = json.loads(metadata_bytes)
    if meta.get('id') != PARENT or sha256(metadata_bytes) != METADATA_SHA256:
        raise ValueError('The pinned Sibling Odds origin or input metadata changed')
    original = (pinned_path.parent / meta['code_file']).read_bytes()
    if sha256(original) != ORIGINAL_SHA256:
        raise ValueError('The fixed Sibling Odds script source changed')
    notebook = json.loads(original)
    target = notebook['cells'][FINAL_CELL]
    base = source(target)
    if target['cell_type'] != 'code' or sha256(base.encode()) != FINAL_SHA256:
        raise ValueError('The expected final Sibling Odds decision cell changed')
    model_bytes = weights_path.read_bytes()
    payload = validate_payload(json.loads(model_bytes))
    patched = apply_postprocess(base, payload, policy)
    target['source'] = patched
    visual = notebook['cells'][VISUAL_CELL]
    if visual['cell_type'] != 'markdown':
        raise ValueError('The expected visual tour markdown cell changed')
    # The four inline screenshots exceed Kaggle's 1 MB source limit after embedding the selector.
    shortened, removed = re.subn(r'<img\b[^>]*\bsrc="data:image/[^";]+;base64,[A-Za-z0-9+/=]+"[^>]*>',
                                 '', source(visual))
    if removed != 4:
        raise ValueError('The expected visual tour screenshots changed')
    visual['source'] = shortened + (
        '\n<p>Charts in the <a href="https://www.kaggle.com/code/ranjeet258/'
        'name-that-llm-sibling-odds?scriptVersionId=372780350">original Sibling Odds notebook</a>.</p>\n'
    )
    meta.pop('id_no', None)
    meta.update(id=KERNEL, title='NTL Sibling Odds Counterfactual', code_file='counterfactual-candidate.ipynb',
                is_private=True, docker_image_pinning_type='original')
    save(output, meta, notebook)
    return dict(parent=PARENT, source_sha256=ORIGINAL_SHA256, edited_cell=FINAL_CELL, final_cell=FINAL_CELL,
                markdown_images_removed=removed, markdown_cell=VISUAL_CELL,
                module_setup_cell=FINAL_CELL, embedded_modules=list(MODULES), arm=payload['arm'],
                policy=policy, model_sha256=sha256(model_bytes), final_cell_sha256=sha256(patched.encode()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--policy', required=True, choices=POLICIES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--pinned', type=Path, default=PINNED)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.pinned, args.model, args.policy), indent=2))
