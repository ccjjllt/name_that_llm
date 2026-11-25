"""Package the frozen stage-selector replay on the independent DRJ panel."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/focal_source_20261125/kaggle_selector')
KERNEL = 'zhuating/ntl-focal-selector'
PINS = ('zhuating/ntl-focal-source-audit/2', 'zhuating/ntl-focal-source-member-trace/1',
        'zhuating/counterfactual-deployment-audit/1')


REPLAY = r"""import hashlib
import json

import numpy as np
import pandas as pd

from ntllm.counterfactual_deployment import FABLE_AUDIT_POLICY, POLICIES, score_predictions
from ntllm.counterfactual_native_guard import verify_route_receipt
from ntllm.member_evidence_transfer import EXPOSURE_KEYS, read_capture
from ntllm.peer_context_report import paired_stats, read_ids, same_ids


def artifact(pin, filename):
    _, slug, _ = pin.split('/')
    path, = [path for path in Path('/kaggle/input').rglob(filename) if slug in path.parts]
    return path


def summarize(rows, names, labels):
    def paired(part):
        return [paired_stats(part, 'baseline', name, 5000) for name in names] if len(part) else []

    by_true_class = {}
    for label in labels:
        group = rows.loc[rows.model.eq(label)]
        by_true_class[label] = dict(rows=len(group), prompts=int(group.prompt_id.nunique()),
                                    native_routed=int(group.native_routed.sum()),
                                    baseline_correct=int(group.baseline.eq(label).sum()),
                                    paired=paired(group))
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()),
                native_routed=int(rows.native_routed.sum()), paired=paired(rows),
                by_true_class=by_true_class)


audit_pin, trace_pin, deployment_pin = PINS
policies = (*POLICIES, FABLE_AUDIT_POLICY)
labels = ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3')
query_file = artifact(audit_pin, 'queries.parquet')
audit = query_file.parent
capture = artifact(trace_pin, 'stage_scores.npz').parent
deployment = artifact(deployment_pin, 'stage_c1.json').parent
queries = read_ids(query_file)
truth = read_ids(audit / 'truth.parquet')
same_ids(queries.index, truth.index, 'Focal audit truth', ordered=True)
if list(queries) != ['text'] or queries.text.isna().any() or truth.prompt_id.isna().any():
    raise ValueError('Expected the original unlabeled queries and complete truth metadata')
audit_report = json.loads((audit / 'report.json').read_text())
retained = audit_report['retained']
if (len(truth) != retained['rows'] or set(truth.model) != set(labels)
        or truth.model.value_counts().to_dict() != retained['by_label']
        or not truth.source_dataset.eq(audit_report['source']).all()
        or not truth.n_words.eq(queries.text.str.split().str.len()).all()
        or truth.loc[truth.n_words.lt(100), 'model'].value_counts().to_dict() != retained['short_by_label']):
    raise ValueError('The focal truth differs from the frozen admission report')

manifest = json.loads((capture / 'probe_manifest.json').read_text())
if (manifest['source_audit'] != audit_pin
        or manifest['queries_sha256'] != hashlib.sha256(query_file.read_bytes()).hexdigest()):
    raise ValueError('The trace was produced from a different source audit query artifact')
trace, incumbent, _, _, _ = read_capture(capture, truth.join(queries))
receipt = json.loads((capture / 'glm_report.json').read_text())
routed, blend_weight = verify_route_receipt(receipt, queries, trace, incumbent)
rows = truth[['model', 'prompt_id', 'n_words', 'identity_evidence']].assign(
    baseline=incumbent, native_routed=routed)

unlabeled = queries.reset_index()[['id', 'text']]
names = []
artifacts = {}
for arm in ('stage_c1', 'stage_multisource'):
    path = deployment / f'{arm}.json'
    model_bytes = path.read_bytes()
    model = json.loads(model_bytes)
    if model['arm'] != arm or model['version'] != 1 or model['label_order'] != list(trace['labels']):
        raise ValueError(f'The frozen model capsule has wrong identity or class order for {arm}')
    artifacts[arm] = dict(sha256=hashlib.sha256(model_bytes).hexdigest(), threshold=model['threshold'])
    for policy in policies:
        name = f'{arm}__{policy}'
        predicted = score_predictions(unlabeled, trace, incumbent, receipt, model, policy)
        if predicted.shape != incumbent.shape or not np.isin(predicted, trace['labels']).all():
            raise ValueError(f'{name} returned invalid predictions')
        rows[name] = predicted
        names.append(name)

exposed = set()
counts = {key: None for key in EXPOSURE_KEYS}
overlap_file = capture / 'body_overlap.json'
if overlap_file.is_file():
    exposure = json.loads(overlap_file.read_text())
    for key in EXPOSURE_KEYS:
        if key in exposure:
            counts[key] = len(exposure[key])
            exposed.update(exposure[key])
    if not exposed.issubset(rows.index):
        raise ValueError('The trace corpus audit contains passages outside the focal source')
rows['runtime_corpus_overlap'] = rows.index.isin(exposed)
scopes = {'focal': rows}
if exposed:
    scopes.update(without_runtime_exact_match=rows.loc[~rows.runtime_corpus_overlap],
                  with_runtime_exact_match=rows.loc[rows.runtime_corpus_overlap])
panels = [dict(scope=scope, subset=subset, **summarize(part, names, labels))
          for scope, group in scopes.items()
          for subset, part in (('all', group), ('short', group.loc[group.n_words.lt(100)]))]

output = Path('/kaggle/working/focal_selector')
output.mkdir(exist_ok=True)
report = dict(pins=dict(audit=audit_pin, trace=trace_pin, deployment=deployment_pin),
              source_audit=retained, policies=list(policies), capsules=artifacts,
              native_glm=dict(status=receipt.get('status'), rows=int(routed.sum()), weight=blend_weight),
              runtime_exact_overlap=dict(unique_rows=len(exposed), counts=counts), panels=panels)
rows.to_parquet(output / 'paired_predictions.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(f'Focal source replay addressed {len(rows)} passages with {len(names)} frozen settings', flush=True)
"""


def build(output):
    modules = ('counterfactual_deployment', 'counterfactual_native_guard', 'counterfactual_selection',
               'glm_prefix_response_replay', 'member_evidence', 'member_evidence_transfer',
               'peer_context_report', 'sibling_odds_replay')
    setup = ("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    metadata = dict(id=KERNEL, title='NTL Focal Selector Replay', code_file='focal-selector.ipynb',
                    language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=False, enable_tpu=False, enable_internet=False, kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(f'PINS = {PINS!r}\n' + REPLAY)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
