"""Package an inference-only comparison of two real Sibling Odds runs."""

import argparse
import hashlib
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/counterfactual_selection_20261125/kaggle_runtime_stability')
PINS = ['zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-sibling-odds-member-trace/1',
        'zhuating/counterfactual-deployment-audit/1']
WEIGHTS = Path('runs/counterfactual_selection_20261125/deployment_remote/counterfactual_deployment')
PUBLIC_TRUTH = Path('runs/atlas_0719_20261123/trace/atlas_baseline/public_truth.parquet')
ARMS = ('stage_c1', 'stage_multisource')


RUN = '''import hashlib
import json

import numpy as np
import pandas as pd

from ntllm.counterfactual_deployment import score_predictions
from ntllm.counterfactual_native_guard import verify_route_receipt
from ntllm.counterfactual_selection import STAGES, build_actions, replay_scoring_model
from ntllm.member_evidence_transfer import verify_manifest
from ntllm.peer_context_report import paired_stats, read_ids, read_trace, same_ids

RUNS = ('original', 'rerun')
POLICIES = ('open', 'native_guard')


def asset(slug, name):
    files = [p for p in Path('/kaggle/input').rglob(name) if slug in p.parts]
    if len(files) != 1:
        raise ValueError(f'Expected one {name} from {slug}, found {files}')
    return files[0]


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    return dict(median=float(values.median()), p95=float(values.quantile(.95)), maximum=float(values.max()))


def agreement(panel, action):
    a, b = [panel[f'{name}_{action}'] for name in RUNS]
    before, after = [panel[f'{name}_baseline'] for name in RUNS]
    original_changed, rerun_changed = a.ne(before), b.ne(after)
    together = original_changed & rerun_changed
    same_incumbent = before.eq(after)
    return dict(original_changes=int(original_changed.sum()), rerun_changes=int(rerun_changed.sum()),
                original_only=int((original_changed & ~rerun_changed).sum()),
                rerun_only=int((rerun_changed & ~original_changed).sum()), both_changed=int(together.sum()),
                same_replacement_label=int((together & a.eq(b)).sum()),
                both_changed_same_incumbent=int((together & same_incumbent).sum()),
                both_changed_same_call_from_same_incumbent=int((together & same_incumbent & a.eq(b)).sum()),
                final_disagreements=int(a.ne(b).sum()), final_disagreements_same_incumbent=int((a.ne(b) & same_incumbent).sum()),
                fixes_in_both=int((together & a.eq(panel.model) & b.eq(panel.model)).sum()))


def summarize(panel):
    same = panel.loc[panel.original_baseline.eq(panel.rerun_baseline)]
    routed_a, routed_b = panel.original_route, panel.rerun_route
    result = dict(rows=len(panel), same_incumbent=len(same),
                  baseline_drift=paired_stats(panel, 'original_baseline', 'rerun_baseline', 3000),
                  routing=dict(original=int(routed_a.sum()), rerun=int(routed_b.sum()),
                               both=int((routed_a & routed_b).sum()), original_only=int((routed_a & ~routed_b).sum()),
                               rerun_only=int((routed_b & ~routed_a).sum())), stages={}, arms={})
    for stage in STAGES:
        result['stages'][stage] = dict(
            different_winner=int(panel[f'original_{stage}_winner'].ne(panel[f'rerun_{stage}_winner']).sum()),
            original_top2_gap=describe(panel[f'original_{stage}_gap']), rerun_top2_gap=describe(panel[f'rerun_{stage}_gap']),
            absolute_gap_delta=describe(panel[f'{stage}_abs_gap_change']), centered_logp_drift=describe(panel[f'{stage}_score_drift']))
    for arm in ARMS:
        original, rerun = (panel[f'{name}_{arm}_proposal'] for name in RUNS)
        result['arms'][arm] = dict(raw_proposal_agreement=int(original.eq(rerun).sum()),
            raw_proposal_agreement_same_incumbent=int(original.loc[same.index].eq(rerun.loc[same.index]).sum()),
            absolute_expected_delta_drift=describe((panel[f'original_{arm}_delta'] - panel[f'rerun_{arm}_delta']).abs()),
            policies={})
        for policy in POLICIES:
            action = f'{arm}_{policy}'
            outcome = agreement(panel, action)
            outcome['same_run'] = {name: paired_stats(panel, f'{name}_baseline', f'{name}_{action}', 3000)
                                   for name in RUNS}
            outcome['direct_rerun'] = paired_stats(panel, f'original_{action}', f'rerun_{action}', 3000)
            if len(same):
                outcome['same_incumbent'] = {name: paired_stats(same, f'{name}_baseline', f'{name}_{action}', 3000)
                                             for name in RUNS}
            result['arms'][arm]['policies'][policy] = outcome
    return result


truth_path = asset('ntl-0719-public-decision-trace', 'public_truth.parquet')
if digest_file(truth_path) != EXPECTED_TRUTH:
    raise ValueError('The original published trace truth bytes differ from the archived cohort')
truth = read_ids(truth_path)
if tuple(truth.cohort.value_counts().get(name, 0) for name in ('development', 'evaluation', 'reconstructed_stress')) != (180, 300, 318):
    raise ValueError('The two traces must cover the original frozen 798 passages')
folders = dict(original=truth_path.parent,
               rerun=asset('ntl-sibling-odds-member-trace', 'probe_manifest.json').parent)
verify_manifest(folders['rerun'] / 'probe_manifest.json', truth)
loaded = {name: read_trace(folder, truth) for name, folder in folders.items()}
frozen = read_ids(folders['rerun'] / 'frozen_predictions.parquet')
same_ids(frozen.index, truth.index, 'Saved repeated-run calls', ordered=True)
if (not np.array_equal(frozen.sibling_odds, loaded['rerun'][1])
        or not np.array_equal(frozen.dennis, loaded['rerun'][0]['final'])):
    raise ValueError('The repeated run did not reproduce its frozen calls')
labels = loaded['original'][0]['labels']
if not np.array_equal(labels, loaded['rerun'][0]['labels']):
    raise ValueError('The original and repeated run have different class orders')
queries = truth.text.to_frame().reset_index()[['id', 'text']]
features = truth[['text', 'n_words']]
if not features.text.str.split().str.len().eq(features.n_words).all():
    raise ValueError('The saved lengths disagree with the identical bodies')
rows = truth[['model', 'cohort', 'prompt_id', 'n_words']].copy()
weights = {}
for arm, sha in MODEL_HASHES.items():
    path = asset('counterfactual-deployment-audit', f'{arm}.json')
    if digest_file(path) != sha:
        raise ValueError(f'Portable artifact {arm} differs from the scored audit')
    model = json.loads(path.read_text())
    if model['arm'] != arm or model['label_order'] != labels.tolist():
        raise ValueError(f'Model {arm} has a different score or label schema')
    weights[arm] = model
native_receipts = {}
for name, (trace, incumbent) in loaded.items():
    rows[f'{name}_baseline'] = incumbent
    receipt = json.loads((folders[name] / 'glm_report.json').read_text())
    route, blend = verify_route_receipt(receipt, truth, trace, incumbent)
    rows[f'{name}_route'] = route
    native_receipts[name] = dict(status=receipt['status'], blended=int(route.sum()), weight=blend)
    for stage in STAGES:
        scores = trace[stage]
        two = np.sort(scores, axis=1)[:, -2:]
        rows[f'{name}_{stage}_gap'] = two[:, 1] - two[:, 0]
        rows[f'{name}_{stage}_winner'] = labels[scores.argmax(1)]
    bank = build_actions(features, trace, incumbent)
    for arm, model in weights.items():
        portable = replay_scoring_model(bank, model, truth.index)
        for policy in POLICIES:
            rows[f'{name}_{arm}_{policy}'] = score_predictions(queries, trace, incumbent, receipt, model, policy)
        if receipt['status'] == 'applied' and not np.array_equal(rows[f'{name}_{arm}_open'], portable.prediction):
            raise ValueError('Raw and deployed open action selection differ')
        rows[f'{name}_{arm}_proposal'] = portable.candidate
        rows[f'{name}_{arm}_delta'] = portable.expected_delta
for stage in STAGES:
    left, right = [loaded[name][0][stage] for name in RUNS]
    delta = right - left
    rows[f'{stage}_score_drift'] = np.abs(delta - delta.mean(axis=1, keepdims=True)).max(1)
    rows[f'{stage}_abs_gap_change'] = (rows[f'rerun_{stage}_gap'] - rows[f'original_{stage}_gap']).abs()

groups = {'all': rows, 'c2': rows.loc[rows.cohort.eq('evaluation')],
          'stress': rows.loc[rows.cohort.eq('reconstructed_stress')]}
panels = {f'{name}_{length}': summarize(part) for name, group in groups.items()
          for length, part in (('full', group), ('short', group.loc[group.n_words.lt(100)]))}
manifest = json.loads((folders['rerun'] / 'probe_manifest.json').read_text())
report = dict(rows=len(rows), body_and_labels_verified=True,
              ordered_ids_sha256=manifest['ordered_ids_sha256'],
              ordered_body_hashes_sha256=hashlib.sha256(('\\n'.join(manifest['input_body_sha256'])+'\\n').encode()).hexdigest(),
              native_receipts=native_receipts, thresholds={arm: model['threshold'] for arm, model in weights.items()},
              scope='Two genuine runtimes on the identical frozen public texts. No refit or simulated perturbation.', panels=panels)
output = Path('/kaggle/working/runtime_stability')
output.mkdir(parents=True, exist_ok=True)
rows.to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Compared deployed runtime stability on', len(rows), 'identical bodies', flush=True)
'''


def build(output=OUTPUT):
    setup = "from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
    modules = ('counterfactual_deployment', 'counterfactual_selection', 'counterfactual_native_guard',
               'glm_prefix_response_replay', 'member_evidence_transfer', 'member_evidence',
               'peer_context_report', 'sibling_odds_replay')
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<embedded {module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    truth_hash = hashlib.sha256(PUBLIC_TRUTH.read_bytes()).hexdigest()
    weights = {arm: hashlib.sha256((WEIGHTS / f'{arm}.json').read_bytes()).hexdigest() for arm in ARMS}
    payload = f'EXPECTED_TRUTH = {truth_hash!r}\nMODEL_HASHES = {weights!r}\nARMS = {ARMS!r}\n' + RUN
    metadata = dict(id='zhuating/counterfactual-runtime-stability', title='Counterfactual Runtime Stability',
                    code_file='runtime-stability.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=PINS)
    notebook = dict(cells=[cell(setup), cell(payload)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
