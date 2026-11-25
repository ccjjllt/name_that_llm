"""Package the fixed-weight, same-base Gemini assignment loss replay."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/pair_preserving_20261125/kaggle_transfer')
PINS = ['zhuating/ntl-exact-pair-loss-readers/1', 'zhuating/ntl-prvsiyan-member-trace/1',
        'zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-heldout-source-audit/2']


RUN = '''import json
import re

import numpy as np
import pandas as pd

from ntllm.glm_prefix_response_replay import restored_route
from ntllm.member_evidence import STAGES
from ntllm.member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan
from ntllm.peer_context_report import paired_stats, read_ids, same_ids, same_meta
from ntllm.sibling_odds_replay import replay_predictions

ARMS = ('plain_ce', 'conditional_pair')
TARGETS = ('gemini-3.7-flash', 'gemini-3.8-flash')
WEIGHT = .15
DRAWS = 3000


def asset(slug, name):
    matches = [p for p in Path('/kaggle/input').rglob(name) if slug in p.parts]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} from {slug}, found {matches}')
    return matches[0]


def training_receipt(folder):
    raw = {name: json.loads((folder / f'{name}.json').read_text()) for name in ARMS}
    same = ('schedule_sha256', 'scheduled_batches', 'row_exposures', 'pair_exposures', 'seed', 'train_rows', 'train_pairs')
    if any(raw[ARMS[0]][key] != raw[ARMS[1]][key] for key in same):
        raise ValueError('The two training arms used different frozen schedules or exposures')
    full = (5566, 89044, 1882)
    per_epoch = (2783, 44522, 941)
    for arm, weight in zip(ARMS, (0., .40), strict=True):
        receipt = raw[arm]
        sizes = ('scheduled_batches', 'row_exposures', 'pair_exposures')
        if (not re.fullmatch(r'[0-9a-f]{64}', receipt['schedule_sha256'])
                or tuple(receipt[key] for key in sizes) != full or receipt['pair_weight'] != weight
                or (receipt['train_rows'], receipt['train_pairs']) != per_epoch[1:] or len(receipt['epoch_logs']) != 2):
            raise ValueError(f'The fixed plain-versus-paired design changed for {arm}')
        for index, epoch in enumerate(receipt['epoch_logs'], 1):
            if (epoch['epoch'] != index or tuple(epoch[key] for key in sizes) != per_epoch
                    or epoch['optimizer_steps'] + epoch['skipped'] != epoch['scheduled_batches']):
                raise ValueError(f'Invalid batch or optimizer accounting for {arm}')
        for key in ('optimizer_steps', 'skipped'):
            if receipt[key] != sum(epoch[key] for epoch in receipt['epoch_logs']):
                raise ValueError(f'Invalid total {key} for {arm}')
    data = json.loads((folder / 'data_report.json').read_text())
    if (data['sampling']['rows'], data['sampling']['verified_pairs']) != per_epoch[1:]:
        raise ValueError('Trainer data manifest disagrees with the recorded exposures')
    return dict(arms=raw, ordered_training_ids_sha256=data['sampling']['ordered_ids_sha256'],
                archived_evaluation_counts=data['evaluation'])


def replay_panel(truth, folder, archived, logits, strict=False):
    trace, baseline, *_ = read_capture(folder, truth)
    labels = np.asarray(trace['labels'])
    if any(list(scores.columns) != labels.tolist() for scores in logits.values()):
        raise ValueError('Reader logit classes differ from the exact base')
    if not np.array_equal(replay_predictions(trace, truth.text, weight=0), baseline):
        raise ValueError('A zero-weight replay changed an original call')
    training_subset = archived.loc[archived.index.isin(truth.index)]
    same_ids(training_subset.index, truth.index, 'Training query output and original capture', ordered=True)
    same_meta(training_subset, truth, ['model', 'prompt_id', 'cohort', 'n_words'], 'Training query metadata')
    strict_flags = truth.strict_topic_holdout if strict else pd.Series(False, index=truth.index)
    if not np.array_equal(training_subset.strict_source_topic, strict_flags):
        raise ValueError('The frozen 316-row strict topic split changed')
    rows = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(
        baseline=baseline, strict_topic_holdout=strict_flags,
        source_dataset=truth.source_dataset if strict else 'historical_public',
        provider_route=truth.route if strict else 'not_recorded')
    receipt = json.loads((folder / 'glm_report.json').read_text())
    routed, _ = restored_route(trace, truth.text, baseline, receipt)
    rows['native_scored'] = routed
    protected = trace['glm_winner'] | trace['fable_guard']
    before_typography = replay_predictions(trace, truth.text, typography=False)
    rows['gemini_incumbent'] = rows.baseline.isin(TARGETS)
    rows['gemini_eligible'] = rows.gemini_incumbent & ~protected & rows.baseline.eq(before_typography)
    rows['native_or_fable_protected'] = protected
    rows['original_typography_change'] = rows.baseline.ne(before_typography)
    target = rows.model.to_numpy()[:, None]
    base_scores = np.where(protected[:, None], trace['glm_logp'], trace['sibling_logp'])

    def top3(scores):
        rank = np.argsort(-scores, axis=1, kind='stable')[:, :3]
        choices = labels[rank]
        return choices, (choices == target).any(axis=1)

    _, rows['baseline_top3'] = top3(base_scores)
    rows['baseline_stage_union'] = np.logical_or.reduce([top3(trace[stage])[1] for stage in STAGES])
    for arm in ARMS:
        score = logits[arm].loc[rows.index]
        choices, rows[f'{arm}_true_top3'] = top3(score.to_numpy())
        rows[f'{arm}_candidates'] = [list(values) for values in choices]
        rows[arm] = choices[:, 0]
        if not training_subset[arm].eq(rows[arm]).all():
            raise ValueError(f'{arm} logit top-1 differs from the archived score output')
        full = pd.Series(replay_predictions(trace, truth.text, external_logits=score, weight=WEIGHT), index=rows.index)
        rows[f'{arm}_full'] = full
        rows[f'{arm}_gemini_only'] = full.where(rows.gemini_eligible & full.isin(TARGETS), rows.baseline)
    return rows


def coverage(rows, focus):
    selected = rows.loc[focus]
    length = selected.plain_ce_true_top3
    paired = selected.conditional_pair_true_top3
    return dict(rows=len(selected), top3=dict(plain_ce=int(length.sum()), conditional_pair=int(paired.sum())),
                union=int((length | paired).sum()), conditional_pair_only=int((paired & ~length).sum()),
                plain_ce_only=int((length & ~paired).sum()))


def summarize(panel, by_model=True):
    wrong = panel.baseline.ne(panel.model)
    both = ['plain_ce_full', 'conditional_pair_full', 'plain_ce_gemini_only', 'conditional_pair_gemini_only']
    contrasts = [('baseline', key) for key in both]
    contrasts += [('plain_ce', 'conditional_pair'), ('plain_ce_full', 'conditional_pair_full'),
                  ('plain_ce_gemini_only', 'conditional_pair_gemini_only')]
    metrics = dict(rows=len(panel), prompts=int(panel.prompt_id.nunique()), true_classes=int(panel.model.nunique()),
                   native_scored=int(panel.native_scored.sum()), gemini_incumbents=int(panel.gemini_incumbent.sum()),
                   gemini_eligible=int(panel.gemini_eligible.sum()), paired=[paired_stats(panel, a, b, DRAWS) for a,b in contrasts],
                   candidates=dict(all=coverage(panel, panel.baseline.notna()), baseline_wrong=coverage(panel, wrong),
                                   outside_original_top3=coverage(panel, wrong & ~panel.baseline_top3),
                                   outside_all_stage_top3=coverage(panel, wrong & ~panel.baseline_stage_union)))
    if by_model:
        metrics['by_true_class'] = {}
        for label, group in panel.groupby('model'):
            correct = group.baseline.eq(label)
            metrics['by_true_class'][label] = dict(rows=len(group), original_correct=int(correct.sum()),
                arms={name: dict(fixed=int((group[name].eq(label) & ~correct).sum()),
                                 harmed=int((group[name].ne(label) & correct).sum())) for name in both},
                top3=coverage(group, group.baseline.notna()))
    return metrics


train = asset('ntl-exact-pair-loss-readers', 'predictions.parquet').parent
historical = asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent
rerun = asset('ntl-sibling-odds-member-trace', 'probe_manifest.json').parent
held_base = asset('ntl-prvsiyan-member-trace', 'probe_manifest.json').parent
source = asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent
schedule = training_receipt(train)
archived = read_ids(train / 'predictions.parquet')
if archived.cohort.value_counts().to_dict() != schedule['archived_evaluation_counts']:
    raise ValueError('Score output cohort counts disagree with the training query manifest')
logits = {arm: read_ids(train / f'{arm}_logits.parquet') for arm in ARMS}
for name, matrix in logits.items():
    same_ids(matrix.index, archived.index, f'{name} archived evaluation', ordered=True)
    if not matrix.columns.is_unique or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(f'{name} has invalid or nonfinite candidate columns')
public = read_ids(historical / 'public_truth.parquet')
held = read_prvsiyan(source, Path('kimi_panel_index.parquet'))
if len(public) != 798 or public.cohort.eq('evaluation').sum() != 300:
    raise ValueError('The canonical historical and C2 sizes changed')
old_rows = replay_panel(public, rerun, archived, logits)
new_rows = replay_panel(held, held_base, archived, logits, strict=True)
exposure = json.loads((held_base / 'body_overlap.json').read_text())
known = {key for name in EXPOSURE_KEYS for key in exposure.get(name, [])}
if not known.issubset(held.index):
    raise ValueError('The base corpus audit mentions unknown source IDs')
old_rows['known_runtime_overlap'] = False
new_rows['known_runtime_overlap'] = new_rows.index.isin(known)
rows = pd.concat([old_rows, new_rows], verify_integrity=True)
scopes = dict(all=rows, historical=old_rows, c2=old_rows.loc[old_rows.cohort.eq('evaluation')],
              reconstructed_stress=old_rows.loc[old_rows.cohort.eq('reconstructed_stress')],
              prvsiyan_895=new_rows, prvsiyan_316=new_rows.loc[new_rows.strict_topic_holdout])
if known:
    for name in ('prvsiyan_895', 'prvsiyan_316'):
        scopes[name + '_runtime_overlap_removed'] = scopes[name].loc[~scopes[name].known_runtime_overlap]
report = dict(weight=WEIGHT, schedule=schedule, zero_replays_exact=True, runtime_exposure=exposure,
              query_verification='Base captures verify exact ordered body hashes. Both trainer matrices match all archived query IDs and label, prompt, cohort and length metadata on the pinned canonical sources.',
              gemini_only='Original and full proposed calls must be Gemini 3.7/3.8. Keep original GLM/Fable and typography transitions.',
              panels={}, by_provider_route={})
for name, group in scopes.items():
    report['panels'][name] = {}
    for length, panel in [('all', group), ('short', group.loc[group.n_words.lt(100)])]:
        pieces = [('all_truth', panel), ('gemini_37_38_truth', panel.loc[panel.model.isin(TARGETS)]),
                  ('gemini_37_38_incumbent', panel.loc[panel.gemini_incumbent])]
        report['panels'][name][length] = {focus: summarize(part) for focus, part in pieces if len(part)}
for route, group in new_rows.groupby('provider_route'):
    report['by_provider_route'][route] = {length: summarize(part, by_model=False)
                                         for length, part in [('all', group), ('short', group.loc[group.n_words.lt(100)])]
                                         if len(part)}
output = Path('/kaggle/working/pair_loss_transfer')
output.mkdir(parents=True, exist_ok=True)
rows.to_parquet(output / 'predictions.parquet')
columns = [f'{arm}_{mode}' for arm in ARMS for mode in ('full', 'gemini_only')]
rows.loc[rows[columns].ne(rows.baseline, axis=0).any(axis=1)].to_parquet(output / 'changed_rows.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Verified exact schedule and completed paired replays:', len(rows), flush=True)
'''


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for module in ('member_evidence_transfer', 'member_evidence', 'glm_prefix_response_replay',
                   'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<embedded {module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    metadata = dict(id='zhuating/ntl-pair-loss-transfer', title='NTL Pair Loss Transfer',
                    code_file='pair-loss-transfer.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=PINS)
    notebook = dict(cells=[cell(setup), cell(RUN)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
