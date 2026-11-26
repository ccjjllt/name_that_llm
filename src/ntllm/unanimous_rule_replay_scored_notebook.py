"""Compare exact scored Gemini dash restore with consensus and Kimi-only changes."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .unanimous_rule_replay_notebook import PINS as TRACE_PINS


OUTPUT = Path('runs/unanimous_rule_replay_20261125/kaggle_scored')
KERNEL = 'zhuating/ntl-unanimous-rule-scored'
PINS = (*TRACE_PINS, 'zhuating/ntl-kimi-flash-holdout-trace/1')


RUN = '''import json

import numpy as np
import pandas as pd

from ntllm.member_evidence_transfer import read_capture, read_prvsiyan
from ntllm.peer_context_report import paired_stats, read_ids, read_trace, same_ids
from ntllm.provenance_family_replay import read_kimi
from ntllm.unanimous_rule_replay import RULES, replay, typography_steps


def mount(slug, name):
    path, = [p for p in Path('/kaggle/input').rglob(name) if slug in p.parts]
    return path.parent


def compare(truth, trace, incumbent):
    rows = replay(truth, trace, incumbent)
    before = rows.after_fable_restore.to_numpy()
    dash = truth.text.str.contains('—', regex=False).to_numpy()
    # Exact rule in formally scored Sibling Odds Gemini Dash 58813845.
    restored = ((incumbent != before) & np.char.startswith(before.astype(str), 'gemini-') & dash)
    broad = np.where(restored, before, incumbent)
    kimi = rows.consensus_non_native & rows.consensus_label.eq('kimi-k3') & truth.text.str.contains(' — ', regex=False)
    only_kimi = typography_steps(trace, truth.text, before, {'spaced_dash': kimi.to_numpy()})[RULES[-1]]
    kimi_changed = only_kimi != incumbent
    narrow = np.where(rows.consensus_label.str.startswith('gemini-') & restored,
                      rows.unanimous_drop_em_dash, incumbent)
    if (kimi_changed & restored).any() or (kimi_changed & ~kimi).any() or np.any(only_kimi[kimi_changed] != 'kimi-k3'):
        raise ValueError('The independent Kimi spaced-dash arm touched an unrelated or scored Gemini row')
    if np.any((narrow != incumbent) & ((narrow != broad) | ~restored)):
        raise ValueError('The narrow Gemini treatment is not a strict subset of scored broad restoration')
    return rows[['cohort', 'model', 'prompt_id', 'n_words', 'strict', 'baseline']].assign(
        scored_broad=broad, narrow_gemini=narrow, kimi_eligible=kimi, kimi_spaced_only=only_kimi,
        broad_plus_kimi=np.where(kimi_changed, only_kimi, broad),
        narrow_plus_kimi=np.where(kimi_changed, only_kimi, narrow))


CONTRASTS = (('baseline', 'scored_broad'), ('baseline', 'narrow_gemini'), ('baseline', 'kimi_spaced_only'),
             ('scored_broad', 'narrow_gemini'), ('scored_broad', 'broad_plus_kimi'),
             ('scored_broad', 'narrow_plus_kimi'), ('narrow_gemini', 'narrow_plus_kimi'))


def describe(frame):
    return dict(rows=len(frame), prompts=int(frame.prompt_id.nunique()),
                classes=frame.model.value_counts().to_dict(), kimi_eligible=int(frame.kimi_eligible.sum()),
                by_true_class={label: dict(rows=len(part), eligible=int(part.kimi_eligible.sum()),
                    new_kimi_fixed=int((part.scored_broad.ne(label) & part.broad_plus_kimi.eq(label)).sum()),
                    new_kimi_harmed=int((part.scored_broad.eq(label) & part.broad_plus_kimi.ne(label)).sum()))
                    for label, part in frame.groupby('model')},
                paired=[paired_stats(frame, before, after, 3500) for before, after in CONTRASTS])


prior = mount('ntl-0719-public-decision-trace', 'public_truth.parquet')
source = mount('ntl-heldout-source-audit', 'candidate_truth.parquet')
public = read_ids(prior / 'public_truth.parquet')
prvsiyan = read_prvsiyan(source, Path('kimi/truth.parquet'))
historical_trace = read_capture(mount('ntl-sibling-odds-member-trace', 'member_scores.npz'), public)
independent_trace = read_capture(mount('ntl-prvsiyan-member-trace', 'member_scores.npz'), prvsiyan)
archive = mount('ntl-kimi-flash-holdout-trace', 'probe_manifest.json')
kimi = read_kimi(Path('kimi'), archive)
kimi_trace, kimi_incumbent = read_trace(archive, kimi)
saved = read_ids(archive / 'frozen_predictions.parquet')
same_ids(saved.index, kimi.index, 'Frozen Kimi / Flash decisions', ordered=True)
if len(kimi) != 191 or not np.array_equal(kimi_incumbent, saved.sibling_odds):
    raise ValueError('Archive does not reproduce the frozen 191 Kimi / Flash rows')
history = compare(public, *historical_trace[:2])
held = compare(prvsiyan, *independent_trace[:2])
focal = compare(kimi, kimi_trace, kimi_incumbent).join(kimi[['source_dataset']])
panels = dict(c2=history.loc[history.cohort.eq('evaluation')], prvsiyan_895=held,
              prvsiyan_316=held.loc[held.strict], kimi_flash_191=focal)
panels.update({f'kimi_source:{name}': sub for name, sub in focal.groupby('source_dataset')})
report = dict(previous_scored_submission=58813845, displayed_scored_baseline_and_patch=.720,
              design='Replay on existing same base with no new scoring, fitting or label-informed decisions.',
              panels={name: {subset: describe(rows) for subset, rows in
                             [('all', frame), ('short', frame.loc[frame.n_words.lt(100)])] if len(rows)}
                      for name, frame in panels.items()})
output = Path('/kaggle/working/unanimous_rule_scored')
output.mkdir(exist_ok=True)
pd.concat([panels['c2'], held, focal], verify_integrity=True).to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Verified previously scored dash rules and 191 exact held Kimi/Flash bodies', flush=True)
'''


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\nPath('kimi').mkdir(exist_ok=True)\n")
    modules = ('unanimous_rule_replay', 'member_evidence', 'member_evidence_transfer', 'peer_context_report',
               'provenance_family_replay', 'sibling_odds_replay')
    for name in modules:
        code = Path(__file__).with_name(f'{name}.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    for name in ('truth.parquet', 'queries.parquet'):
        encoded = base64.b64encode((Path('runs/kimi_flash_20261125/data/holdout') / name).read_bytes()).decode()
        setup += f"Path('kimi/{name}').write_bytes(base64.b64decode({encoded!r}))\n"
    for code in (setup, RUN):
        compile(code, '<scored typography replay>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Unanimous Rule Scored', code_file='unanimous-rule-scored.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(RUN)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
