"""Replay frozen reader evidence with the original and Claude-only decision scopes."""

import hashlib
import json

import numpy as np
import pandas as pd

from .peer_context_report import paired_stats, read_ids, read_panel, read_trace, same_ids, same_meta
from .sibling_odds_replay import replay_predictions


ARMS = ('length_matched', 'provenance_balanced')
WEIGHT = .15


def digest(value):
    return hashlib.sha256(value).hexdigest()


def verify_files(root, hashes):
    for file, expected in hashes.items():
        if digest((root / file).read_bytes()) != expected:
            raise ValueError(f'Pinned artifact bytes differ for {root / file}')


def read_kimi(panel, trace):
    query, truth = [read_ids(panel / f'{part}.parquet') for part in ('queries', 'truth')]
    same_ids(query.index, truth.index, 'Kimi query and truth', ordered=True)
    manifest = json.loads((trace / 'probe_manifest.json').read_text())
    ordered_ids = digest(('\n'.join(query.index) + '\n').encode())
    bodies = [digest(text.encode()) for text in query.text]
    if (manifest['rows'] != len(query) or manifest['ordered_ids_sha256'] != ordered_ids
            or manifest['input_body_sha256'] != bodies):
        raise ValueError('The frozen Kimi capture used different IDs or bodies')
    return truth.join(query).assign(cohort='kimi_flash')


def replay_panel(folder, truth, scores, archived, historical):
    trace, baseline = read_trace(folder, truth)
    labels = pd.Index(trace['labels'])
    rows = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(baseline=baseline)
    if not truth.text.str.split().str.len().eq(rows.n_words).all():
        raise ValueError('Body length disagrees with the frozen truth')
    same_meta(archived.loc[rows.index], rows, ['model', 'prompt_id', 'cohort', 'n_words'], 'Archived readers')
    if historical:
        if not archived.loc[rows.index, 'original'].eq(rows.baseline).all():
            raise ValueError('Historical zero-weight decisions differ from the reader audit')
    else:
        frozen = read_ids(folder / 'frozen_predictions.parquet')
        same_ids(frozen.index, rows.index, 'Frozen Sibling Odds', ordered=True)
        if not (np.array_equal(frozen.sibling_odds, baseline) and np.array_equal(frozen.dennis, trace['final'])):
            raise ValueError('Independent zero-weight decisions differ from the frozen trace')
    rows['glm_fable_protected'] = trace['glm_winner'] | trace['fable_guard']
    rows['original_typography_changed'] = rows.baseline.ne(replay_predictions(trace, truth.text, typography=False))
    eligible = (rows.baseline.str.startswith('claude-') & ~rows.glm_fable_protected
                & ~rows.original_typography_changed)
    for arm, matrix in scores.items():
        if not matrix.columns.equals(labels):
            raise ValueError(f'The reader and trace use different label order for {arm}')
        logits = matrix.loc[rows.index]
        rows[arm + '_standalone'] = logits.idxmax(axis=1)
        if not archived.loc[rows.index, arm].eq(rows[arm + '_standalone']).all():
            raise ValueError(f'Archived reader top-1 differs for {arm}')
        full = pd.Series(replay_predictions(trace, truth.text, external_logits=logits, weight=WEIGHT), index=rows.index)
        if historical and not archived.loc[rows.index, arm + '_blend'].eq(full).all():
            raise ValueError(f'Historical fixed-weight replay differs for {arm}')
        rows[arm + '_full'] = full
        allowed = eligible & full.str.startswith('claude-')
        rows[arm + '_claude'] = full.where(allowed, rows.baseline)
    return rows


def panel_report(panel, draws):
    fixed = ['baseline', *(f'{arm}_{scope}' for arm in ARMS for scope in ('full', 'claude'))]
    comparisons = [('baseline', move) for move in fixed[1:]]
    comparisons += [tuple(f'{arm}_{scope}' for arm in ARMS) for scope in ('standalone', 'full', 'claude')]
    report = dict(paired=[paired_stats(panel, before, after, draws) for before, after in comparisons], by_true_class={})
    for label, group in panel.groupby('model'):
        correct = group.baseline.eq(label)
        arms = {}
        for arm in fixed[1:]:
            accepted = group[arm].eq(label)
            arms[arm] = dict(correct=int(accepted.sum()), fixed=int((accepted & ~correct).sum()),
                             harmed=int((correct & ~accepted).sum()))
        report['by_true_class'][label] = dict(rows=len(group), original_correct=int(correct.sum()), arms=arms)
    return report


def run(historical, gen003, kimi, reader, inputs, output, draws=4000):
    hashes = json.loads((inputs / 'hashes.json').read_text())
    for key, folder in [('historical', historical), ('gen003', gen003), ('kimi', kimi),
                        ('reader', reader), ('embedded', inputs)]:
        verify_files(folder, hashes[key])
    public = read_ids(historical / 'public_truth.parquet')
    claude = read_panel(inputs / 'gen003').assign(cohort='gen003')
    frozen_csv = pd.read_csv(inputs / 'gen003_trace_input.csv', dtype=str, keep_default_na=False).set_index('id')
    same_ids(frozen_csv.index, claude.index, 'GEN003 trace input', ordered=True)
    if not frozen_csv.text.eq(claude.text).all():
        raise ValueError('GEN003 archived reader text differs from the input sent to its trace')
    pair = read_kimi(inputs / 'kimi_flash', kimi)
    if (len(public), len(claude), len(pair), int(claude.n_words.lt(100).sum())) != (798, 191, 191, 106):
        raise ValueError('A frozen cohort size differs')
    archived = read_ids(reader / 'reader_predictions.parquet')
    expected = public.index.append([claude.index, pair.index])
    standalone_official = archived.cohort.eq('official_external_only')
    if int(standalone_official.sum()) != 2699:
        raise ValueError('Expected the original official prefix for both reader matrices')
    same_ids(archived.loc[~standalone_official].index, expected, 'Frozen reader rows', ordered=True)
    scores = {arm: read_ids(reader / f'{arm}_logits.parquet') for arm in ARMS}
    for arm, matrix in scores.items():
        same_ids(matrix.index, archived.index, f'{arm} archived score order', ordered=True)
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f'Nonfinite reader logits for {arm}')
    rows = pd.concat([replay_panel(folder, truth, scores, archived, scope == 'historical')
                      for scope, folder, truth in (('historical', historical, public),
                                                   ('gen003', gen003, claude), ('kimi', kimi, pair))], verify_integrity=True)
    panel_names = {'c1': 'development', 'c2': 'evaluation', 'gen003': 'gen003', 'kimi_flash': 'kimi_flash'}
    report = dict(weight=WEIGHT, hashes=hashes, zero_matches_original=True,
                  scope='Claude-only requires both original and proposed final calls to be Claude and freezes original GLM/Fable and typography-changed rows.',
                  limitation='The Claude arm was specified after seeing standalone GEN003 results. Untouched Prvsiyan replication is still needed.',
                  panels={})
    for name, cohort in panel_names.items():
        panel = rows.loc[rows.cohort.eq(cohort)]
        report['panels'][name] = {length: panel_report(part, draws) for length, part in
                                  [('all', panel), ('short', panel.loc[panel.n_words.lt(100)])]}
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    columns = [f'{arm}_{scope}' for arm in ARMS for scope in ('full', 'claude')]
    rows.loc[rows[columns].ne(rows.baseline, axis=0).any(axis=1)].to_parquet(output / 'changed_rows.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
