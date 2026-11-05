"""Replay cached q8 replacement through explicit stages of the public recipe.

The source recipe is runs/public_scan_20261105/generated/infer.py. Its late
pathlib error makes the historical tail's execution uncertain. ``pre_tail``
and ``corrected_tail`` are explicit evaluation scopes, not claims that either
exactly reproduces hidden 0.668 execution. Missing stages stop the replay.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax


BRANCHES = {'plain': 2.0, 'centered': 0.70, 'gaussian': 0.44}
FAMILIES = ('claude', 'gemini', 'gpt', 'glm')
TYPO_MARKS = [' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':', '(', ')',
              '[', ']', '**', '`', '…', '•', '\n', '\n\n', 'e.g.', 'i.e.', 'However,',
              'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ',
              '\n- ', '\n1.', '###', '```', '> ']


def load_cohort(path):
    path = Path(path)
    return dict(rows=pd.read_parquet(path / 'rows.parquet'),
                manifest=json.loads((path / 'manifest.json').read_text()),
                branches=dict(np.load(path / 'branches.npz', allow_pickle=False)))


def load_cache(path):
    if path is None:
        return dict(metadata={}, arrays={})
    path = Path(path)
    metadata = json.loads(path.read_text())
    arrays_path = path.parent / metadata.get('arrays_file', path.with_suffix('.npz').name)
    return dict(metadata=metadata, arrays=dict(np.load(arrays_path, allow_pickle=False)))


def _probability(value, shape, name, normalize=True):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f'{name}: expected finite nonnegative probabilities with shape {shape}.')
    if not np.allclose(value.sum(-1), 1, rtol=1e-5, atol=1e-7):
        raise ValueError(f'{name}: probabilities do not sum to one.')
    return value / value.sum(-1, keepdims=True) if normalize else value


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite array with shape {shape}.')
    return value


def _mask(value, rows, name):
    value = np.asarray(value)
    if value.shape != (rows,) or value.dtype.kind != 'b':
        raise ValueError(f'{name}: expected a Boolean row mask.')
    return value


def _validate_cohorts(reference, candidate):
    if not isinstance(reference, dict):
        reference = load_cohort(reference)
    if not isinstance(candidate, dict):
        candidate = load_cohort(candidate)
    rows, other = reference['rows'], candidate['rows']
    first, second = reference['manifest'], candidate['manifest']
    labels = first['labels']
    if len(labels) != 18 or len(set(labels)) != 18 or labels != second['labels']:
        raise ValueError('Both q8 cohorts need the same ordered 18 labels.')
    columns = ['id', 'text', 'inferred_group', 'group_size']
    for column in ('prompt_id', 'model', 'true_model', 'fold', 'n_words'):
        if column in rows or column in other:
            if column not in rows or column not in other:
                raise ValueError(f'Q8 cohorts disagree on presence of evaluation metadata {column}.')
            columns.append(column)
    if not len(rows) or not rows.id.is_unique or not rows[columns].equals(other[columns]):
        raise ValueError('Q8 cohorts differ in ordered IDs, text, inferred groups or sizes.')
    if not first.get('cohort_signature') or first['cohort_signature'] != second.get('cohort_signature'):
        raise ValueError('Q8 cohort signatures are missing or differ.')
    if first.get('ids') != rows.id.astype(str).tolist() or second.get('ids') != other.id.astype(str).tolist():
        raise ValueError('Q8 manifest IDs do not match the rows.')
    if not rows.groupby('inferred_group').id.transform('size').equals(rows.group_size):
        raise ValueError('Recorded inferred group sizes disagree with the actual cohort.')
    if rows.group_size.lt(1).any() or rows.group_size.gt(18).any():
        raise ValueError('Q8 replay requires inferred groups of size 1..18.')
    for item, title in ((reference, 'reference'), (candidate, 'candidate')):
        available = item['manifest']['branch_availability']
        for branch in BRANCHES:
            values = _finite(item['branches'][branch + '_lp'], (len(rows), 18), f'{title}.{branch}')
            active = (np.ones(len(rows), dtype=bool) if branch == 'plain' else
                      _mask(item['branches'][branch + '_active'], len(rows), f'{title}.{branch}_active'))
            if branch != 'plain' and ((active & rows.group_size.lt(2).to_numpy()).any()
                                      or np.any(values[~active] != 0)):
                raise ValueError(f'{title}.{branch}: singleton/inactive contributions must be zero.')
            if branch != 'plain' and available.get(branch, False) and not np.array_equal(
                    active, rows.group_size.between(2, 18).to_numpy()):
                raise ValueError(f'{title}.{branch}: an available formal branch must cover every inferred group of size 2..18.')
            if not available.get(branch, False) and (active.any() if branch != 'plain' else np.any(values)):
                raise ValueError(f'{title}.{branch}: unavailable branch contains active predictions.')
            if available.get(branch, False) and active.any():
                # Adapter-averaged log probabilities can be unnormalized, but may
                # differ from normalized values only by a finite row constant.
                if (values[active] > 1e-5).any() or (logsumexp(values[active], axis=1) > 1e-5).any():
                    raise ValueError(f'{title}.{branch}: expected unweighted log probabilities, not raw logits.')
        if not available.get('plain', False):
            raise ValueError('A valid plain branch is required for q8 replay.')
    for branch in ('centered', 'gaussian'):
        both_available = all(item['manifest']['branch_availability'].get(branch, False)
                             for item in (reference, candidate))
        if both_available and not np.array_equal(reference['branches'][branch + '_active'], candidate['branches'][branch + '_active']):
            raise ValueError(f'Old/new {branch} activation masks differ; choose an explicit partial-branch experiment instead.')
    return reference, candidate, rows, labels


def probability_metrics(rows, labels, probability, include_slices=True):
    """Observed-class BA is labelled explicitly when a small cohort lacks classes."""
    truth_column = 'model' if 'model' in rows else 'true_model'
    if truth_column not in rows:
        return dict(rows=len(rows), labelled=False)
    truth = rows[truth_column].to_numpy()
    if not set(truth).issubset(labels):
        raise ValueError('Evaluation labels are outside the recorded class order.')
    prediction = np.asarray(labels)[probability.argmax(1)]
    targets = pd.Series(truth).map({name: i for i, name in enumerate(labels)}).to_numpy()
    correct = prediction == truth
    nll = -np.log(np.clip(probability[np.arange(len(rows)), targets], 1e-30, 1))
    recalls = []
    for label in labels:
        selected = truth == label
        recalls.append(dict(model=label, rows=int(selected.sum()),
                            recall=float(correct[selected].mean()) if selected.any() else None))
    observed = [r['recall'] for r in recalls if r['rows']]
    ba = float(np.mean(observed)) if observed else None
    result = dict(rows=len(rows), labelled=True, observed_classes=len(observed),
                balanced_accuracy=ba, balanced_accuracy_18=ba if len(observed) == 18 else None,
                ba_scope='mean recall over observed classes; 18-class BA only when all are present',
                nll=float(nll.mean()) if len(rows) else None, per_class=recalls)
    if include_slices:
        masks = {}
        if 'group_size' in rows:
            masks.update(singleton=rows.group_size.eq(1).to_numpy(),
                         grouped=rows.group_size.ge(2).to_numpy())
        if 'n_words' in rows:
            bins = pd.cut(rows.n_words, [-1, 59, 99, 159, np.inf], labels=['<60', '60-99', '100-159', '>=160'])
            masks.update({str(name): bins.eq(name).to_numpy() for name in bins.cat.categories})
        result['slices'] = {name: {key: value for key, value in probability_metrics(
            rows.loc[mask], labels, probability[mask], False).items() if key != 'per_class'}
                           for name, mask in masks.items() if mask.any()}
    return result


def paired_changes(rows, labels, reference, candidate):
    column = 'model' if 'model' in rows else 'true_model'
    if column not in rows:
        return dict(rows=len(rows), labelled=False)
    truth = rows[column].to_numpy()
    first = np.asarray(labels)[reference.argmax(1)]
    second = np.asarray(labels)[candidate.argmax(1)]
    a, b = first == truth, second == truth
    return dict(rows=len(rows), changed=int((first != second).sum()),
                rescued=int((~a & b).sum()), lost=int((a & ~b).sum()),
                delta_balanced_accuracy=probability_metrics(rows, labels, candidate)['balanced_accuracy']
                - probability_metrics(rows, labels, reference)['balanced_accuracy'])


def apply_typography(scores, evidence, length_gate, coverage_gate):
    probability = softmax(scores, axis=1)
    ordered = np.sort(probability, axis=1)
    confidence = np.clip((.60 - (ordered[:, -1] - ordered[:, -2])) / .60, 0, 1)
    gate = confidence * length_gate * coverage_gate
    return scores + gate[:, None] * evidence, gate


def apply_style(scores, labels, style_probabilities, reliabilities):
    out = np.asarray(scores, dtype=np.float64).copy()
    gates = {}
    for family in FAMILIES:
        cols = np.flatnonzero([label.startswith(family) for label in labels])
        # Source V40A clips tiny class probabilities without renormalizing them.
        style = _probability(style_probabilities[family], (len(out), len(cols)),
                             f'style_{family}_prob', normalize=False)
        reliability = float(reliabilities[family])
        if not np.isfinite(reliability) or not 0 <= reliability <= 1:
            raise ValueError('Style reliability must be finite in [0,1].')
        original = out[:, cols].copy()
        p = softmax(original, axis=1)
        ordered = np.sort(p, axis=1)
        gate = np.clip((.65 - (ordered[:, -1] - ordered[:, -2])) / .65, 0, 1)
        alpha = .10 * reliability * gate
        mixture = (1 - alpha[:, None]) * p + alpha[:, None] * style
        out[:, cols] = logsumexp(original, axis=1, keepdims=True) + np.log(np.maximum(mixture, 1e-12))
        gates[family] = alpha
    return out, gates


def apply_pairwise(scores, pair_probability):
    p = softmax(scores, axis=1)
    ordered = np.sort(p, axis=1)
    gate = np.clip((.35 - (ordered[:, -1] - ordered[:, -2])) / .35, 0, 1)
    log_pair = np.log(np.clip(pair_probability, 1e-8, None))
    log_pair -= log_pair.mean(axis=1, keepdims=True)
    return scores + .10 * gate[:, None] * log_pair, gate


def replay_q8(reference, candidate, cache, beta, tail='pre_tail', require_complete=False):
    """Return {'report': JSON-safe evidence, 'probabilities': stage arrays}.

    cache = {'metadata': {...}, 'arrays': {...}} or a JSON path whose adjacent
    NPZ holds arrays. Missing layers stop the continuous prefix. A completed
    pre_tail replay remains distinct from the complete corrected tail.
    """
    if not np.isfinite(beta) or not 0 <= beta <= 1 or tail not in ('pre_tail', 'corrected_tail'):
        raise ValueError('Use beta in [0,1] and an explicit pre_tail/corrected_tail scope.')
    reference, candidate, rows, labels = _validate_cohorts(reference, candidate)
    cache = load_cache(cache) if not isinstance(cache, dict) else cache
    metadata, arrays = cache.get('metadata', {}), cache.get('arrays', {})
    n = len(rows)
    missing = []
    for side, value in (('reference', reference), ('candidate', candidate)):
        for branch in BRANCHES:
            if not value['manifest']['branch_availability'].get(branch, False):
                missing.append(f'q8.{side}.{branch}')
    required_meta = ['ids', 'labels', 'cohort_signature', 'reference_M', 'system_kind',
                     'fit_scope', 'evaluation_scope', 'model_provenance']
    missing += ['metadata.' + key for key in required_meta if key not in metadata or metadata[key] in (None, '', {})]
    if metadata.get('fit_scope') not in (None, 'outer_train_only', 'independent_confirmation'):
        missing.append('metadata.legal_fit_scope_declaration')
    if metadata.get('evaluation_scope') not in (None, 'prompt_holdout', 'independent_confirmation'):
        missing.append('metadata.legal_evaluation_scope_declaration')
    if metadata.get('fit_prompt_ids') is not None and 'prompt_id' in rows:
        if set(map(str, metadata['fit_prompt_ids'])) & set(rows.prompt_id.astype(str)):
            raise ValueError('Cached fitted components include evaluation prompts.')
    required_arrays = ['other_lp', 'comp_prob', 'atlas_prob', 'atlas_available']
    if n >= 18:
        required_arrays.append('train_prior')
    if tail == 'corrected_tail':
        required_arrays += ['typo_evidence', 'typo_length_gate', 'typo_coverage_gate', 'pairwise_prob']
        required_arrays += ['style_' + family + '_prob' for family in FAMILIES]
        if 'style_reliability' not in metadata:
            missing.append('metadata.style_reliability')
    missing += ['arrays.' + name for name in required_arrays if name not in arrays]
    for key, expected in (('ids', rows.id.astype(str).tolist()), ('labels', labels),
                          ('cohort_signature', reference['manifest']['cohort_signature'])):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f'Cache {key} does not match this exact cohort.')
    report = dict(beta=float(beta), tail=tail, system_kind=metadata.get('system_kind', 'unspecified'),
                  missing=missing, stages={}, coverage={}, requested_scope_complete=False,
                  full_system_supported=False, historical_0668_execution_equivalence=False,
                  reference_M=None, cohort_signature=reference['manifest']['cohort_signature'],
                  q8_provenance={side: value['manifest'].get('provenance')
                                 for side, value in (('reference', reference), ('candidate', candidate))},
                  interpretation='Cache replay coverage is not proof of hidden-score gain or independently verified training provenance. pre_tail omits the uncertain historical tail; corrected_tail intentionally repairs its execution path.',
                  provenance={key: metadata.get(key) for key in ('fit_scope', 'evaluation_scope', 'model_provenance')})
    probabilities = {}

    def stage(name, p):
        probabilities[name] = np.array(p, copy=True)
        report['stages'][name] = probability_metrics(rows, labels, p)

    def finish():
        report['missing'] = sorted(set(report['missing']))
        if require_complete and not report['requested_scope_complete']:
            raise ValueError('Cannot replay requested scope: ' + ', '.join(report['missing']))
        return dict(report=report, probabilities=probabilities)

    common_branches = [branch for branch in BRANCHES if all(
        value['manifest']['branch_availability'].get(branch, False) for value in (reference, candidate))]
    block = sum(BRANCHES[name] * ((1 - beta) * reference['branches'][name + '_lp']
                               + beta * candidate['branches'][name + '_lp']) for name in common_branches)
    stage('q8_available_branches', softmax(block / 2, axis=1))
    report['q8_available_branches'] = common_branches
    if len(common_branches) != 3 or any(key not in metadata for key in ('ids', 'labels', 'cohort_signature', 'reference_M')) or 'other_lp' not in arrays:
        return finish()
    other_lp = np.asarray(arrays['other_lp'], dtype=np.float64)
    if other_lp.ndim != 3 or other_lp.shape[1:] != (n, 18) or not np.isfinite(other_lp).all():
        raise ValueError('other_lp must be the actual weighted [branches,rows,labels] upstream arrays.')
    reference_count = sum(np.any(np.abs(values) > 0) for values in other_lp)
    reference_count += sum(np.any(np.abs(reference['branches'][name + '_lp'] * weight) > 0)
                           for name, weight in BRANCHES.items())
    if metadata['reference_M'] != int(reference_count):
        raise ValueError('reference_M disagrees with the original nonzero LP branch count.')
    report['reference_M'] = int(reference_count)
    report['other_weighted_branches'] = len(other_lp)
    p = softmax((other_lp.sum(axis=0) + block) / max(1, reference_count), axis=1)
    stage('main', p)
    if 'comp_prob' not in arrays:
        return finish()
    comp = _probability(arrays['comp_prob'], (n, 18), 'comp_prob')
    p = .85 * p + .15 * comp
    p /= p.sum(axis=1, keepdims=True)
    stage('complementary', p)
    ordered = np.sort(p, axis=1)
    candidates = ordered[:, -1] - ordered[:, -2] < .32 if n >= 50 else np.ones(n, dtype=bool)
    report['coverage']['atlas_candidates'] = int(candidates.sum())
    if 'atlas_prob' not in arrays or 'atlas_available' not in arrays:
        return finish()
    available = _mask(arrays['atlas_available'], n, 'atlas_available')
    atlas = np.array(arrays['atlas_prob'], dtype=np.float64, copy=True)
    if atlas.shape != (n, 18):
        raise ValueError('atlas_prob must match the cohort and labels, including unavailable rows.')
    if available.any():
        atlas[available] = _probability(atlas[available], (int(available.sum()), 18), 'available ATLAS probabilities')
    report['coverage'].update(atlas_available_candidates=int((candidates & available).sum()),
                              atlas_missing_candidates=int((candidates & ~available).sum()))
    if (candidates & ~available).any():
        report['missing'].append('atlas_scores_for_current_candidates')
        report['coverage']['atlas_missing_ids'] = rows.loc[candidates & ~available, 'id'].astype(str).tolist()
        return finish()
    p[candidates] = .80 * p[candidates] + .20 * atlas[candidates]
    p /= p.sum(axis=1, keepdims=True)
    stage('atlas', p)
    if n >= 18:
        if 'train_prior' not in arrays:
            return finish()
        prior = _probability(np.asarray(arrays['train_prior'])[None], (1, 18), 'train_prior')[0]
        predicted = np.clip(p.mean(axis=0), 1e-6, None); predicted /= predicted.sum()
        target = .65 * predicted + .35 * prior
        target = np.clip(target, 1e-6, None); target /= target.sum()
        q = p.copy()
        for _ in range(100):
            q *= (n * target[None]) / np.clip(q.sum(axis=0, keepdims=True), 1e-12, None)
            q /= np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
        scores = .70 * np.log(p + 1e-12) + .30 * np.log(q + 1e-12)
    else:
        scores = np.log(p + 1e-12)
    stage('soft_marginal', softmax(scores, axis=1))
    if tail == 'corrected_tail':
        for name in ('typo_evidence', 'typo_length_gate', 'typo_coverage_gate'):
            if name not in arrays:
                return finish()
        evidence = _finite(arrays['typo_evidence'], (n, 18), 'typo_evidence')
        length = _finite(arrays['typo_length_gate'], (n,), 'typo_length_gate')
        coverage = _finite(arrays['typo_coverage_gate'], (n,), 'typo_coverage_gate')
        if (np.abs(evidence) > .3000001).any() or any(((value < 0) | (value > 1)).any() for value in (length, coverage)):
            raise ValueError('Typography evidence or gates lie outside the source recipe bounds.')
        scores, gate = apply_typography(scores, evidence, length, coverage)
        report['coverage']['typography_active'] = int((gate > 0).sum())
        stage('typography', softmax(scores, axis=1))
        if 'style_reliability' not in metadata or any('style_' + name + '_prob' not in arrays for name in FAMILIES):
            return finish()
        if set(metadata['style_reliability']) != set(FAMILIES):
            raise ValueError('Style cache needs one reliability for each of the four families.')
        scores, gates = apply_style(scores, labels, {name: arrays['style_' + name + '_prob'] for name in FAMILIES}, metadata['style_reliability'])
        report['coverage']['style_active'] = {name: int((gate > 0).sum()) for name, gate in gates.items()}
        stage('family_style', softmax(scores, axis=1))
        if 'pairwise_prob' not in arrays:
            return finish()
        pair = _probability(arrays['pairwise_prob'], (n, 18), 'pairwise_prob')
        scores, gate = apply_pairwise(scores, pair)
        report['coverage']['pairwise_active'] = int((gate > 0).sum())
        stage('pairwise', softmax(scores, axis=1))
    report['requested_scope_complete'] = not report['missing']
    report['full_system_supported'] = report['requested_scope_complete'] and tail == 'corrected_tail'
    return finish()


def fit_typography_cache(train, query, labels):
    """Fit source typography evidence on the caller's legal training side only."""
    import re
    if len(labels) != 18 or not set(train.model).issubset(labels):
        raise ValueError('Typography labels do not match the 18-class order.')
    if 'prompt_id' in query and set(train.prompt_id) & set(query.prompt_id):
        raise ValueError('Typography fitting and evaluation prompts overlap.')
    # Preserve source dtypes: clipping float32 global rates can leave tiny
    # positive marker signals, which participate in its positive-signal median.
    x = np.asarray([[float(mark in str(text)) for mark in TYPO_MARKS] for text in train.text], dtype=np.float32)
    q = np.asarray([[float(mark in str(text)) for mark in TYPO_MARKS] for text in query.text], dtype=np.float32)
    y = train.model.map({label: i for i, label in enumerate(labels)}).to_numpy()
    counts = np.bincount(y, minlength=18).astype(float)
    hits = np.stack([x[y == index].sum(axis=0) for index in range(18)]).astype(np.float64)
    global_rate = (x.sum(axis=0) + 2) / (len(x) + 4)
    rate = np.clip((hits + 24 * global_rate[None]) / (counts[:, None] + 24), .01, .99)
    global_rate = np.clip(global_rate, .01, .99)
    signal = np.sqrt(np.mean((rate - global_rate[None]) ** 2, axis=0))
    denominator = float(np.median(signal[signal > 0])) if (signal > 0).any() else 1.0
    feature_weights = np.clip(signal / max(denominator, 1e-6), .25, 2)
    evidence = (q * feature_weights) @ np.log(rate).T + ((1 - q) * feature_weights) @ np.log1p(-rate).T
    evidence -= (((q * feature_weights) @ np.log(global_rate)) + (((1 - q) * feature_weights) @ np.log1p(-global_rate)))[:, None]
    evidence -= evidence.mean(axis=1, keepdims=True)
    evidence = np.clip(1.35 * evidence / max(float(feature_weights.sum()), 1e-6), -.30, .30)
    words = np.asarray([len(re.findall(r'\b\w+\b', str(text))) for text in query.text], dtype=np.float32)
    prior = counts + 1; prior /= prior.sum()
    return dict(typo_evidence=evidence,
                typo_length_gate=np.clip(np.sqrt(np.maximum(words, 1) / 80), .40, 1),
                typo_coverage_gate=np.clip(np.sqrt((q.sum(axis=1) + 1) / 4), .45, 1),
                train_prior=prior)



# Feature definition retained from the public sibling_style.py recipe.
import re
import numpy as np
from functools import lru_cache
MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE=re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE=re.compile(r'(?<=[.?!])\s+')
@lru_cache(maxsize=15000)
def _style_row(t):
    t = str(t)
    s = WRE.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\n' for c in t)/nt]
    base += [min(t.count(m)/n,3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)
def _style_features(texts):
    return np.stack([_style_row(str(t)) for t in texts])



def fit_style_cache(train, query, labels):
    """Fit source V40A family models and reliability on legal training prompts.

    Returns cache arrays plus metadata.style_reliability. No query labels are
    read. Reliability uses five prompt-disjoint internal training folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(labels) != 18 or set(train.model) != set(labels):
        raise ValueError('Style fitting requires all 18 classes in the recorded order.')
    if 'prompt_id' in query and set(train.prompt_id) & set(query.prompt_id):
        raise ValueError('Style fitting and evaluation prompts overlap.')
    x, q = _style_features(train.text), _style_features(query.text)
    y = train.model.map({name: i for i, name in enumerate(labels)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    arrays, reliability, diagnostics = {}, {}, {}
    for family, strength in [('claude', .003), ('gemini', .003), ('gpt', .003), ('glm', .03)]:
        cols = np.flatnonzero([name.startswith(family) for name in labels])
        selected = np.flatnonzero(np.isin(y, cols))
        target, groups = np.searchsorted(cols, y[selected]), prompts[selected]
        if len(np.unique(groups)) < 5:
            raise ValueError('V40A reliability needs at least five training prompts per family.')
        prediction = np.empty(len(selected), dtype=int)
        def classifier():
            return make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=400, class_weight='balanced'))
        for fit, valid in GroupKFold(5).split(x[selected], target, groups):
            model = classifier().fit(x[selected][fit], target[fit])
            prediction[valid] = model.predict(x[selected][valid])
        ba = float(balanced_accuracy_score(target, prediction))
        reliability[family] = float(np.clip((ba - 1 / len(cols)) / .15, 0, 1))
        if reliability[family] > 0:
            model = classifier().fit(x[selected], target)
            probability = np.maximum(model.predict_proba(q), 1e-8)
        else:
            # The source skips this family's residual entirely when alpha=0.
            probability = np.full((len(query), len(cols)), 1 / len(cols))
        arrays['style_' + family + '_prob'] = probability
        diagnostics[family] = dict(oof_balanced_accuracy=ba, reliability=reliability[family],
                                   residual_enabled=reliability[family] > 0)
    return dict(arrays=arrays, metadata={'style_reliability': reliability}, diagnostics=diagnostics)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--beta', nargs='+', type=float, default=[0, .25, 1])
    parser.add_argument('--tail', choices=['pre_tail', 'corrected_tail'], default='pre_tail')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    args = parser.parse_args()
    reference, candidate, cache = load_cohort(args.reference), load_cohort(args.candidate), load_cache(args.cache)
    fractions = list(dict.fromkeys([0.0, *args.beta]))
    results = {beta: replay_q8(reference, candidate, cache, beta, args.tail, args.require_complete) for beta in fractions}
    rows, labels = reference['rows'], reference['manifest']['labels']
    reports, arrays = {}, {}
    predictions = rows[['id', 'inferred_group', 'group_size']].copy()
    for beta, result in results.items():
        key = str(beta).replace('.', 'p')
        report = result['report']
        report['paired_vs_beta0'] = {}
        for name, probability in result['probabilities'].items():
            arrays[f'beta_{key}__{name}'] = probability
            predictions[f'beta_{key}__{name}'] = np.asarray(labels)[probability.argmax(1)]
            if name in results[0.0]['probabilities']:
                report['paired_vs_beta0'][name] = paired_changes(rows, labels, results[0.0]['probabilities'][name], probability)
        reports[str(beta)] = report
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / f'{stamp}_q8_replay_{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'replay.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output / 'stage_probabilities.npz', **arrays)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    print(json.dumps({str(beta): {'last_stage': list(result['probabilities'])[-1],
                                 'full_system_supported': result['report']['full_system_supported'],
                                 'missing': result['report']['missing']} for beta, result in results.items()}, ensure_ascii=False))
    print(output.resolve())


if __name__ == '__main__':
    main()
