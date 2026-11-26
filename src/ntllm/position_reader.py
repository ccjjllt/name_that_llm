"""Attribution from equal-word windows with a frozen external training sample."""

import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace
from .sibling_odds_replay import replay_predictions


WINDOW = 48
VIEWS = ('first', 'middle', 'last', 'full')
WEIGHTS = (0., .05, .10, .20)


def word_window(text, view):
    words = list(re.finditer(r'\S+', text))
    if view == 'full' or len(words) <= WINDOW:
        return text
    excess = len(words) - WINDOW
    start = {'first': 0, 'middle': excess // 2, 'last': excess}[view]
    lo = words[start - 1].end() if start else 0
    end = start + WINDOW
    hi = words[end].start() if end < len(words) else len(text)
    return text[lo:hi]


def runs(training_path, public_rows, public_capture, held_rows, held_capture, output):
    sample_report = json.loads(training_path.with_name('report.json').read_text())
    training = read_ids(training_path)
    ids = hashlib.sha256(('\n'.join(training.index.astype(str)) + '\n').encode()).hexdigest()
    if len(training) != 44522 or ids != sample_report['ordered_train_ids_sha256']:
        raise ValueError('The fixed external sample changed')
    truth = pd.concat([public_rows.assign(strict_topic_holdout=False), held_rows], verify_integrity=True)
    if training.text.isna().any() or set(training.text) & set(truth.text):
        raise ValueError('The window reader requires valid bodies disjoint from evaluation')
    public_trace, public_base = read_trace(public_capture, public_rows)
    held_trace, held_base = read_trace(held_capture, held_rows)
    labels = np.asarray(public_trace['labels'])
    if not np.array_equal(labels, held_trace['labels']) or set(training.model) != set(labels):
        raise ValueError('Both frozen traces and the training sample must cover the same 18 labels')
    code = pd.Index(labels).get_indexer(truth.model)
    stage_hits = []
    for trace in (public_trace, held_trace):
        top = [np.argsort(-trace[stage], axis=1, kind='stable')[:, :3]
               for stage in ('stack_logp', 'glm_logp', 'sibling_logp')]
        stage_hits.append(np.concatenate(top, axis=1))
    rows = truth[['model', 'prompt_id', 'n_words', 'cohort', 'strict_topic_holdout']].assign(
        baseline=np.r_[public_base, held_base], stage_top3=(np.vstack(stage_hits) == code[:, None]).any(axis=1))
    focus = {'c2': rows.cohort.eq('evaluation'), 'independent': rows.cohort.eq('prvsiyan'),
             'independent_strict': rows.strict_topic_holdout}
    c1 = rows.cohort.eq('development').to_numpy()
    if (int(c1.sum()), *(int(mask.sum()) for mask in focus.values())) != (180, 300, 895, 316):
        raise ValueError('Expected the original development and evaluation cohorts')

    report = dict(window_words=WINDOW, training_rows=len(training), training_ids_sha256=ids, arms={})
    output.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        vector = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=8, max_features=85000,
                                sublinear_tf=True, lowercase=False, dtype=np.float32)
        x = vector.fit_transform(training.text.map(lambda text: word_window(text, view)))
        model = LinearSVC(C=1., dual=True, max_iter=1500, class_weight='balanced', random_state=2411)
        model.fit(x, training.model)
        del x
        scores = model.decision_function(vector.transform(truth.text.map(lambda text: word_window(text, view))))
        margins = pd.DataFrame(log_softmax(scores, axis=1), columns=model.classes_, index=truth.index)[labels]
        margins.to_parquet(output / f'{view}_logp.parquet')
        rows[view + '_single'] = margins.idxmax(axis=1)
        reader_top = np.argsort(-margins.to_numpy(), axis=1, kind='stable')[:, :3]
        rows[view + '_top3'] = (reader_top == code[:, None]).any(axis=1)
        fused = {}
        for weight in WEIGHTS:
            fused[weight] = np.r_[replay_predictions(public_trace, public_rows.text, external_logits=margins, weight=weight),
                                  replay_predictions(held_trace, held_rows.text, external_logits=margins, weight=weight)]
            # Use the same Fable protection for every view after its failure on previous transfer.
            fused[weight] = np.where(rows.baseline.eq('claude-fable-5'), rows.baseline, fused[weight])
        if not np.array_equal(fused[0.], rows.baseline):
            raise ValueError('The weight-zero control did not reproduce the base')
        dev = {weight: balanced_accuracy(rows.model[c1], predicted[c1]) for weight, predicted in fused.items()}
        selected = max(WEIGHTS, key=dev.get)
        rows[view] = fused[selected]
        arm = report['arms'][view] = dict(c1_weight=selected, c1_ba=dev, features=len(model.coef_[0]), scopes={})
        for name, mask in focus.items():
            for length, piece in (('all', rows.loc[mask]), ('short', rows.loc[mask & rows.n_words.lt(100)])):
                misses = piece.baseline.ne(piece.model) & ~piece.stage_top3
                arm['scopes'][f'{name}__{length}'] = dict(
                    single_ba=balanced_accuracy(piece.model, piece[view + '_single']),
                    new_top3=int((misses & piece[view + '_top3']).sum()), stage_misses=int(misses.sum()),
                    distinct_window=int(piece.n_words.gt(WINDOW).sum()),
                    matched=paired_stats(piece, 'baseline', view, 4000),
                )
        print(f'Completed {view}, C1 weight {selected}', flush=True)

    report['last_vs_first'] = {f'{name}__{length}': paired_stats(piece, 'first', 'last', 4000)
                               for name, mask in focus.items()
                               for length, piece in (('all', rows.loc[mask]),
                                                     ('short', rows.loc[mask & rows.n_words.lt(100)]))}
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
