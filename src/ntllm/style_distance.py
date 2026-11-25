"""Test a content-independent style encoder against its unadapted backbone."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids
from .provenance_readers import curate
from .sibling_odds_replay import replay_predictions


ENCODERS = ('roberta', 'style')
HELD_SOURCE = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
PRIMARY_BLEND = .15


def prepare_inputs(official, public, audit, pool):
    train = pd.read_csv(official, dtype={'id': str, 'prompt_id': str}).set_index('id')
    train = train.assign(cohort='official')
    benchmark = read_ids(public / 'public_truth.parquet')
    all_truth = read_ids(audit / 'candidate_truth.parquet')
    all_queries = read_ids(audit / 'candidate_queries.parquet')
    same_ids(all_queries.index, all_truth.index, 'Source audit queries and labels', ordered=True)
    held = all_truth[all_truth.source_dataset.eq(HELD_SOURCE)].join(all_queries[['text']])
    held = held.assign(cohort='prvsiyan_clean')
    labels = sorted(train.model.unique())
    if len(held) != 895 or sorted(held.model.unique()) != labels or len(labels) != 18:
        raise ValueError('Expected the fixed Prvsiyan holdout and all 18 classes')
    columns = ['text', 'model', 'prompt_id', 'cohort']
    queries = pd.concat([part[columns] for part in (train, benchmark, held)], verify_integrity=True)
    sampled = curate(pool, queries, labels)
    rng = np.random.default_rng(112525)
    by_provider = sampled.groupby(['model', 'source_dataset']).text.transform('size')
    draw = -np.log1p(-rng.random(len(sampled))) * np.sqrt(24. + by_provider)
    external = sampled.assign(draw=draw).sort_values(['draw', 'id']).groupby('model').head(500)
    external = external.set_index('id').rename(columns={'prompt_canon': 'prompt_id'}).assign(cohort='external')
    full = pd.concat([external[columns], queries], verify_integrity=True)
    return full.assign(n_words=full.text.str.split().str.len()), labels


def embeddings(model_dir, text, batch=24, max_tokens=192, *, float32=False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).eval()
    if float32:
        model.float()
    ids = np.argsort(text.str.len().to_numpy(), kind='stable')
    result = np.empty((len(text), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            take = ids[start:start + batch]
            inputs = tokenizer(text.iloc[take].tolist(), padding=True, truncation=True, max_length=max_tokens,
                               return_tensors='pt')
            features = model(**inputs).last_hidden_state.float()
            mask = inputs['attention_mask'][..., None]
            mean = (features * mask).sum(1) / mask.sum(1)
            result[take] = torch.nn.functional.normalize(mean, dim=1).numpy()
            if start % (batch * 100) == 0:
                print(model_dir.name, start, len(ids), round(time.monotonic() - started, 1), flush=True)
    if not np.isfinite(result).all():
        raise ValueError('Encoder returned nonfinite features')
    return result


def fit_head(features, frame, rows):
    samples = frame.iloc[rows]
    group = samples.cohort.where(samples.cohort.eq('external'), 'official')
    counts = samples.groupby([group, 'model']).text.transform('size').to_numpy()
    weight = (1. / counts)
    weight /= weight.mean()
    model = LogisticRegression(C=4., solver='lbfgs', max_iter=500)
    return model.fit(features[rows], samples.model, sample_weight=weight)


def score(features, frame, labels):
    position = np.arange(len(frame))
    external = position[frame.cohort.eq('external')]
    official = position[frame.cohort.eq('official')]
    evaluation = position[~frame.cohort.eq('external')]
    outside_official = position[~frame.cohort.isin(['external', 'official'])]
    all_head = fit_head(features, frame, np.r_[external, official])
    external_head = fit_head(features, frame, external)
    if list(all_head.classes_) != labels or list(external_head.classes_) != labels:
        raise ValueError('Heads must contain all label columns')
    external_scores = external_head.predict_log_proba(features[evaluation])
    result = np.empty((len(frame), len(labels)))
    result[outside_official] = all_head.predict_log_proba(features[outside_official])
    for fit, test in GroupKFold(5).split(official, groups=frame.prompt_id.iloc[official]):
        model = fit_head(features, frame, np.r_[external, official[fit]])
        result[official[test]] = model.predict_log_proba(features[official[test]])
    index = frame.index[evaluation]
    return pd.DataFrame(result[evaluation], index=index, columns=labels), pd.DataFrame(
        external_scores, index=index, columns=labels)


def compare(frame, public, heads, output):
    truth = read_ids(public / 'public_truth.parquet')
    trace, original = read_trace(public, truth)
    rows = frame.drop(columns='text').loc[heads[ENCODERS[0]].index].copy()
    rows.loc[truth.index, 'baseline'] = original
    panel_true = truth.model.to_numpy()
    labels = np.asarray(trace['labels'])
    stage = np.where((trace['glm_winner'] | trace['fable_guard'])[:, None],
                     trace['glm_logp'], trace['sibling_logp'])
    top = labels[np.argsort(-stage, axis=1, kind='stable')[:, :3]]
    missed = (original != panel_true) & ~(top == panel_true[:, None]).any(1)
    rows.loc[truth.index, 'original_top3_miss'] = missed
    for arm, logits in heads.items():
        columns = logits.columns.to_numpy()
        rank = np.argsort(-logits.to_numpy(), axis=1, kind='stable')
        rows[arm] = columns[rank[:, 0]]
        rows[f'{arm}_top3'] = (columns[rank[:, :3]] == rows.model.to_numpy()[:, None]).any(1)
        rows.loc[truth.index, f'{arm}_blend'] = replay_predictions(
            trace, truth.text, external_logits=logits, weight=PRIMARY_BLEND)
    results = []
    for cohort, collection in rows.groupby('cohort'):
        for subset, group in (('all', collection), ('short', collection[collection.n_words.lt(100)])):
            if not len(group):
                continue
            entry = dict(cohort=cohort, subset=subset,
                         frozen_encoder=paired_stats(group, *ENCODERS, draws=3000),
                         coverage={key: float(group[f'{key}_top3'].mean()) for key in ENCODERS})
            if group.baseline.notna().all():
                needs_new = group.original_top3_miss.eq(True)
                entry['baseline_errors_outside_top3'] = int(needs_new.sum())
                entry['new_true_top3'] = {key: int((needs_new & group[f'{key}_top3']).sum()) for key in ENCODERS}
                entry['blend'] = {arm: paired_stats(group, 'baseline', f'{arm}_blend', draws=3000)
                                  for arm in ENCODERS}
            results.append(entry)
    rows.to_parquet(output / 'paired_predictions.parquet')
    return results


def run(official, public, audit, pool, encoders, output, roberta_cache=None, style_float32=False):
    import torch

    frame, labels = prepare_inputs(official, public, audit, pool)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    sample = frame.loc[frame.cohort.eq('external')].drop(columns='text')
    sample.to_parquet(output / 'sample.parquet')
    heads, external_only = {}, {}
    if roberta_cache is not None:
        if not pd.read_parquet(roberta_cache / 'sample.parquet').equals(sample):
            raise ValueError('The saved RoBERTa arm used a different external sample')
        expected_ids = frame.index[~frame.cohort.eq('external')]
    for arm in ENCODERS:
        if arm == 'roberta' and roberta_cache is not None:
            heads[arm] = pd.read_parquet(roberta_cache / 'roberta_logp.parquet')
            external_scores = pd.read_parquet(roberta_cache / 'roberta_external_logp.parquet')
            for saved in (heads[arm], external_scores):
                same_ids(expected_ids, saved.index, 'Saved RoBERTa predictions', ordered=True)
                if list(saved.columns) != labels or not np.isfinite(saved.to_numpy()).all():
                    raise ValueError('The saved RoBERTa scores are not aligned')
            print('Restored both RoBERTa score matrices', len(expected_ids), flush=True)
        else:
            representation = embeddings(encoders / arm, frame.text, float32=arm == 'style' and style_float32)
            if arm == 'style':
                np.savez(output / 'style_embedding.npz', ids=frame.index.to_numpy(dtype=str), values=representation)
            heads[arm], external_scores = score(representation, frame, labels)
        heads[arm].to_parquet(output / f'{arm}_logp.parquet')
        external_scores.to_parquet(output / f'{arm}_external_logp.parquet')
        predictions = external_scores.idxmax(1)
        scored = frame.loc[predictions.index]
        external_only[arm] = {cohort: balanced_accuracy(part.model, predictions.loc[part.index])
                              for cohort, part in scored.groupby('cohort')}
    report = dict(sample_rows=len(sample), training_label_counts=sample.model.value_counts().to_dict(),
                  fixed_blend=PRIMARY_BLEND, external_only_ba=external_only,
                  style_precision='float32' if style_float32 else 'checkpoint',
                  cohorts=compare(frame, public, heads, output))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('StyleDistance diagnostics complete', len(sample), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('official', 'public', 'audit', 'pool', 'encoders', 'output'):
        cli.add_argument('--' + arg, type=Path, required=True)
    cli.add_argument('--roberta-cache', type=Path)
    cli.add_argument('--style-float32', action='store_true')
    run(**vars(cli.parse_args()))
