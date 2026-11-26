"""Compare plain and train-only topic-centered GPT heads on unchanged traces."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .heldout_source_audit import locate, text_key
from .matched_request_supply import GPT
from .member_evidence_transfer import EXPOSURE_KEYS, read_prvsiyan, verify_manifest
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, read_trace, same_ids


ENAI = 'enaimaan/ntl-openai-generated-responses/3'
PRV = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
NAMES = ('plain', 'centered')
WEIGHTS = (0.0, .05, .10, .20, .35, .50)
PRIMARY_WEIGHT = .20
NEAR_PROMPT = .80


def prompt_content(messages):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return ''
    if not isinstance(messages, list):
        return ''
    parts = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block['text'] for block in content if isinstance(block, dict) and isinstance(block.get('text'), str))
    return '\n'.join(parts)


def normalize_prompts(values):
    return text_key(pd.Series(values, dtype='string')).str.replace(
        r'\b\d{1,3}(?:\s*[–-]\s*\d{1,3})?\s+(?:english\s+)?words\b', ' n words', regex=True,
    ).replace('', pd.NA)


def lexical_overlap(candidates, reference):
    reference = reference.dropna().drop_duplicates()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), dtype=np.float32)
    known = vectorizer.fit_transform(reference)
    distinct = candidates.dropna().drop_duplicates()
    result = []
    for offset in range(0, len(distinct), 64):
        batch = distinct.iloc[offset:offset + 64]
        match = vectorizer.transform(batch) @ known.T
        result.append(pd.Series(match.max(axis=1).toarray().ravel(), index=batch.to_numpy()))
    return candidates.map(pd.concat(result))


def fit_rows(audit, enai, external_prompts, official, public, held):
    queries = read_ids(audit / 'candidate_queries.parquet')
    truth = read_ids(audit / 'candidate_truth.parquet')
    same_ids(queries.index, truth.index, 'Audited source input', ordered=True)
    rows = truth.loc[truth.source_dataset.eq(ENAI) & truth.model.isin(GPT)].join(queries)
    if len(rows) != 5237:
        raise ValueError(f'Expected the frozen 5,237 Enaimaan rows. Found {len(rows)}')
    raw = pd.read_parquet(enai / 'train.parquet', columns=['id', 'model', 'text', 'kind']).set_index('id')
    raw.index = raw.index.astype(str)
    selected = raw.loc[rows.source_id]
    if not (np.array_equal(selected.model, rows.model) and np.array_equal(selected.text, rows.text)
            and selected.kind.eq('new').all()):
        raise ValueError('Audited bodies or model labels differ from the original Enaimaan publications')
    with (enai / 'prompts.jsonl').open() as handle:
        messages = [json.loads(line) for line in handle]
    user_prompt = {str(row['response_id']): prompt_content(row['messages']) for row in messages if row.get('messages')}
    prompts = normalize_prompts(rows.source_id.map(user_prompt))
    if prompts.isna().any():
        raise ValueError('The Enaimaan prompt ledger is missing ordered user content')

    known_bodies = dict(official=official.text, c1=public.loc[public.cohort.eq('development'), 'text'],
                        c2=public.loc[public.cohort.eq('evaluation'), 'text'], prvsiyan=held.text)
    body = text_key(rows.text)
    evidence = {name + '_body': body.isin(set(text_key(text))) for name, text in known_bodies.items()}
    external = pd.read_csv(external_prompts, dtype=str, usecols=['prompt_id', 'split', 'prompt_body'])
    held_prompts = set(held.prompt_id.str.removeprefix('prvsiyan:'))
    known_prompts = normalize_prompts(external.loc[external.prompt_id.isin(held_prompts)
                                                  | external.split.eq('train_regen'), 'prompt_body'])
    evidence['known_prompt_exact'] = prompts.isin(set(known_prompts.dropna()))
    evidence['known_prompt_near'] = lexical_overlap(prompts, known_prompts).ge(NEAR_PROMPT)
    condition = pd.DataFrame(evidence).any(axis=1)
    excluded = rows.prompt_id.isin(set(rows.loc[condition, 'prompt_id']))
    subset = rows.loc[~excluded]
    supports = subset.groupby('prompt_id').model.nunique()
    matched = subset.loc[subset.prompt_id.isin(supports.index[supports.eq(len(GPT))])].copy()
    if set(matched.model) != set(GPT) or not matched.index.is_unique:
        raise ValueError('Exact topic support must include all four model classes')
    report = dict(input_rows=len(rows), direct_row_exclusions={k: int(v.sum()) for k, v in evidence.items()},
                  excluded_topic_rows=int(excluded.sum()), retained_complete_topics=int(matched.prompt_id.nunique()),
                  fit_rows=len(matched), fit_by_model=matched.model.value_counts().to_dict(),
                  comparable_prompt_texts=int(known_prompts.nunique()), near_prompt_cutoff=NEAR_PROMPT,
                  row_ids_sha256=hashlib.sha256(('\n'.join(matched.index) + '\n').encode()).hexdigest())
    return matched, report


def train_and_score(training, texts):
    word = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|[^\w\s]', lowercase=False,
                           min_df=3, max_features=20_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), lowercase=False, min_df=3,
                           max_features=40_000, sublinear_tf=True)
    fit = sparse.hstack([word.fit_transform(training.text), char.fit_transform(training.text)], format='csr')
    test = sparse.hstack([word.transform(texts), char.transform(texts)], format='csr')
    codes, groups = pd.factorize(training.prompt_id, sort=True)
    membership = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                                   shape=(len(codes), len(groups)))
    means = sparse.diags(1. / np.bincount(codes)) @ membership.T @ fit
    centered = fit - membership @ means
    scores = {}
    for name, values in zip(NAMES, (fit, centered), strict=True):
        head = LogisticRegression(class_weight='balanced', C=1.0, max_iter=350, solver='lbfgs')
        head.fit(values, training.model)
        scores[name] = pd.DataFrame(head.predict_log_proba(test), index=texts.index, columns=head.classes_).loc[:, list(GPT)]
        print('Fitted', name, 'rows', values.shape[0], 'features', values.shape[1], 'iterations', head.n_iter_.tolist(), flush=True)
    return scores, dict(word_features=len(word.vocabulary_), char_features=len(char.vocabulary_),
                        inference='Both heads see the same one uncentered body. No peer bodies or topic IDs at inference.',
                        centered_train_rows=int(centered.shape[0]))


def archived_panel(folder, truth):
    trace, base = read_trace(folder, truth)
    if (folder / 'probe_manifest.json').is_file():
        verify_manifest(folder / 'probe_manifest.json', truth)
    frozen = read_ids(folder / 'frozen_predictions.parquet')
    same_ids(frozen.index, truth.index, 'Archived Sibling Odds outputs', ordered=True)
    if not np.array_equal(base, frozen.sibling_odds):
        raise ValueError('Live replay differs from archived original Sibling Odds')
    # Use the original accepted sibling adjustment when constructing native
    # four-class margins, then anchor any post-typography winner.
    labels = trace['labels']
    family = np.array([label.split('-')[0] for label in labels])
    focus = pd.Index(labels).get_indexer(GPT)
    full = trace['sibling_logp'].astype(float)
    gap = np.sort(full, axis=1)[:, -2:]
    delta = trace['glm_logp'] - trace['stack_logp']
    modified = full.copy()
    for group in ('gpt', 'gemini', 'claude'):
        subset = family == group
        candidate = delta[:, subset]
        modified[:, subset] += .55 * np.clip(candidate - candidate.mean(axis=1, keepdims=True), -3., 3.)
    accepted = (family[modified.argmax(1)] == family[full.argmax(1)]) & (gap[:, 1] - gap[:, 0] < .8)
    four = np.where(accepted[:, None], modified, full)[:, focus]
    route = np.isin(base, GPT)
    anchor = pd.Index(GPT).get_indexer(base)
    reanchor = route & (four.argmax(1) != anchor)
    idx = np.flatnonzero(reanchor)
    four[idx, anchor[idx]] = four[idx].max(axis=1) + 1e-5
    predictions = truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(
        incumbent=base, route_gpt4=route, native_gpt4=np.asarray(GPT)[four.argmax(1)], reanchored_native_margins=reanchor,
        strict=truth.get('strict_topic_holdout', False),
    )
    return predictions, pd.DataFrame(four, index=truth.index, columns=GPT)


def add_labels(rows, native, heads):
    mask = rows.route_gpt4
    for name, logp in heads.items():
        rows[name] = logp.idxmax(axis=1)
        for weight in WEIGHTS:
            rank = (1. - weight) * native + weight * logp
            rows[f'{name}_{weight:g}'] = rows.incumbent.where(~mask, rank.idxmax(axis=1))
        if not rows[f'{name}_0'].equals(rows.incumbent):
            raise ValueError(f'{name} zero-weight control diverged from the baseline')
    return rows


def summarize(rows, common_weight, draws=2500):
    report = {}
    for name, part in [('C1', rows.loc[rows.cohort.eq('development')]),
                       ('C2', rows.loc[rows.cohort.eq('evaluation')]),
                       ('Prvsiyan', rows.loc[rows.cohort.eq('prvsiyan')]),
                       ('Prvsiyan_strict', rows.loc[rows.strict])]:
        for scope, block in [('all', part), ('short', part.loc[part.n_words.lt(100)])]:
            truth_gpt = block.loc[block.model.isin(GPT)]
            arms = {}
            for tag, weight in [('fixed', PRIMARY_WEIGHT), ('c1_locked', common_weight)]:
                arms[tag] = {head: paired_stats(block, 'incumbent', f'{head}_{weight:g}', draws) for head in NAMES}
                arms[tag]['centered_vs_plain'] = paired_stats(block, f'plain_{weight:g}', f'centered_{weight:g}', draws)
            report[f'{name}/{scope}'] = dict(conditional=arms, gpt_route_rows=int(block.route_gpt4.sum()),
                                             native_margin_reanchors=int(block.reanchored_native_margins.sum()),
                                             standalone=paired_stats(truth_gpt, 'plain', 'centered', draws),
                                             standalone_vs_native={head: paired_stats(truth_gpt, 'native_gpt4', head, draws) for head in NAMES})
    return report


def run(root, prior, output):
    audit = locate(root, 'candidate_truth.parquet', 'ntl-heldout-source-audit').parent
    history_path = locate(root, 'public_truth.parquet', 'ntl-0719-public-decision-trace')
    public = read_ids(history_path)
    hold = read_prvsiyan(audit, prior)
    enai = locate(root, 'lineage-aliases.json', 'ntl-openai-generated-responses').parent
    official = pd.read_csv(locate(root, 'train.csv', 'name-that-llm'), usecols=['text'])
    external_prompts = locate(root, 'prompts.csv', 'prvsiyans-extended-name-that-llm-dataset')
    train, audit_report = fit_rows(audit, enai, external_prompts, official, public, hold)
    print('Training source audited', audit_report, flush=True)
    all_truth = pd.concat((public, hold), verify_integrity=True)
    heads, details = train_and_score(train, all_truth.text)
    historical_folder = locate(root, 'frozen_predictions.parquet', 'ntl-sibling-odds-member-trace').parent
    held_folder = locate(root, 'frozen_predictions.parquet', 'ntl-prvsiyan-member-trace').parent
    archive = [archived_panel(folder, truth) for (folder, truth) in ((historical_folder, public), (held_folder, hold))]
    rows = add_labels(pd.concat([row for row, _ in archive]), pd.concat([score for _, score in archive]), heads)
    dev = rows.loc[rows.cohort.eq('development')]
    lock = {weight: {head: balanced_accuracy(dev.model, dev[f'{head}_{weight:g}']) for head in NAMES} for weight in WEIGHTS}
    locked = max(WEIGHTS, key=lambda w: np.mean(list(lock[w].values())))
    overlap = json.loads((held_folder / 'body_overlap.json').read_text())
    relevant_exposure = {key: len(overlap.get(key, [])) for key in EXPOSURE_KEYS}
    report = dict(training_source=ENAI, training=audit_report, readout=details, fixed_weight=PRIMARY_WEIGHT,
                  c1_shared_weight=locked, c1_scores=lock, original_prvsiyan_runtime_body_exposure=relevant_exposure,
                  panels=summarize(rows, locked),
                  caveats=['Both arms fit identical Enaimaan rows from complete canonical topic groups, not identical sampled requests.',
                           'Publisher labels were previously body-audited against Open Pool v18. Original stack weights may have unreported pretraining exposures.',
                           'Prvsiyan and published reconstructed official prompt bodies are screened; C1/C2 only expose prompt IDs and body duplicate checking is available.',
                           'Enaimaan Astra was collected at low reasoning and the other three at none. Training may learn some settings-specific differences.'])
    output.mkdir(parents=True, exist_ok=True)
    train[['model', 'prompt_id', 'source_id', 'n_words']].to_parquet(output / 'training_index.parquet')
    rows.to_parquet(output / 'predictions.parquet')
    for name, scores in heads.items():
        scores.to_parquet(output / f'{name}_logp.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Shared C1 weight:', locked, flush=True)
