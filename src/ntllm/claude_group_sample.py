"""Freeze a single class- and length-matched sample for the Claude loss ablation."""

import hashlib
import json

import numpy as np
import pandas as pd

from .peer_context_report import read_ids, same_ids
from .provenance_readers import curate, read_inputs, text_key


CLAUDE = ('claude-opus-4-6', 'claude-opus-4-8', 'claude-sonnet-5')
PROVIDER = 'tadeuhayashida/ntl-gen003-claude-panel/1'
PRVSIYAN = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/13'
N = 44522
GROUP_KEY = ('model', 'word_bin')


def topic_key(prompt):
    return prompt.astype('string').str.strip().str.casefold().str.replace(r'^(?:pool|prvsiyan):', '', regex=True)


def digest(lines):
    return hashlib.sha256(('\n'.join(map(str, lines)) + '\n').encode()).hexdigest()


def audited_rows(directory):
    truth = read_ids(directory / 'candidate_truth.parquet')
    bodies = read_ids(directory / 'candidate_queries.parquet')
    same_ids(truth.index, bodies.index, 'Frozen verified source data', ordered=True)
    return truth.join(bodies)


def build_sample(official, public, pool, frozen_panels, provenance, source, groups_path, output):
    eval_base = read_inputs(official, public, frozen_panels)
    source_rows = audited_rows(source)
    prv = source_rows.loc[source_rows.source_dataset.eq(PRVSIYAN), ['model', 'prompt_id', 'text', 'n_words']]
    if len(prv) != 895:
        raise ValueError('Expected the same 895 blinded Prvsiyan queries')
    reserve = pd.concat([eval_base, prv.assign(cohort='prvsiyan')], verify_integrity=True)
    reserved_text = set(text_key(reserve.text))
    reserved_prompts = set(topic_key(reserve.prompt_id))
    labels = sorted(eval_base.model.unique())
    curated = curate(pool, eval_base, labels).set_index('id', verify_integrity=True)
    old_index = read_ids(provenance).index
    if len(old_index) != N or not old_index.isin(curated.index).all():
        raise ValueError('The frozen provenance-balanced draw has changed')
    original = curated.loc[old_index, ['model', 'text', 'n_words', 'source_dataset']].rename_axis('id')
    original['word_bin'] = original.n_words // 5
    if text_key(original.text).isin(reserved_text).any():
        raise ValueError('The existing draw includes a reserved body')

    supplied = pd.read_parquet(groups_path)
    supplied = supplied.loc[supplied.source.eq('tadeuhayashida') & supplied.grouping.eq('recorded_settings')]
    if len(supplied) != 604 or not supplied.group.is_unique or supplied.near_short_size.eq(3).sum() != 402:
        raise ValueError('Expected the pinned equal-settings Claude group supply')
    ids = pd.Index([q for values in supplied.ids for q in values], name='id')
    if len(ids) != 1610 or not ids.is_unique:
        raise ValueError('A short Claude passage appears in more than one provider request group')
    claude = source_rows.loc[ids].copy()
    if (not claude.source_dataset.eq(PROVIDER).all() or not claude.model.isin(CLAUDE).all()
            or not claude.model.eq(claude.served_id).all()):
        raise ValueError('Matched groups must have the audited returned-model identity')
    claude['word_bin'] = claude.n_words // 5
    claude['body_key'] = text_key(claude.text)
    excluded = (claude.body_key.isin(reserved_text) | topic_key(claude.prompt_id).isin(reserved_prompts)
                | claude.body_key.duplicated(keep=False))
    allowed = set(claude.index[~excluded])
    available = {
        group: part.sort_index().index.tolist()
        for group, part in original.groupby(list(GROUP_KEY))
        if group[0] in CLAUDE
    }
    capacity = {key: len(ids) for key, ids in available.items()}
    kept_groups, replaced, keep = [], [], []
    rejection = {'reserved_body_or_prompt': 0, 'no_length_matched_original': 0}
    ranked = supplied.sort_values(['near_short_size', 'spread', 'group'], ascending=[False, True, True])
    for item in ranked.itertuples(index=False):
        member_ids = list(item.ids)
        part = claude.loc[member_ids]
        if (len(part) != item.near_short_size or len(part) != part.model.nunique()
                or part.prompt_id.nunique() != 1 or not part.n_words.between(40, 99).all()
                or part.n_words.max() - part.n_words.min() > 12 or part.n_words.max() > 1.15 * part.n_words.min()):
            raise ValueError('The archived group violates frozen label/request or length conditions')
        if not all(identifier in allowed for identifier in member_ids):
            rejection['reserved_body_or_prompt'] += 1
            continue
        quota = list(part.loc[:, list(GROUP_KEY)].itertuples(index=False, name=None))
        if any(capacity.get(key, 0) == 0 for key in quota):
            rejection['no_length_matched_original'] += 1
            continue
        for key in quota:
            removed = available[key].pop()
            capacity[key] -= 1
            replaced.append(removed)
        keep.append(part[['model', 'text', 'n_words', 'source_dataset', 'word_bin']].assign(match_group=item.group))
        kept_groups.append(dict(group=item.group, rows=len(part), prompt=part.prompt_id.iloc[0], ids=member_ids))
    if not kept_groups:
        raise ValueError('No complete Claude matched groups survive the frozen holdouts and quotas')
    addition = pd.concat(keep, verify_integrity=True)
    sample = pd.concat([original.drop(index=replaced).assign(match_group=None), addition],
                       verify_integrity=True).sort_index()
    if (len(sample) != N or not text_key(sample.text).is_unique
            or not sample.groupby(list(GROUP_KEY)).size().equals(original.groupby(list(GROUP_KEY)).size())):
        raise ValueError('Combined training must preserve counts per label and five-word bin')
    if topic_key(claude.loc[addition.index].prompt_id).isin(reserved_prompts).any():
        raise ValueError('Matched Claude training has an evaluation topic')
    covered = sample.loc[sample.match_group.notna(), 'match_group'].value_counts()
    replaced_rows = original.loc[replaced]
    report = dict(rows=len(sample), shared_original_rows=N - len(addition), added_claude_rows=len(addition),
                  supplied_groups=len(supplied), surviving_by_group_size=covered.value_counts().sort_index().to_dict(),
                  surviving_by_model=addition.model.value_counts().to_dict(),
                  retained_publisher_style_screen=claude.loc[addition.index].groupby('model').protocol.value_counts().unstack(
                      fill_value=0).astype(int).to_dict(orient='index'),
                  replaced_source_rows=replaced_rows.source_dataset.value_counts().to_dict(),
                  rejected_group_counts=rejection, exact_body_overlap_rows=0,
                  matched_claude_reserved_topic_overlap_rows=0,
                  reserved_cohort_sizes=reserve.cohort.value_counts().to_dict(),
                  ordered_train_ids_sha256=digest(sample.index),
                  ordered_train_body_hashes_sha256=digest(sample.text.map(lambda x:hashlib.sha256(x.encode()).hexdigest())),
                  unchanged_class_word_bin_counts=True)
    output.mkdir(parents=True, exist_ok=True)
    sample[['text', 'model', 'n_words', 'source_dataset', 'match_group']].to_parquet(output / 'training.parquet')
    pd.DataFrame(kept_groups).to_parquet(output / 'selected_groups.parquet', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Frozen two-arm sample', report, flush=True)
    return report
