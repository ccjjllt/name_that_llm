"""Replay the fixed candidate selector on the archived Terra/GPT-5.5 short panel."""

import argparse
import base64
import gzip
from hashlib import sha256
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/terra_selector_safety_20261125/kaggle_audit')
KERNEL = 'zhuating/ntl-terra-selector-safety'
PINS = dict(trace='zhuating/ntl-terra-gpt55-holdout-trace/1',
            selector='zhuating/ntl-support-censored-selector-transfer/1',
            prior='zhuating/ntl-0719-public-decision-trace/1',
            independent='zhuating/ntl-heldout-source-audit/2',
            prvsiyan='prvsiyan/prvsiyans-extended-name-that-llm-dataset/13',
            pool='crtproduction/name-that-llm-open-pool/18',
            c1='beroli/ntl-benchmark-c1-generated-panels/1', drj='drjcfmd/ntl-generated-v2/4')
CAPSULE = Path('runs/support_censored_20261125/audit_remote/support_censored_selector/capsules.json')


REPLAY = r'''import hashlib
import json

import pandas as pd

from ntllm.counterfactual_selection_audit import canonical_prompt
from ntllm.heldout_source_audit import text_key
from ntllm.peer_context_report import paired_stats, read_ids, same_ids
from ntllm.support_censored_selector import ARMS, score_capsules
from ntllm.terra_pair_report import read_holdout


def asset(key, name):
    slug = PINS[key].split('/')[1]
    path, = [p for p in Path('/kaggle/input').rglob(name) if slug in p.parts]
    return path


def registry(path, key, value):
    table = pd.read_csv(path, dtype=str, usecols=[key, value]).dropna().drop_duplicates()
    if not table[key].is_unique:
        raise ValueError(f'Conflicting prompt bodies for {path.name}')
    return table.set_index(key)[value]


def prompt_keys(ids):
    return canonical_prompt(ids).str.removeprefix('ntl-benchmark-c1-generated-panels:')


def paired_panel(path):
    labels = read_ids(path / 'truth.parquet')
    queries = read_ids(path / 'queries.parquet')
    same_ids(labels.index, queries.index, path.name, ordered=True)
    return labels.join(queries[['text']])


archive = asset('trace', 'stage_scores.npz').parent
truth, trace, incumbent, overlap = read_holdout(Path('frozen/terra'), archive)
if (len(truth) != 176 or truth.prompt_id.nunique() != 88 or truth.groupby('pair_id').size().ne(2).any()
        or truth.groupby('pair_id').provider_route.nunique().ne(1).any()
        or truth.groupby('provider_route').pair_id.nunique().to_dict() != {'kaggle_proxy': 55, 'openai': 33}
        or overlap['active_pool_ids'] or overlap['official_train_ids']):
    raise ValueError('Archive does not reproduce the frozen clean 88 pairs')

published = read_ids(asset('prior', 'public_truth.parquet'))
c1 = published.loc[published.cohort.eq('development')]
gen003 = paired_panel(Path('frozen/gen003'))
kimi = paired_panel(Path('frozen/kimi'))
new = read_ids(asset('independent', 'candidate_truth.parquet'))
prv = new.loc[new.source_dataset.eq(PINS['prvsiyan'])].join(
    read_ids(asset('independent', 'candidate_queries.parquet'))[['text']])
if list(map(len, (c1, gen003, kimi, prv))) != [180, 191, 191, 895]:
    raise ValueError('The known calibration and previously inspected panels changed')

at_v8 = registry(Path('frozen/prvsiyan_v8_prompts.csv'), 'prompt_id', 'prompt_body')
at_v13 = registry(asset('prvsiyan', 'prompts.csv'), 'prompt_id', 'prompt_body')
target_prompt = truth.prompt_id.map(at_v8)
prior_prompt = prv.prompt_id.str.removeprefix('prvsiyan:').map(at_v13)
if target_prompt.isna().any() or prior_prompt.isna().any():
    raise ValueError('Missing original cross-version Prvsiyan prompts')
pool = pd.read_parquet(asset('pool', 'prompts.parquet'), columns=['prompt_canon', 'prompt_text'])
pool = pool.dropna().drop_duplicates('prompt_canon').set_index('prompt_canon').prompt_text
gen_prompt = gen003.prompt_id.str.removeprefix('pool:').map(pool)
drj = pd.read_json(asset('drj', 'prompts_new.jsonl'), lines=True).set_index('prompt_id').prompt
kimi_prv = kimi.prompt_id.loc[kimi.prompt_id.str.startswith('prvsiyan:')].str.removeprefix('prvsiyan:').map(at_v13)
kimi_drj = kimi.prompt_id.loc[kimi.prompt_id.str.startswith('drjcfmd:')].str.removeprefix('drjcfmd:').map(drj)
c1_raw = pd.read_csv(asset('c1', 'passages.csv'), dtype=str)
prompt_column = next((name for name in ('prompt_text', 'prompt_body', 'prompt', 'user_prompt', 'prompt_content')
                      if name in c1_raw), None)
c1_prompt = pd.Series(dtype=str)
if prompt_column:
    c1_prompt = c1.prompt_id.str.removeprefix('ntl-benchmark-c1-generated-panels:').map(
        c1_raw.drop_duplicates('prompt_id').set_index('prompt_id')[prompt_column])
if gen_prompt.isna().any() or kimi_prv.isna().any() or kimi_drj.isna().any():
    raise ValueError('Missing a published calibration prompt expected in its versioned registry')

keys = text_key(truth.text)
prompt_ids = prompt_keys(truth.prompt_id)
body_sets = dict(calibration=pd.concat([c1.text, gen003.text, kimi.text], ignore_index=True), prvsiyan_v13=prv.text)
prompt_sets = dict(calibration=pd.concat([c1.prompt_id, gen003.prompt_id, kimi.prompt_id], ignore_index=True),
                   prvsiyan_v13=prv.prompt_id)
real_prompts = dict(calibration=pd.concat([c1_prompt, gen_prompt, kimi_prv, kimi_drj], ignore_index=True),
                    prvsiyan_v13=prior_prompt)
flags = {}
for name in body_sets:
    flags[f'{name}_body'] = keys.isin(text_key(body_sets[name].dropna()))
    flags[f'{name}_prompt_id'] = prompt_ids.isin(prompt_keys(prompt_sets[name]))
    flags[f'{name}_prompt_body'] = text_key(target_prompt).isin(text_key(real_prompts[name].dropna()))
flags = pd.DataFrame(flags, index=truth.index)
removed = truth.pair_id.isin(truth.loc[flags.any(axis=1), 'pair_id'])

capsule = asset('selector', 'capsules.json').read_bytes()
if hashlib.sha256(capsule).hexdigest() != CAPSULE_SHA256:
    raise ValueError('The selector capsule does not match the frozen /1 download')
receipt = json.loads((archive / 'glm_report.json').read_text())
calls = score_capsules(truth.text.to_frame().reset_index(), trace, incumbent, receipt, json.loads(capsule))
records = truth.drop(columns=['text']).join(calls).join(flags)
records['admitted'] = ~removed
clean = records.loc[records.admitted]
if clean.empty:
    print('All exact archived pairs overlap a known prompt or body')


def summarize(rows):
    comparisons = [('baseline', name) for name in ARMS]
    comparisons += [('original', 'post_rank_veto'), ('original', 'pre_rank_eligibility'),
                    ('post_rank_veto', 'pre_rank_eligibility'), ('pre_rank_eligibility', 'censored_fit')]
    both = pd.DataFrame({name: rows[name].eq(rows.model) for name in ('baseline', *ARMS)})
    group = rows.pair_id
    complete = both.groupby(group).all().loc[group.value_counts().loc[lambda count: count.eq(2)].index]
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()),
                true_classes=rows.model.value_counts().to_dict(), native_routed=int(rows.native_routed.sum()),
                supported_action=int(rows.supported_action.sum()), original_top_supported=int(rows.original_top_supported.sum()),
                complete_pairs=len(complete), both_right={name: int(complete[name].sum()) for name in complete},
                pair_changes={name: dict(fixed=int((~complete.baseline & complete[name]).sum()),
                                         harmed=int((complete.baseline & ~complete[name]).sum())) for name in ARMS},
                paired=[paired_stats(rows, before, after, 5000) for before, after in comparisons],
                by_true_class={label: {name: dict(original_correct=int(group.baseline.eq(label).sum()),
                                                 correct=int(group[name].eq(label).sum()),
                                                 fixed=int((group.baseline.ne(label) & group[name].eq(label)).sum()),
                                                 harmed=int((group.baseline.eq(label) & group[name].ne(label)).sum()))
                                       for name in ARMS} for label, group in rows.groupby('model')})


scopes = dict(all=clean, near_v18_clean=clean.loc[clean.max_prompt_similarity_v18.lt(.85)], full_archive=records)
scopes.update({f'provider_{name}': subset for name, subset in clean.groupby('provider_route')})
scopes.update({f'native_{label}': clean.loc[clean.native_routed.eq(value)]
               for label, value in [('routed', True), ('not_routed', False)]})
report = dict(pins=PINS, capsule_sha256=CAPSULE_SHA256, archived_rows=len(truth),
              scope='Same-publisher transfer on two GPT true classes with 18-way decisions',
              flags=flags.sum().to_dict(), removed_rows=int(removed.sum()), removed_prompts=truth.loc[removed].prompt_id.nunique(),
              known_prompt_body_matches=dict(c1_template_column=prompt_column, c1_rows_resolved=int(c1_prompt.notna().sum()),
                                             gen003_rows_resolved=int(gen_prompt.notna().sum()),
                                             kimi_rows_resolved=int(kimi_prv.notna().sum() + kimi_drj.notna().sum()),
                                             calibration_rows=len(c1) + len(gen003) + len(kimi),
                                             prvsiyan_v13_rows_resolved=int(prior_prompt.notna().sum())),
              near_v18_prompt_limit=.85, native_status=receipt['status'], archive_visible_body_overlap=overlap,
              panels={name: summarize(part) for name, part in scopes.items() if len(part)})
output = Path('/kaggle/working/terra_selector_safety')
output.mkdir(exist_ok=True)
records.to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('Retained', len(clean), 'short passages from 176 archived passages', flush=True)
'''


def build(output=OUTPUT):
    modules = ('counterfactual_native_guard', 'counterfactual_selection', 'counterfactual_selection_audit',
               'glm_prefix_response_replay', 'heldout_source_audit', 'member_evidence', 'member_evidence_transfer',
               'peer_context_report', 'provenance_family_replay', 'sibling_odds_replay', 'support_censored_selector',
               'terra_pair_report')
    setup = ("import base64\nimport gzip\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for name in modules:
        code = Path(__file__).with_name(name + '.py').read_text()
        compile(code, f'<{name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    folders = dict(terra='runs/terra_pair_20261124/data/holdout', gen003='runs/peer_retrieval_20261124/independent',
                   kimi='runs/kimi_flash_20261125/data/holdout')
    data = {f'{key}/{name}': (Path(folder) / name).read_bytes() for key, folder in folders.items()
            for name in ('truth.parquet', 'queries.parquet')}
    data['prvsiyan_v8_prompts.csv'] = Path('data/external/prvsiyan-extended-v8/prompts.csv').read_bytes()
    for name, body in data.items():
        path = 'frozen/' + name
        payload = base64.b64encode(gzip.compress(body, mtime=0)).decode()
        setup += f"Path({path!r}).parent.mkdir(parents=True, exist_ok=True)\n"
        setup += f"Path({path!r}).write_bytes(gzip.decompress(base64.b64decode({payload!r})))\n"
    run = f'PINS = {PINS!r}\nCAPSULE_SHA256 = {sha256(CAPSULE.read_bytes()).hexdigest()!r}\n' + REPLAY
    compile(setup, '<Terra setup>', 'exec')
    compile(run, '<Terra replay>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Terra Selector Safety', code_file='terra-selector-safety.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[PINS[key] for key in ('prvsiyan', 'pool', 'c1', 'drj')],
                    kernel_sources=[PINS[key] for key in ('trace', 'selector', 'prior', 'independent')])
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.out)
