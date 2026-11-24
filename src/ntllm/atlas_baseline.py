"""Replay Dennis v1 on public labeled passages and capture its decisions."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

from .glm_prefix_hybrid import cell_index, replace_once, source


PROBES = {
    'ntl-benchmark-c1-generated-panels': ('beroli', 'development'),
    'ntl-benchmark-c2-generated-panels': ('beroli', 'evaluation'),
    'ntl-matched-passages-four-models': ('markbarney', 'reconstructed_stress'),
}
SEED = 2719


def text_keys(values):
    return (values.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\*\*(.*)\*\*$', r'\1', regex=True)
            .str.replace(r'\s+', ' ', regex=True))


def public_queries(root, official, pool=None):
    arms = []
    for dataset, (_, split) in PROBES.items():
        matches = [p for p in root.rglob('passages.csv') if p.parent.name == dataset]
        path, = matches
        frame = pd.read_csv(path, dtype={'prompt_id': str})
        if split == 'reconstructed_stress':
            response = frame.served_model.str.rsplit('/', n=1).str[-1]
            valid = response.eq(frame.model) | (frame.model.eq('qwen3.8-max') & response.eq('qwen3.8-max-0902'))
        else:
            frame = frame.rename(columns={'label': 'model'})
            valid = frame.served_matches.eq(True) & frame.passage_ok.eq(True)
        rows = frame.loc[valid, ['text', 'model', 'prompt_id']].dropna().copy()
        rows['cohort'] = split
        rows['prompt_id'] = dataset + ':' + rows.prompt_id
        arms.append(rows)
    frame = pd.concat(arms, ignore_index=True)
    frame['n_words'] = frame.text.str.split().str.len()
    frame = frame.loc[frame.model.isin(official.model) & frame.n_words.between(40, 250)].copy()
    frame['_key'] = text_keys(frame.text)
    seen = set(text_keys(official.text))
    if pool is not None:
        seen.update(text_keys(pd.read_parquet(pool, columns=['text']).text))
    frame = frame.loc[~frame._key.isin(seen) & ~frame._key.duplicated(keep=False)].copy()

    panels = frame.sample(frac=1, random_state=SEED).drop_duplicates(['prompt_id', 'model'])
    broad = panels.loc[panels.cohort.ne('reconstructed_stress')].groupby('prompt_id').head(6)
    stress = panels.loc[panels.cohort.eq('reconstructed_stress')]
    choose = stress.prompt_id.drop_duplicates().head(80)
    rows = pd.concat((broad, stress.loc[stress.prompt_id.isin(choose)]), ignore_index=True)
    rows['id'] = rows._key.map(lambda value: 'probe-' + hashlib.sha256(value.encode()).hexdigest()[:20])
    columns = ['id', 'text', 'model', 'prompt_id', 'cohort', 'n_words']
    if not rows['id'].is_unique or rows.model.nunique() < 12 or len(rows) < 400:
        raise ValueError('Expected a distinct broad public query cohort')
    return rows[columns].sort_values('id').reset_index(drop=True)


def prepare_notebook(notebook, probe_source):
    result = deepcopy(notebook)
    setup = result['cells'][cell_index(result, 'import glob\n')]
    text = source(setup)
    first = text.index('# ---- ours (S14): V1 rehearsal')
    last = text.index("print('FAST_COMMIT'", first)
    setup['source'] = text[:first] + '''# Execute the real stack on a labeled public query cohort.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
    raise RuntimeError('The labeled diagnostic notebook is for public data only')
from atlas_public_probe import public_queries
official = pd.read_csv(Path(DATA) / 'train.csv', dtype={'id': str, 'prompt_id': str})
pool_paths = sorted(Path('/kaggle/input').rglob('ntl_open_pool.parquet'))
probe = public_queries(Path('/kaggle/input'), official, pool_paths[0] if pool_paths else None)
proof = Path('/kaggle/working/atlas_baseline')
proof.mkdir(exist_ok=True)
probe.to_parquet(proof / 'public_truth.parquet', index=False)
print('Public diagnostic cohort:', probe.groupby('cohort').model.agg(['size', 'nunique']).to_dict(), flush=True)
query_data = Path('/kaggle/temp/atlas-public-probe')
query_data.mkdir(parents=True, exist_ok=True)
for path in Path(DATA).iterdir():
    link = query_data / path.name
    if path.name != 'test.csv' and not link.exists():
        link.symlink_to(path)
probe[['id', 'text']].to_csv(query_data / 'test.csv', index=False)
DATA = str(query_data)
os.environ['NTL_REHEARSAL_DATA'] = DATA
FAST_COMMIT = False
V1_REHEARSAL = True
''' + text[last:]
    apex = next(cell for cell in result['cells'] if '_protect = fable_return(' in source(cell))
    text = replace_once(source(apex), "        _v1_glm_mask |= _protect\n",
                        "        v1_predictions = current_preds.copy()\n"
                        "        v1_glm_winner = _v1_glm_mask.copy()\n"
                        "        _v1_glm_mask |= _protect\n")
    text = replace_once(text, '        _typo_fixed = 0\n',
                        '        sibling_predictions = current_preds.copy()\n'
                        '        _typo_fixed = 0\n')
    text = replace_once(text, "        current_sub['model'] = current_preds\n", '''        stack = pd.read_parquet(V1_OUT / 'stack_submission.parquet')
        trace_ids = query_df.id.astype(str).to_numpy(dtype=str)
        assert stack.id.astype(str).tolist() == trace_ids.tolist()
        assert current_sub.id.astype(str).tolist() == trace_ids.tolist()
        post_lp = _post if _report.get('status') == 'applied' else base_lp
        np.savez_compressed(proof / 'stage_scores.npz', ids=trace_ids, labels=np.asarray(labels, dtype=str),
                            stack_logp=base_lp, glm_logp=post_lp, sibling_logp=refined_lp,
                            typography_probs=P_final, stack=stack.model.to_numpy(dtype=str),
                            glm=np.asarray(v1_predictions, dtype=str), siblings=np.asarray(sibling_predictions, dtype=str),
                            final=np.asarray(current_preds, dtype=str), fable_guard=_protect, glm_winner=v1_glm_winner)
        (proof / 'glm_report.json').write_text(json.dumps(_report, indent=2) + '\\n')
        current_sub['model'] = current_preds
''')
    apex['source'] = text
    result['cells'].insert(0, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                               'source': "from pathlib import Path\nPath('atlas_public_probe.py').write_text(" + repr(probe_source) + ")\n"})
    return result


def build(source_dir, output):
    metadata = json.loads((source_dir / 'kernel-metadata.json').read_text())
    original = json.loads((source_dir / metadata['code_file']).read_text())
    module = Path(__file__).read_text()
    # The staged helper only needs the standalone data functions.
    first = module.index('PROBES = ')
    last = module.index('def prepare_notebook(')
    probe_source = 'import hashlib\n\nimport pandas as pd\n\n\n' + module[first:last]
    notebook = prepare_notebook(original, probe_source)
    metadata.pop('id_no', None)
    metadata.update(id='zhuating/ntl-0719-public-decision-trace', title='NTL 0719 Public Decision Trace',
                    is_private=True, enable_gpu=True, enable_internet=False, docker_image_pinning_type='original')
    metadata['dataset_sources'] += [f'{owner}/{slug}' for slug, (owner, _) in PROBES.items()]
    output.mkdir(parents=True, exist_ok=True)
    for path, value in ((output / metadata['code_file'], notebook), (output / 'kernel-metadata.json', metadata)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == '__main__':
    main()
