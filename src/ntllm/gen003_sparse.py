"""Freeze a sparsely observed GEN003 panel, keeping labels outside inference."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from .atlas_baseline import prepare_notebook
from .atlas_notebook import cell, save
from .glm_prefix_hybrid import cell_index, replace_once, source


SEED = 'GEN003-sparse-20261124'
ARCHIVE = Path('runs/fingerprint_q4_20261123/gen003_blind')
OUTPUT = Path('runs/peer_retrieval_20261124/independent')
DENNIS = Path('runs/atlas_0719_20261123/source/v1')


def rank(key, namespace):
    return hashlib.sha256(f'{SEED}/{namespace}/{key}'.encode()).hexdigest()


def freeze(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((archive / 'audit.json').read_text())
    for name in ('queries', 'truth'):
        path = archive / f'{name}.parquet'
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit[name + '_sha256']:
            raise ValueError(f'The frozen {name} file has changed')
    queries = pd.read_parquet(archive / 'queries.parquet')
    ledger = pd.read_parquet(archive / 'truth.parquet')
    if not queries.id.is_unique or set(queries.id) != set(ledger.id) or not ledger.id.is_unique:
        raise ValueError('Frozen ids do not align')
    # Size choices use only archived prompt identifiers. Within-prompt selection
    # and batch order depend only on passage ids with independent hash salts.
    groups = sorted(ledger.prompt_canon.unique(), key=lambda key: rank(key, 'prompt'))
    assert len(groups) == 77 and ledger.groupby('prompt_canon').size().eq(4).all()
    sizes = {group: i % 4 + 1 for i, group in enumerate(groups)}
    ledger['visible_prompt_size'] = ledger.prompt_canon.map(sizes)
    ledger['within'] = ledger.id.map(lambda key: rank(key, 'subset'))
    ledger = ledger.sort_values('within').drop(columns='within')
    ledger['visible'] = ledger.groupby('prompt_canon').cumcount() < ledger.visible_prompt_size
    ledger['source_dataset'] = audit['source']
    selected = ledger[ledger.visible].copy()
    ids = sorted(selected.id, key=lambda key: rank(key, 'batch'))
    visible = queries.set_index('id').loc[ids].reset_index()[['id', 'text']]
    visible[['id']].to_parquet(output / 'visible_ids.parquet', index=False)
    visible.to_parquet(output / 'queries.parquet', index=False)
    ledger.to_parquet(output / 'archive_selection.parquet', index=False)
    truth = selected.set_index('id').loc[ids].reset_index().rename(columns={'prompt_canon': 'prompt_id'})
    truth['cohort'] = 'gen003_sparse'
    truth.to_parquet(output / 'truth.parquet', index=False)
    report = dict(source=audit['source'], source_archive=audit, seed=SEED, archive_rows=len(ledger),
                  visible_rows=len(visible), visible_groups=ledger.groupby('prompt_canon').visible.sum().value_counts().sort_index().to_dict(),
                  visible_classes=truth.model.value_counts().to_dict(),
                  visible_under_100=int(truth.n_words.lt(100).sum()),
                  ordered_ids_sha256=hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest(),
                  batch='GEN003 only, one mixed batch, 18 possible prediction labels')
    (output / 'panel.json').write_text(json.dumps(report, indent=2) + '\n')
    return visible


def build(queries, output):
    metadata = json.loads((DENNIS / 'kernel-metadata.json').read_text())
    original = json.loads((DENNIS / metadata['code_file']).read_text())
    # Reuse the exact Dennis trace instrumentation and replace only query setup.
    notebook = prepare_notebook(original, '')
    payload = base64.b64encode(gzip.compress(queries.to_csv(index=False).encode(), mtime=0)).decode()
    notebook['cells'][0] = cell(
        "import base64, gzip\nfrom pathlib import Path\n"
        "Path('/kaggle/working/gen003_queries.csv').write_bytes(gzip.decompress(base64.b64decode("
        + repr(payload) + ")))\n")
    setup = notebook['cells'][cell_index(notebook, 'import glob\n')]
    s = source(setup)
    first, last = s.index('from atlas_public_probe import'), s.index('query_data = Path(')
    setup['source'] = s[:first] + """probe = pd.read_csv('/kaggle/working/gen003_queries.csv', dtype={'id': str})
assert list(probe) == ['id', 'text'] and probe.id.is_unique
proof = Path('/kaggle/working/atlas_baseline')
proof.mkdir(exist_ok=True)
""" + s[last:]
    # Record exact corpus exposure for this run; model training histories in
    # third-party checkpoints cannot be inferred from matching file names.
    setup['source'] = replace_once(setup['source'], "open_pool = open_pool[~open_pool.flag_train_prompt_regen.astype(bool)]",
        """open_pool = open_pool[~open_pool.flag_train_prompt_regen.astype(bool)]
def _body_keys(series):
    return (series.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\\*\\*(.*)\\*\\*$', r'\\1', regex=True).str.replace(r'\\s+', ' ', regex=True))
_keys = _body_keys(probe.text)
_pool_overlap = probe.id[_keys.isin(_body_keys(open_pool.text))]
_official_overlap = probe.id[_keys.isin(_body_keys(pd.read_csv(Path(DATA) / 'train.csv').text))]
(proof / 'body_overlap.json').write_text(json.dumps(dict(
    active_pool_ids=_pool_overlap.tolist(), official_train_ids=_official_overlap.tolist(),
    active_pool_rows=len(open_pool), checkpoint_training_histories='unknown'), indent=2))
""")
    replay = Path(__file__).with_name('sibling_odds_replay.py').read_text()
    notebook['cells'].append(cell(replay + """
with np.load(proof / 'stage_scores.npz', allow_pickle=False) as data:
    trace = dict(data)
assert trace['ids'].tolist() == probe.id.tolist()
texts = probe.set_index('id').text
dennis = replay_predictions(trace, texts, sibling_odds=False)
assert np.array_equal(dennis, trace['final'])
sibling = replay_predictions(trace, texts)
pre = replay_predictions(trace, texts, typography=False)
restore = ((sibling != pre) & np.char.startswith(pre.astype(str), 'gemini-')
           & texts.loc[trace['ids']].str.contains('—', regex=False).to_numpy())
pd.DataFrame(dict(id=trace['ids'], dennis=dennis, sibling_odds=sibling, pre_typography=pre,
                  dash_only=np.where(restore, pre, sibling))).to_parquet(proof / 'frozen_predictions.parquet', index=False)
"""))
    # Keep only the verified parent mounts. None of the GEN003 source dataset,
    # full four-answer groups, local archives or labels are attached here.
    metadata.pop('id_no', None)
    metadata.update(id='zhuating/ntl-gen003-sparse-trace', title='NTL GEN003 Sparse Trace', is_private=True,
                    enable_gpu=True, enable_internet=False, docker_image_pinning_type='original')
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(freeze(args.archive, args.out), args.out / 'kaggle')
