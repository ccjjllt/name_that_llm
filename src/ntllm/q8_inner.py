"""Fixed inner prompt holdouts inside the existing official outer folds."""
import hashlib
import json
from pathlib import Path

import numpy as np


def content_identity(frame):
    columns = [name for name in ('id', 'prompt_id', 'text', 'model', 'fold') if name in frame]
    records = frame.sort_values('id')[columns].astype(str).to_dict(orient='records')
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def fixed_inner_split(official, outer_fold, seed=42, dev_fraction=.2):
    """Return fit, development, untouched outer rows, and their frozen identity.

    Original fold values remain unchanged. An evaluation cache may use the
    outer-fold number as its cache key, but must retain role=inner_dev/outer_test.
    Splitting uses prompt identity only, never validation scores or text features.
    """
    if not 0 < dev_fraction < 1 or outer_fold not in set(official.fold):
        raise ValueError('A valid outer fold and nonempty inner fraction are required.')
    if not official.id.is_unique or official.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Official IDs must be unique and prompt folds disjoint.')
    outer = official.loc[official.fold.eq(outer_fold)].copy()
    train = official.loc[official.fold.ne(outer_fold)].copy()
    prompts = sorted(train.prompt_id.astype(str).unique())
    count = max(1, min(len(prompts) - 1, round(len(prompts) * dev_fraction)))
    chosen = set(np.random.default_rng(seed + 1009 * int(outer_fold)).permutation(prompts)[:count])
    dev = train.loc[train.prompt_id.astype(str).isin(chosen)].copy()
    fit = train.loc[~train.prompt_id.astype(str).isin(chosen)].copy()
    if any(set(a.prompt_id) & set(b.prompt_id) for a,b in ((fit,dev),(fit,outer),(dev,outer))):
        raise ValueError('Inner/outer prompt membership overlaps.')
    labels = sorted(official.model.unique())
    if set(fit.model) != set(labels) or set(dev.model) != set(labels):
        raise ValueError('This holdout does not cover every target class; predeclare another split.')
    manifest = dict(policy='fixed_inner_prompt_holdout_v1', outer_fold=int(outer_fold), seed=int(seed),
                    dev_fraction=float(dev_fraction), labels=labels, official_identity=content_identity(official),
                    split_basis='Prompt IDs only; no score, label-stratification or text-based selection.',
                    public_external_history='Fixed public upstream data may have official-informed selection history; this is a development protocol, not a wholly independent new-prompt dataset.')
    for name, frame in (('inner_fit',fit),('inner_dev',dev),('outer_test',outer)):
        manifest[name] = dict(ids=frame.id.astype(str).tolist(), prompts=sorted(frame.prompt_id.astype(str).unique()),
                              rows=len(frame), identity=content_identity(frame),
                              content_sha256=content_identity(frame.drop(columns='fold')))
    manifest['split_sha256'] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
    return fit.reset_index(drop=True), dev.reset_index(drop=True), outer.reset_index(drop=True), manifest


def write_split(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('An existing inner holdout cannot be silently changed.')
        return
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def validate_split_manifest(manifest):
    identity = hashlib.sha256(json.dumps({key:value for key,value in manifest.items() if key!='split_sha256'},
                                         ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if identity != manifest.get('split_sha256'):
        raise ValueError('Frozen inner split changed.')
    names=('inner_fit','inner_dev','outer_test')
    for field in ('ids','prompts'):
        sets=[set(manifest[name][field]) for name in names]
        if any(not values for values in sets) or any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Frozen inner/outer membership is missing or overlapping.')
    return manifest


def checkpoint_identity(artifacts):
    """Bind actual saved model contents, using stable semantic file names."""
    from .q8_data import file_sha
    if not artifacts:
        raise ValueError('Checkpoint identity needs actual model artifacts, not config alone.')
    files = {name: {'path':str(Path(path).resolve()), 'sha256':file_sha(Path(path))}
             for name,path in sorted(artifacts.items())}
    identity = hashlib.sha256(json.dumps({name:value['sha256'] for name,value in files.items()},
                                         sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'model_identity_sha256':identity,'model_artifacts':files}


def validate_training_evidence(evidence, split):
    """Require explicit optimizer/representation fit membership, not OOF labels."""
    from .q8_data import file_sha
    import pandas as pd
    validate_split_manifest(split)
    if evidence.get('split_sha256') != split['split_sha256']:
        raise ValueError('Training evidence belongs to another inner split.')
    required = set(split['inner_fit']['ids'])
    forbidden = set(split['inner_dev']['ids']) | set(split['outer_test']['ids'])
    path = Path(evidence['official_training_rows_file'])
    if file_sha(path) != evidence['official_training_rows_sha256']:
        raise ValueError('Official training-membership artifact changed.')
    training = pd.read_parquet(path)
    seen = set(training.id.astype(str))
    if not seen or seen & forbidden or not seen <= required:
        raise ValueError('The checkpoint saw inner-development/outer rows or another fit scope.')
    if not set(training.prompt_id.astype(str)) <= set(split['inner_fit']['prompts']):
        raise ValueError('Training prompt identities are outside the inner fit set.')
    for name in ('representation_fit_ids', 'head_fit_ids'):
        ids = set(evidence.get(name, []))
        if not ids or ids & forbidden or not ids <= required:
            raise ValueError(f'{name} is missing or includes held-out official rows.')
    if evidence.get('model_scope') not in ('q8_inner_fit', 'cpu_inner_fit'):
        raise ValueError('An ordinary outer OOF checkpoint is not an inner-held-out checkpoint.')
    artifacts=evidence.get('model_artifacts',{})
    if not artifacts:
        raise ValueError('Training evidence needs saved model artifacts.')
    actual=checkpoint_identity({name:record['path'] for name,record in artifacts.items()})
    if actual['model_identity_sha256'] != evidence.get('model_identity_sha256') or any(
            actual['model_artifacts'][name]['sha256'] != record['sha256'] for name,record in artifacts.items()):
        raise ValueError('Checkpoint artifact contents changed.')
    return dict(official_training_rows=len(seen), prompt_count=int(training.prompt_id.nunique()),
                scope=evidence['model_scope'], source_rows_sha256=evidence['official_training_rows_sha256'],
                model_identity_sha256=actual['model_identity_sha256'],
                membership_verified=True,
                limitation='Membership is verified against explicit training artifacts; upstream external-data history is retained separately.')
