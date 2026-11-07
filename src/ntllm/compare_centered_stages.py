"""Compare external × official centered switches on the same data and recipe."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_centered import validate_fold_predictions
from .compare_lora import KEYS, combine_predictions, metrics, outcomes, paired_interval, read_prediction
from .prepare_centered_reference import ARMS, non_auxiliary, physical_identity
from .q8_data import file_sha
from .q8_inner import fixed_inner_split


PAIRS = {'joint': ('off_off', 'on_on'),
         'official_given_external_off': ('off_off', 'off_on'),
         'official_given_external_on': ('on_off', 'on_on'),
         'external_given_official_off': ('off_off', 'on_off'),
         'external_given_official_on': ('off_on', 'on_on'),
         'external_class_vs_global': ('global_on', 'on_on')}


def score_aligned(loaded):
    """Score already aligned real predictions; this alone makes no design claim."""
    required = set(ARMS) - {'global_on'}
    if not required <= set(loaded) or set(loaded) - set(ARMS):
        raise ValueError('Provide the four stage-switch arms and optionally global_on.')
    rows, labels, _ = loaded['off_off']
    for arm, (other, other_labels, _) in loaded.items():
        if labels != other_labels or not rows[KEYS].equals(other[KEYS]):
            raise ValueError(f'{arm}: prediction IDs, labels, folds, prompts or lengths differ.')
    if rows.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('An evaluation prompt spans folds.')
    predicted = {arm: np.asarray(labels)[p.argmax(axis=1)] for arm, (_, _, p) in loaded.items()}

    def contrasts(mask):
        return {name: outcomes(rows.loc[mask], predicted[a][mask], predicted[b][mask])
                for name, (a, b) in PAIRS.items() if a in loaded and b in loaded}

    pairs = contrasts(np.ones(len(rows), dtype=bool))
    for name, (a, b) in PAIRS.items():
        if name in pairs:
            pairs[name]['prompt_bootstrap_95pct'] = paired_interval(rows, predicted[a], predicted[b])
    group, prompts = pd.factorize(rows.prompt_id)
    target, classes = pd.factorize(rows.true_model)
    count = np.zeros((len(prompts), len(classes)), dtype=int)
    change = np.zeros_like(count)
    truth = rows.true_model.to_numpy()
    correctness = {arm: (predicted[arm] == truth).astype(int) for arm in required}
    interaction = correctness['on_on'] - correctness['on_off'] - correctness['off_on'] + correctness['off_off']
    np.add.at(count, (group, target), 1)
    np.add.at(change, (group, target), interaction)
    draws = np.random.default_rng(42).multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=2000)
    denom = draws @ count
    boot = np.nanmean((draws @ change) / np.where(denom, denom, np.nan), axis=1)
    lengths = pd.cut(rows.n_words, [-1, 59, 99, 159, np.inf], labels=['<60', '60-99', '100-159', '>=160'])
    masks = {f'class/{label}': rows.true_model.eq(label).to_numpy() for label in labels}
    masks.update({f'words/{value}': lengths.eq(value).to_numpy() for value in lengths.cat.categories})
    masks.update({f'fold/{fold}': rows.fold.eq(fold).to_numpy() for fold in sorted(rows.fold.unique())})
    report = dict(primary_readout='plain', labels=labels,
                  arm_metrics={arm: metrics(rows, labels, p) for arm, (_, _, p) in loaded.items()},
                  contrasts=pairs,
                  interaction=dict(definition='on_on - on_off - off_on + off_off',
                                   delta=float(pd.Series(interaction).groupby(truth).mean().mean()),
                                   prompt_bootstrap_95pct=np.quantile(boot, [.025, .975]).tolist()),
                  slices={name: contrasts(mask) for name, mask in masks.items()},
                  interval_scope='Descriptive paired prompt bootstrap, 2000 draws, seed 42; not independent generalization confirmation.')
    paired = rows[KEYS].copy()
    for arm, prediction in predicted.items():
        paired[arm] = prediction
    return report, paired.reset_index()


def verify_and_load(paths):
    """Read completed stages and check the intended controlled comparison."""
    loaded, records, references, cached = {}, {}, {}, {}
    common_initialization = set()

    def stage(path, phase, weight, balance):
        path = Path(path).resolve()
        if path not in cached:
            c = json.loads((path / 'config.json').read_text())
            if c.get('status') != 'complete' or c.get('prepared_only', True):
                raise ValueError(f'{path}: only completed training can estimate an effect.')
            m = json.loads((path / 'metrics.json').read_text())
            if m.get('successful_updates') != c['supervision']['active_updates']:
                raise ValueError(f'{path}: completed updates differ from the declared prefix.')
            cached[path] = (c, physical_identity(path))
        c, physical = cached[path]
        s = c['settings']
        if c['phase'] != phase or s['centered_weight'] != weight or s['centered_balance'] != balance:
            raise ValueError(f'{path}: stage policy does not match the named arm.')
        mass = c.get('actual_centered_coefficient_mass')
        if mass is None or not np.isfinite(mass) or (mass <= 0 if weight else mass != 0):
            raise ValueError(f'{path}: actual centered supervision does not match the stage switch.')
        for key in ('training_code_sha256', 'dtype', 'data_sha256', 'data_manifest_sha256'):
            if not c.get(key):
                raise ValueError(f'{path}: missing {key} evidence.')
        key = (phase, c['fold'])
        comparable = dict(settings=non_auxiliary(s), physical=physical,
                          **{k: c[k] for k in ('training_code_sha256', 'dtype', 'data_sha256', 'data_manifest_sha256', 'labels')})
        if key in references and references[key] != comparable:
            raise ValueError(f'{path}: actual data, order, partners, schedule or non-intervention settings differ.')
        references[key] = comparable
        return c

    for arm, arm_paths in paths.items():
        if arm not in ARMS:
            raise ValueError(f'Unknown arm {arm}.')
        ew, ow = ARMS[arm]
        eb = 'global_mass' if arm == 'global_on' else 'class_mass'
        snapshots, folds, parents = [], {}, set()
        for path in map(Path, arm_paths):
            c = stage(path, 'official', ow, 'class_mass')
            fold = c['fold']
            if type(fold) is not int or fold not in range(5) or fold in folds:
                raise ValueError(f'{arm}: expected one official run per fold 0..4.')
            if c.get('initialization_partial') or not c.get('initialization_run'):
                raise ValueError(f'{path}: a complete external parent is required.')
            parent_path = Path(c['initialization_run']).resolve()
            parent = stage(parent_path, 'external', ew, eb)
            if parent['fold'] != 'shared' or parent.get('conditioning_fold') != 'shared':
                raise ValueError('External runs must be shared across official folds.')
            if c.get('external_initialization_config') != parent:
                raise ValueError(f'{path}: saved external parent metadata changed.')
            proof = parent.get('initialization_proof') or {}
            identity = proof.get('initialization_sha256')
            if not identity or proof.get('status') not in ('available', 'recorded_unavailable'):
                raise ValueError(f'{parent_path}: actual common initial tensors must be recorded.')
            common_initialization.add(identity)
            official_path = Path(c['data_path'])
            if file_sha(official_path) != c['data_sha256']:
                raise ValueError('Official source data changed.')
            official = pd.read_parquet(official_path)
            _, _, _, split = fixed_inner_split(official, fold, c['settings']['inner_seed'], c['settings']['inner_dev_fraction'])
            if split != c.get('inner_split'):
                raise ValueError('Expected the fixed inner-fit/development/outer prompt split.')
            prediction, source = read_prediction(path, view='plain', cohort='full')
            validate_fold_predictions(prediction, official, fold, source['path'])
            snapshots.append(prediction)
            parents.add(str(parent_path))
            folds[fold] = dict(path=str(path.resolve()), parent=str(parent_path), prediction=source)
        if set(folds) != set(range(5)) or len(parents) != 1:
            raise ValueError(f'{arm}: five official folds must share one complete external parent.')
        loaded[arm] = combine_predictions(snapshots)
        records[arm] = folds
    if len(common_initialization) != 1:
        raise ValueError('External arms must share the same recorded actual initial tensors.')
    for a, b in [('off_off', 'off_on'), ('on_off', 'on_on')]:
        if a not in records or b not in records or records[a][0]['parent'] != records[b][0]['parent']:
            raise ValueError(f'{a} and {b} must share the same external checkpoint.')
    return loaded, dict(runs=records, common_initialization_sha256=common_initialization.pop(),
                        scope='Same-data stage-switch comparison; information supply and physical training schedule held fixed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--' + arm.replace('_', '-'), type=Path, nargs='+', required=arm != 'global_on')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)}
    loaded, design = verify_and_load(paths)
    report, paired = score_aligned(loaded)
    report['design'] = design
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    paired.to_parquet(args.output_dir / 'paired_predictions.parquet', index=False)
    print(args.output_dir.resolve())


if __name__ == '__main__':
    main()
