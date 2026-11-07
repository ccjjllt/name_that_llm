"""Prepare the missing stage-wise centered references from existing Q8 plans.

This only measures local data; it never loads or downloads model weights.
Existing on/on and global/on plans remain immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .q8_config import load_q8_config
from .q8_data import file_sha


ARMS = {'off_off': (0., 0.), 'off_on': (0., 1.),
        'on_off': (1., 0.), 'on_on': (1., 1.), 'global_on': (1., 1.)}
PHYSICAL_COLUMNS = ['update', 'epoch_index', 'window_index', 'microbatch',
                    'slot_in_update', 'physical_pack', 'slot_in_physical_pack',
                    'physical_pack_rows', 'row_position', 'training_slot_id',
                    'text_id', 'exact_text_sha256', 'model', 'prompt_id', 'word_bin',
                    'role', 'relation_group_id', 'relation_group_size',
                    'relation_partner_ids', 'relation_partner_text_sha256',
                    'plain_loss_weight', 'plain_loss_coefficient', 'window_denominator']


def read(path):
    return json.loads(Path(path).read_text())


def physical_identity(path):
    """Exclude auxiliary coefficients while retaining actual inputs and order."""
    rows = pd.read_parquet(Path(path) / 'actual_exposure.parquet', columns=PHYSICAL_COLUMNS)
    return dict(rows=len(rows),
                sha256=hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).values.tobytes()).hexdigest(),
                update_schedule_sha256=file_sha(Path(path) / 'update_schedule.parquet'))


def non_auxiliary(settings):
    return {key: value for key, value in settings.items()
            if key not in ('centered_weight', 'centered_balance', 'supervision')}


def prepare(reference_index, config_path, out):
    out = Path(out).resolve()
    if (out / 'summary.json').exists():
        raise FileExistsError('A reference study already exists; use a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    source = read(reference_index)
    plans = [p for p in source['plans'] if p['experiment'] == 'auxiliary']
    external = {p['external_balance']: Path(p['prepared_plan']) for p in plans if p['phase'] == 'external'}
    parent = read(external['class_mass'] / 'config.json')
    expected = non_auxiliary(parent['settings'])
    for arm in ARMS:
        configured = load_q8_config(config_path, arm)
        configured.setdefault('loraplus_ratio', 16.)
        if non_auxiliary(configured) != expected:
            raise ValueError(f'{arm}: settings differ from the existing auxiliary experiment.')
    data_dir = Path(parent['data_path']).parent
    common = parent['shared_initialization_path']
    tokenizer = parent['tokenizer_path']

    def create(arm, phase, fold, initial=None):
        dest = out / 'plans' / arm / f'{phase}_{fold}'
        dest.mkdir(parents=True, exist_ok=True)
        candidates = list(dest.glob('*/config.json'))
        if candidates:
            if len(candidates) != 1 or not all((candidates[0].parent / name).exists()
                                             for name in ('exposure_manifest.json', 'token_profile.json')):
                raise ValueError(f'Incomplete plan at {dest}; retain it and use another output directory.')
            return candidates[0].parent
        args = [sys.executable, '-m', 'ntllm.train_lora', '--config', str(config_path),
                '--experiment', arm, '--phase', phase, '--variant', 'expanded',
                '--fold', str(fold), '--data-dir', str(data_dir), '--plan-only',
                '--tokenizer', tokenizer, '--initial-state', common, '--output-dir', str(dest)]
        if phase == 'external':
            args += ['--max-updates', str(parent['max_updates'])]
        if initial is not None:
            args += ['--init', str(initial)]
        (dest / 'command.json').write_text(json.dumps(args, ensure_ascii=False, indent=2) + '\n')
        print(f'Preparing {arm} {phase} {fold}', flush=True)
        with (dest / 'prepare.log').open('w') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        candidates = list(dest.glob('*/config.json'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one prepared plan in {dest}.')
        return candidates[0].parent

    external['off'] = create('off_off', 'external', 'shared')
    arms = {}
    for arm, (ext_weight, off_weight) in ARMS.items():
        balance = 'global_mass' if arm == 'global_on' else 'class_mass'
        initial = external[balance if ext_weight else 'off']
        if arm in ('on_on', 'global_on'):
            official = {str(p['fold']): p['prepared_plan'] for p in plans
                        if p['phase'] == 'official' and p['external_balance'] == balance}
        else:
            official = {str(fold): str(create(arm, 'official', fold, initial)) for fold in range(5)}
        arms[arm] = dict(external=str(initial), official=official,
                         external_weight=ext_weight, official_weight=off_weight)

    # Read each unique artifact once, then compare every same-fold physical plan.
    paths = {p for arm in arms.values() for p in [arm['external'], *arm['official'].values()]}
    identities = {p: physical_identity(p) for p in sorted(paths)}
    configs = {p: read(Path(p) / 'config.json') for p in paths}
    for arm, spec in arms.items():
        for phase, key, path in [('external', 'external', spec['external']),
                                 *[('official', fold, p) for fold, p in spec['official'].items()]]:
            ref = arms['on_on']['external'] if phase == 'external' else arms['on_on']['official'][key]
            if identities[path] != identities[ref]:
                raise ValueError(f'{arm} {phase}/{key}: inputs, partners, window order or LR schedule differ.')
            c, r = configs[path], configs[ref]
            if non_auxiliary(c['settings']) != non_auxiliary(r['settings']):
                raise ValueError(f'{arm} {phase}/{key}: non-auxiliary settings differ.')
            if c['settings']['centered_weight'] != spec[f'{phase}_weight']:
                raise ValueError(f'{arm} {phase}/{key}: wrong stage switch.')
            # Older official plans bind their root through the external parent,
            # rather than duplicating --initial-state on the child invocation.
            declared_root = c.get('shared_initialization_path')
            actual_root = (c.get('external_initialization_config') or c).get('shared_initialization_path')
            if actual_root != common or (declared_root is not None and declared_root != common):
                raise ValueError(f'{arm}: common initialization path differs.')
    token_profiles = {p: read(Path(p) / 'token_profile.json') for p in paths}
    budget = dict(training_stages=len(paths),
                  optimizer_updates=sum(configs[p]['supervision']['active_updates'] for p in paths),
                  row_exposures=sum(token_profiles[p]['plain']['row_exposures'] for p in paths),
                  input_tokens=sum(token_profiles[p]['forward']['input_tokens'] for p in paths),
                  padded_tokens=sum(token_profiles[p]['forward']['padded_tokens'] for p in paths))
    summary = dict(status='prepared_not_trained', arms=arms,
                   unique_external_stages=3, unique_official_stages=25,
                   added_stages=16, retained_stages=12, planned_budget=budget,
                   external_updates=parent['supervision']['active_updates'],
                   official_updates_per_fold=192,
                   primary_readout='Plain-head, prompt-grouped OOF balanced accuracy; same 18-class competition.',
                   primary_contrast='on_on - off_off',
                   stage_contrasts=['off_on - off_off', 'on_off - off_off', 'on_on - off_on', 'on_on - on_off'],
                   interaction='on_on - on_off - off_on + off_off',
                   weighting_contrast='on_on - global_on; external weighting only, official class_mass fixed',
                   secondary_readout='Legal P/C/G and inner-selected fusion describe complete systems; C availability differs across arms.',
                   physical_plan_identity=identities,
                   initialization=dict(common_path=common, status='pending_actual_Q8_tensors'),
                   provenance=dict(reference_index=str(Path(reference_index).resolve()),
                                   reference_index_sha256=file_sha(reference_index),
                                   config=str(Path(config_path).resolve()), config_sha256=file_sha(config_path)),
                   limits=['No Q8 training or efficacy result.',
                           'Current public pool and previously observed official prompts support development comparisons, not independent generalization confirmation.',
                           'Information supply is held fixed; changing coverage or partner sampling is a separate experiment.'])
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(out / 'summary.json', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-index', type=Path, default=Path('runs/scale_preparation_20261107/summary.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/centered_reference.toml'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/centered_reference_20261107/preparation'))
    args = parser.parse_args()
    prepare(args.reference_index, args.config, args.output_dir)


if __name__ == '__main__':
    main()
