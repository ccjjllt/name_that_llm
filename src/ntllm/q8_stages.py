"""Declare auxiliary policies for external and official training separately."""
from copy import deepcopy


KEYS = ('centered_plan', 'centered_balance', 'centered_weight')
LEGACY = dict(centered_plan='native', centered_balance='none', centered_weight=1.)


def _policy(settings):
    return {key: settings.get(key, LEGACY[key]) for key in KEYS}


def _declaration(settings):
    declared = settings.get('supervision')
    if declared is None:
        return None
    if not isinstance(declared, dict) or set(declared) != {'external', 'official'}:
        raise ValueError('supervision must explicitly declare both external and official policies.')
    for phase, values in declared.items():
        if not isinstance(values, dict) or set(values) != set(KEYS):
            raise ValueError(f'supervision.{phase} must declare exactly {KEYS}.')
    return deepcopy(declared)


def resolve_stage_supervision(config, phase, parent=None, overrides=None, *, frozen=False):
    """Resolve one stage, retaining actual external history in the declaration.

    Explicit stage policies permit an external-only ablation. Legacy configs
    retain their former joint-stage inheritance and label that scope clearly.
    CLI values affect only the current stage of an explicit declaration.
    """
    if phase not in ('external', 'official'):
        raise ValueError('Choose external or official supervision.')
    overrides = {key: value for key, value in (overrides or {}).items()
                 if key in KEYS and value is not None}
    if frozen and any(config.get(key, LEGACY[key]) != value for key, value in overrides.items()):
        raise ValueError('A frozen or resumed stage cannot change its auxiliary policy.')
    declared = _declaration(config)
    parent_settings = (parent or {}).get('settings', {})
    source = 'explicit_config' if declared else None
    if declared is None and parent:
        declared = _declaration(parent_settings)
        if declared:
            source = 'explicit_external_declaration'
    external = _policy(parent_settings) if parent else None
    if declared is not None:
        selected = dict(declared[phase], **overrides)
        declared[phase] = selected
        if phase == 'official' and external is not None:
            # The actual external arm may differ from the config's default arm.
            # Never relabel a global_mass checkpoint as class_mass history.
            declared['external'] = external
        config['supervision'] = declared
        scope = 'explicit_stage_policies'
    else:
        inherited = external if parent else _policy(config)
        selected = dict(inherited, **overrides)
        if parent and selected != inherited:
            raise ValueError('Legacy joint-stage inheritance cannot change auxiliary policy. Declare supervision.external and supervision.official for a stage-specific experiment.')
        source = 'legacy_external_inheritance' if parent else 'legacy_config_or_cli'
        scope = 'legacy_joint_two_stage_policy'
        declared = dict(external=selected if phase == 'external' else external,
                        official=selected)
    config.update(selected)
    return dict(policy='stage_supervision_v1', phase=phase, scope=scope, source=source,
                effective=selected, declared=declared,
                actual_external_policy=external,
                interpretation='Compare both actual external policies and actual official policies before attributing a result to one stage. Equal coefficient totals do not imply equal gradients or partner coverage.')


def centered_training_history(metadata):
    """Use the same executed auxiliary history for inner and outer availability."""
    import json
    from pathlib import Path
    history = [metadata]
    if metadata.get('external_initialization_config'):
        history.append(metadata['external_initialization_config'])
    elif metadata.get('initialization_run'):
        history.append(json.loads((Path(metadata['initialization_run']) / 'config.json').read_text()))
    evidence = []
    for stage in history:
        mass = stage.get('actual_centered_coefficient_mass')
        basis = 'completed_update_coefficients'
        if mass is None:
            mass = sum(epoch.get('effective_total_mass', 0.)
                       for epoch in stage.get('supervision', {}).get('epochs', []))
            basis = 'legacy_supervision_record'
        evidence.append(dict(phase=stage['phase'], effective_mass=float(mass),
                             recorded=('actual_centered_coefficient_mass' in stage or bool(stage.get('supervision'))),
                             basis=basis))
    return evidence
