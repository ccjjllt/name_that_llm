"""Read shared Q8 settings plus one small, explicit experiment override."""
from pathlib import Path
import tomllib


def load_q8_config(path, experiment=None, *, default_experiment=None):
    with Path(path).open('rb') as handle:
        config = tomllib.load(handle)
    # Recovery cadence and segment limits may change without changing the
    # scientific recipe or invalidating an existing frozen exposure plan.
    config.pop('execution', None)
    selected_default = config.pop('default_experiment', None)
    experiments = config.pop('experiments', {})
    if not experiments:
        if experiment not in (None, 'base'):
            raise ValueError('This flat configuration has no named experiments.')
        return config
    selected = experiment or default_experiment or selected_default
    if selected == 'base':
        return config
    def resolve(name, parents=()):
        if name not in experiments:
            raise ValueError(f'Unknown Q8 experiment {name!r}; choose from {list(experiments)}.')
        if name in parents:
            raise ValueError(f'Circular Q8 experiment inheritance: {(*parents, name)}.')
        overrides = dict(experiments[name])
        parent = overrides.pop('extends', None)
        # Stage tables remain complete declarations, not recursive merging.
        inherited = resolve(parent, (*parents, name)) if parent else {}
        return {**inherited, **overrides}

    return {**config, **resolve(selected)}
