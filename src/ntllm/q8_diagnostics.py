"""Sparse, state-preserving objective and parameter-update observations.

The backward callback reuses the production loss implementation. It must only
backpropagate, never step an optimizer. Two same-RNG passes (lambda=0 and 1)
separate the objectives without maintaining another copy of the Q8 objective.
"""

import copy
import math
import random

import numpy as np


def component(name):
    if 'lora_A' in name:
        return 'lora_A'
    if 'lora_B' in name:
        return 'lora_B'
    if name.startswith('plain.'):
        return 'plain_head'
    if name.startswith('centered.'):
        return 'centered_head'
    if name.startswith('norms.'):
        return 'layer_norm'
    if name.startswith('backbone.'):
        return 'backbone_other'
    return 'other'


def _groups(names):
    groups = {'all': list(names)}
    for name in names:
        groups.setdefault(component(name), []).append(name)
    groups['lora'] = groups.get('lora_A', []) + groups.get('lora_B', [])
    groups['heads'] = groups.get('plain_head', []) + groups.get('centered_head', [])
    return {name: values for name, values in groups.items() if values}


def _rng_state(model):
    import torch
    device = torch.device(getattr(model, 'device', 'cpu'))
    return dict(python=random.getstate(), numpy=copy.deepcopy(np.random.get_state()),
                cpu=torch.get_rng_state().clone(), device=device,
                accelerator=(torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else
                             torch.mps.get_rng_state().clone() if device.type == 'mps' else None))


def _restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    device = state['device']
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['accelerator'], device)
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['accelerator'])


def _capture_gradients(parameters, scale):
    # Keep accumulated buffers off the GPU. No optimizer or main GradScaler
    # state is touched by the observer.
    return {name: parameter.grad.detach().cpu().to(dtype=__import__('torch').float64).clone() / scale
            for name, parameter in parameters.items() if parameter.grad is not None}


def _norm(values, names):
    return math.sqrt(sum(float(values[name].square().sum()) for name in names if name in values))


def _finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def _direction(first, second, names):
    first_norm, second_norm = _norm(first, names), _norm(second, names)
    dot = sum(float((first[name] * second[name]).sum()) for name in names
              if name in first and name in second)
    denominator = first_norm * second_norm
    cosine = (max(-1., min(1., dot / denominator))
              if denominator > 0 and math.isfinite(denominator) and math.isfinite(dot) else None)
    return dict(plain_norm=_finite_or_none(first_norm), centered_raw_norm=_finite_or_none(second_norm),
                dot=_finite_or_none(dot), cosine=cosine,
                zero_or_nonfinite_direction=cosine is None)


def diagnose_window(model, backward, *, centered_weight=1., gradient_scale=1., clip_norm=1., context=None):
    """Observe two objectives at fixed parameters while restoring execution state.

    `backward(lambda)` calls the production backward_window with an independent
    GradScaler (or disabled scaler) and returns (rows, plain_sum, centered_sum).
    It must not zero/step an external optimizer or update a scaler. All model
    .grad references, registered buffers, module modes and Python/NumPy/Torch
    CPU/current-device RNG are restored, including after an exception.

    The centered gradient is g(lambda=1)-g(lambda=0), so floating-point
    subtraction error is explicit. This is a small-window diagnostic, not an
    assertion that whole-run objective gradients or backend quality are equal.
    """
    import torch
    if not math.isfinite(centered_weight) or centered_weight < 0:
        raise ValueError('centered_weight must be finite and nonnegative.')
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError('gradient_scale must be positive and finite.')
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError('clip_norm must be positive and finite.')
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not parameters:
        raise ValueError('No trainable parameters to observe.')
    previous_gradients = {name: value.grad for name, value in model.named_parameters()}
    versions = {name: value._version for name, value in model.named_parameters()}
    modes = [(module, module.training) for module in model.modules()]
    # References plus copies permit restoration of a buffer mutated in-place
    # or reassigned by a forward, without copying the multi-GB frozen backbone.
    buffers = [(module, dict(module._buffers),
                {name: value.detach().clone() if value is not None else None for name, value in module._buffers.items()},
                set(module._non_persistent_buffers_set)) for module in model.modules()]
    rng = _rng_state(model)

    def restore_forward_state():
        _restore_rng(rng)
        for module, mode in modes:
            module.training = mode
        with torch.no_grad():
            for module, references, values, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(references)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
                for name, reference in references.items():
                    if reference is not None and not torch.equal(reference, values[name]):
                        reference.copy_(values[name])

    results, gradients = [], []
    try:
        for strength in (0., 1.):
            restore_forward_state()
            for parameter in model.parameters():
                parameter.grad = None
            with torch.enable_grad():
                result = backward(strength)
            if any(parameter._version != versions[name] for name, parameter in model.named_parameters()):
                raise RuntimeError('Diagnostic callback modified model parameters; only backward is permitted.')
            results.append(tuple(float(value) for value in result))
            gradients.append(_capture_gradients(parameters, gradient_scale))
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = previous_gradients[name]
        restore_forward_state()
    plain, total_unit = gradients
    centered = {name: total_unit[name] - plain[name] if name in plain and name in total_unit else
                total_unit[name] if name in total_unit else -plain[name]
                for name in set(plain) | set(total_unit)}
    total = {name: (plain.get(name, 0.) + centered_weight * centered.get(name, 0.))
             for name in set(plain) | set(centered)}
    components = {}
    for group, names in _groups(parameters).items():
        values = _direction(plain, centered, names)
        centered_norm = values['centered_raw_norm']
        values.update(centered_weighted_norm=centered_norm * centered_weight if centered_norm is not None else None,
                      combined_norm=_finite_or_none(_norm(total, names)),
                      parameter_tensors=len(names), parameter_elements=sum(parameters[name].numel() for name in names),
                      plain_gradient_tensors=sum(name in plain for name in names),
                      centered_gradient_tensors=sum(name in centered for name in names),
                      plain_nonzero_gradient_tensors=sum(name in plain and bool(torch.count_nonzero(plain[name])) for name in names),
                      centered_nonzero_gradient_tensors=sum(name in centered and bool(torch.count_nonzero(centered[name])) for name in names))
        components[group] = values
    derived_norm = components['all']['combined_norm']
    clip_scale = min(1., clip_norm / (derived_norm + 1e-6)) if derived_norm is not None else None
    losses = []
    for strength, (rows, plain_sum, centered_sum) in zip((0., 1.), results):
        if rows <= 0:
            raise ValueError('The diagnostic callback returned an empty optimizer window.')
        losses.append(dict(probe_centered_weight=strength, rows=int(rows),
                           plain_mean=_finite_or_none(plain_sum / rows),
                           centered_raw_mean=_finite_or_none(centered_sum / rows)))
    return dict(context=dict(context or {}), centered_weight=float(centered_weight), gradient_scale=float(gradient_scale),
                components=components, losses=losses, clip_norm=float(clip_norm),
                repeated_forward_loss_difference={
                    'plain_mean': _finite_or_none(abs(results[1][1] / results[1][0] - results[0][1] / results[0][0])),
                    'centered_raw_mean': _finite_or_none(abs(results[1][2] / results[1][0] - results[0][2] / results[0][0]))},
                derived_global_clip_scale=clip_scale,
                method='Same-parameter same-RNG production backward at lambda=0 and 1; centered = total_unit - plain.',
                state_preserved=['existing .grad object references', 'Python/NumPy/Torch CPU and active-device RNG',
                                 'registered buffer values and references', 'module train/eval flags'],
                limitations=['Finite-precision difference of two gradient evaluations; not an exact symbolic decomposition.',
                             'Scalar global clipping does not make its coefficient an Adam learning-rate multiplier.',
                             'Target-class coefficient equality is not backbone-gradient equality.',
                             'Directions are interpretable only when both objectives have nonzero gradients on the same parameter subset; head-only zero norms have undefined cosine.',
                             'Single fixed-window direction is not evidence of persistent objective suppression.'])


def capture_parameters(model):
    """Copy actual trainable values immediately before optimizer.step()."""
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def measure_parameter_updates(model, before):
    """Measure optimizer-applied changes, not gradient-times-LR predictions."""
    import torch
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(parameters) != set(before):
        raise ValueError('The trainable parameter set changed during the observed update.')
    old, change = {}, {}
    for name, parameter in parameters.items():
        current = parameter.detach().cpu()
        if current.shape != before[name].shape or current.dtype != before[name].dtype:
            raise ValueError('The trainable parameter structure changed during the observed update.')
        old[name] = before[name].to(torch.float64)
        change[name] = current.to(torch.float64) - old[name]
    groups = {}
    for group, names in _groups(parameters).items():
        norm, delta = _norm(old, names), _norm(change, names)
        groups[group] = dict(parameter_norm_before=_finite_or_none(norm), update_norm=_finite_or_none(delta),
                             relative_update_norm=_finite_or_none(delta / norm) if norm > 0 else None,
                             zero_norm_before=norm == 0,
                             changed_tensors=sum(bool(torch.count_nonzero(change[name])) for name in names),
                             parameter_tensors=len(names))
    return dict(components=groups, interpretation='Actual parameter difference across optimizer.step; includes optimizer state and weight decay.')


def gradient_norms(model, *, scale=1.):
    """Read existing gradients before or after clipping without changing them."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Gradient scale must be positive and finite.')
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    values = _capture_gradients(parameters, scale)
    return {group: _finite_or_none(_norm(values, names)) for group, names in _groups(parameters).items()}
