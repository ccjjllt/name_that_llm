"""Physical forward batching and explicit LoRA A/B optimizer choices."""

import math


def lr_multiplier(update_index, total_updates, warmup_fraction):
    """LR multiplier used by the update at zero-based successful-update index.

    LambdaLR applies index zero on construction; after update k its step()
    prepares index k+1. Keeping this function shared avoids a one-step shift
    between the frozen schedule and the optimizer's actual learning rate.
    """
    if total_updates < 1 or update_index < 0:
        raise ValueError('Learning-rate schedules need positive updates and a nonnegative index.')
    if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction <= 1:
        raise ValueError('warmup_fraction must lie in [0, 1].')
    warmup = max(1, int(total_updates * warmup_fraction))
    if update_index < warmup:
        return (update_index + 1) / warmup
    fraction = (update_index - warmup) / max(1, total_updates - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))


def optimizer_group_rates(config, phase):
    """Base LR of every actual optimizer group, including LoRA+ A/B."""
    mode = config.get('optimizer', 'adamw')
    if mode not in ('adamw', 'loraplus'):
        raise ValueError('optimizer must be adamw or loraplus.')
    ratio = float(config.get('loraplus_ratio', 16.0)) if mode == 'loraplus' else 1.0
    if not 0 < ratio < float('inf'):
        raise ValueError('loraplus_ratio must be positive and finite.')
    rate, head_rate = float(config[f'{phase}_lr']), float(config[f'{phase}_head_lr'])
    if not all(math.isfinite(value) and value > 0 for value in (rate, head_rate, rate * ratio)):
        raise ValueError('Optimizer learning rates must be positive and finite.')
    rates = ({'adapter': rate} if mode == 'adamw' else
             {'adapter_A': rate, 'adapter_B': rate * ratio})
    return dict(rates, heads=head_rate)


def learning_rate_schedule(config, phase, total_updates):
    """Materialize the LR used at each update before any model is loaded."""
    rates = optimizer_group_rates(config, phase)
    rows = []
    for index in range(total_updates):
        multiplier = lr_multiplier(index, total_updates, config['warmup_fraction'])
        groups = {name: rate * multiplier for name, rate in rates.items()}
        rows.append(dict(update=index + 1, scheduler_index=index, multiplier=multiplier,
                         adapter_lr=next(iter(groups.values())), head_lr=groups['heads'],
                         group_lrs=groups))
    return rows


def packed_forwards(microbatches, maximum_rows):
    """Keep relation groups intact while combining independent transformer rows."""
    if maximum_rows < 1:
        raise ValueError('forward_batch_rows must be positive.')
    packed, count = [], 0
    for rows, same_prompt in microbatches:
        if len(rows) > maximum_rows:
            raise ValueError('A physical forward cannot split a centered relation group.')
        if packed and count + len(rows) > maximum_rows:
            yield packed
            packed, count = [], 0
        packed.append((rows, same_prompt))
        count += len(rows)
    if packed:
        yield packed


def make_optimizer(model, config, phase):
    """Retain head LR; optionally use a distinct LoRA B LR (LoRA+)."""
    import torch

    mode = config.get('optimizer', 'adamw')
    if mode not in ('adamw', 'loraplus'):
        raise ValueError('optimizer must be adamw or loraplus.')
    rates = optimizer_group_rates(config, phase)
    groups, a_parameters, b_parameters, ordered_adapter = [], [], [], []
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            ordered_adapter.append(parameter)
            (b_parameters if '.lora_B.' in name else a_parameters).append(parameter)
    if mode == 'loraplus' and not b_parameters:
        raise ValueError('LoRA+ needs identifiable lora_B parameters.')
    if mode == 'adamw':
        groups.append(dict(params=ordered_adapter, lr=rates['adapter'], name='adapter'))
    else:
        groups.extend([dict(params=a_parameters, lr=rates['adapter_A'], name='adapter_A'),
                       dict(params=b_parameters, lr=rates['adapter_B'], name='adapter_B')])
    heads = list(model.norms.parameters()) + list(model.plain.parameters()) + list(model.centered.parameters())
    groups.append(dict(params=heads, lr=rates['heads'], name='heads'))
    parameters = ordered_adapter + heads
    fused = bool(config.get('fused_adamw', False)) and model.device.type == 'cuda'
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'], fused=fused)
    return optimizer, parameters
