"""Run a small real Q8 training comparison on Kaggle, without synthetic models."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from .train_lora import prepare, runtime, train, update_windows, write_json
from .q8_data import file_sha
from .q8_config import load_q8_config


def canonical_parameters(model):
    result = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        key = 'backbone/' + name[name.index('layers.'):] if name.startswith('backbone.') else name
        if key in result:
            raise ValueError(f'Ambiguous trainable parameter: {key}')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/q8'))
    parser.add_argument('--config', type=Path, default=Path('configs/q8.toml'))
    parser.add_argument('--experiment', help='Optional named experiment; default uses shared model settings only')
    parser.add_argument('--backend', choices=['hf', 'unsloth', 'liger'], default='hf')
    parser.add_argument('--forward-batch-rows', type=int, default=3)
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--reference-probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if args.resume is not None:
            if args.experiment is not None:
                raise ValueError('A resumed benchmark uses its saved settings.')
            config = json.loads((args.resume / 'run_config.json').read_text())['settings']
            if config.get('backend', 'hf') != args.backend or config.get('forward_batch_rows', config['batch_size']) != args.forward_batch_rows:
                raise ValueError('A resumed benchmark must keep its saved backend and physical batch size.')
        else:
            config = load_q8_config(args.config, args.experiment, default_experiment='base')
        config.update(backend=args.backend, forward_batch_rows=args.forward_batch_rows,
                      optimizer='adamw', fused_adamw=False)
        request = argparse.Namespace(data_dir=args.data_dir, init=None, variant='baseline',
                                     phase='external', fold='shared', plan_only=False,
                                     max_updates=args.updates, centered_plan=None,
                                     centered_balance=None, centered_weight=None)
        frame, _, labels, targets, weights, auxiliary, plan, details = prepare(request, config)
        import torch
        from safetensors.torch import load_file, save_file
        device, dtype = runtime('cuda', 'float16')
        if args.backend == 'unsloth':
            from .unsloth_model import UnslothQ8Classifier as Model
        elif args.backend == 'liger':
            from .liger_model import LigerQ8Classifier as Model
        else:
            from .lora_model import Q8Classifier as Model
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model_start = time.perf_counter()
        model = Model(args.base, config, labels, device, dtype, args.resume)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - model_start
        parameters = canonical_parameters(model)
        restored_parameter_check = None
        if args.resume is not None:
            saved = load_file(str(args.resume / 'adapter/adapter_model.safetensors'))
            expected = {}
            for name, value in saved.items():
                if 'layers.' in name:
                    key = 'backbone/' + name[name.index('layers.'):]
                    key = key.replace('.lora_A.weight', '.lora_A.default.weight')
                    key = key.replace('.lora_B.weight', '.lora_B.default.weight')
                    expected[key] = value
            expected.update(load_file(str(args.resume / 'heads.safetensors')))
            if not set(parameters).issubset(expected):
                raise ValueError('A restored trainable parameter is absent from the saved checkpoint.')
            deviations = [float((value.detach().cpu() - expected[name]).abs().max())
                          for name, value in parameters.items()]
            restored_parameter_check = dict(maximum_absolute_difference=max(deviations),
                                            all_bitwise_equal=all(value == 0 for value in deviations),
                                            tensors=len(deviations))
        if args.resume is None:
            if args.initial_state.exists():
                initial = load_file(str(args.initial_state))
                if set(initial) != set(parameters):
                    raise ValueError('Backends do not expose the same canonical trainable parameters.')
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(initial[name].to(device=parameter.device, dtype=parameter.dtype))
            elif args.backend == 'hf' and args.forward_batch_rows == 3:
                args.initial_state.parent.mkdir(parents=True, exist_ok=True)
                save_file({name: value.detach().cpu().contiguous() for name, value in parameters.items()},
                          str(args.initial_state))
            else:
                raise ValueError('Run the HF reference first to establish a common initialization.')

        windows = [chunk for epoch in plan for chunk in update_windows(
            epoch, config['batch_size'] * config['accumulation'])][:args.updates]
        selected = sorted({index for chunk in windows for rows, _ in chunk for index in rows})
        tokens = [None] * len(frame)
        for index, encoded in zip(selected, model.encode(frame.iloc[selected].text.tolist())):
            tokens[index] = encoded
        probe_tokens = [tokens[index] for index in selected[:6]]
        token_sha = hashlib.sha256(json.dumps(probe_tokens).encode()).hexdigest()
        all_token_sha = hashlib.sha256(json.dumps([tokens[index] for index in selected]).encode()).hexdigest()
        probe_comparison = None
        if args.resume is None:
            probe = model.predict_outputs(probe_tokens, 3, keep_features=True)
            np.savez_compressed(args.out / 'initial_probe.npz', **probe)
            if args.reference_probe.exists():
                reference = np.load(args.reference_probe)
                difference = probe['features'].astype(np.float64) - reference['features']
                probe_comparison = dict(
                    features_relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(reference['features']), 1e-20)),
                    features_max_abs=float(np.abs(difference).max()),
                    plain_logits_max_abs=float(np.abs(probe['plain_logits'] - reference['plain_logits']).max()),
                    centered_logits_max_abs=float(np.abs(probe['raw_centered_logits'] - reference['raw_centered_logits']).max()),
                    plain_argmax_agreement=float(np.mean(probe['plain_logits'].argmax(1) == reference['plain_logits'].argmax(1))))
            else:
                np.savez_compressed(args.reference_probe, **probe)

        # Equal adapter/head tensors, data and optimizer budget are shared. Actual
        # dropout kernels can consume RNG differently across backends/batch shapes.
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        metadata = dict(details, settings=config, labels=labels, phase='external', fold='shared',
                        variant='baseline', max_updates=args.updates, base=str(args.base),
                        data_manifest_sha256=file_sha(args.data_dir / 'manifest.json'),
                        status='running', prepared_only=False,
                        benchmark_scope='Real same-data Q8 optimizer updates; not a validation accuracy experiment.')
        write_json(args.out / 'config.json', metadata)
        torch.cuda.reset_peak_memory_stats()
        metrics = train(model, tokens, targets, weights, auxiliary, plan, config, 'external', args.out,
                        args.updates, resume=args.resume, checkpoint_every=12,
                        stop_after_updates=args.stop_after_updates, metadata=metadata)
        torch.cuda.synchronize()
        model.save(args.out)
        metrics.update(backend=args.backend, forward_batch_rows=args.forward_batch_rows,
                       model_load_seconds=load_seconds, total_wall_seconds=time.perf_counter() - started,
                       allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                       reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                       trainable_parameters=sum(value.numel() for value in parameters.values()),
                       initial_probe=probe_comparison, probe_token_sha256=token_sha,
                       selected_token_sha256=all_token_sha,
                       restored_parameter_check=restored_parameter_check,
                       selected_rows=len(selected), resume=str(args.resume) if args.resume else None,
                       versions={name: importlib.metadata.version(name) for name in
                                 ('torch', 'transformers', 'peft', 'accelerate')})
        records = [json.loads(line) for line in (args.out / 'train_log.jsonl').read_text().splitlines()]
        if len(records) > 6:
            metrics['steady_seconds_per_update'] = (records[-1]['elapsed_seconds'] - records[4]['elapsed_seconds']) / (records[-1]['update'] - records[4]['update'])
        metrics['gradient_scale_retries'] = sum(row['gradient_scale_retries'] for row in records)
        metrics['finite_logged_losses'] = all(np.isfinite(row['plain_ce']) and np.isfinite(row['centered_ce']) for row in records)
        save_file({name: value.detach().cpu().contiguous() for name, value in parameters.items()},
                  str(args.out / 'final_trainables.safetensors'))
        write_json(args.out / 'metrics.json', metrics)
        write_json(args.out / 'status.json', {'status': 'complete', 'interrupted_training_segment': metrics['interrupted']})
        print('BENCHMARK_RESULT', json.dumps(metrics), flush=True)
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
