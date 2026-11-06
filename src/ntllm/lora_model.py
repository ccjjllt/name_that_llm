"""Qwen LoRA classifier following the published q8v5 inference structure."""

import json
import hashlib
from pathlib import Path

import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj']


class Q8Classifier(nn.Module):
    def __init__(self, base: Path, config: dict, labels: list[str],
                 device: torch.device, dtype: torch.dtype,
                 checkpoint: Path | None = None):
        super().__init__()
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.labels = list(labels)
        self.layers = list(config['layers'])
        self.max_len = int(config.get('max_len', 384))
        self.base_config_sha256 = hashlib.sha256((base / 'config.json').read_bytes()).hexdigest()
        self.model_config = {
            'keep': int(config['keep']), 'layers': self.layers,
            'max_len': self.max_len, 'r': int(config.get('r', 16)),
            'alpha': int(config.get('alpha', 32)),
            'lora_dropout': float(config.get('lora_dropout', 0.05)),
            'attention_implementation': config.get('attention_implementation', 'sdpa'),
        }
        if checkpoint is not None:
            saved = json.loads((checkpoint / 'model_config.json').read_text())
            if saved['labels'] != self.labels:
                raise ValueError('Checkpoint labels differ from the requested label order.')
            if saved['base_config_sha256'] != self.base_config_sha256:
                raise ValueError('Checkpoint and requested base have different model configurations.')
            for key in ('keep', 'layers', 'r', 'alpha', 'lora_dropout'):
                if saved['model_config'][key] != self.model_config[key]:
                    raise ValueError(f'Checkpoint architecture differs for {key}.')

        model_config = AutoConfig.from_pretrained(str(base), local_files_only=True)
        keep = self.model_config['keep']
        if (not self.layers or self.layers != sorted(set(self.layers))
                or not 0 < self.layers[0] <= self.layers[-1] <= keep <= model_config.num_hidden_layers):
            raise ValueError(f'Invalid pooling layers={self.layers} or keep={keep}.')
        if self.max_len < 2:
            raise ValueError('max_len must leave room for tokens after the excluded first token.')
        model_config.num_hidden_layers = keep
        model_config.use_cache = False
        if getattr(model_config, 'layer_types', None) is not None:
            model_config.layer_types = model_config.layer_types[:keep]
        self.hidden_size = model_config.hidden_size
        if checkpoint is not None and saved['hidden_size'] != self.hidden_size:
            raise ValueError('Checkpoint hidden size differs from the requested base model.')

        self.tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('The tokenizer needs a padding token or an EOS token for padding.')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loading directly onto one device avoids a second full-sized copy.
        trunk = AutoModel.from_pretrained(
            str(base), config=model_config, dtype=dtype, local_files_only=True,
            device_map={'': self.device},
            attn_implementation=self.model_config['attention_implementation'],
        )
        trunk.requires_grad_(False)
        if config.get('gradient_checkpointing', True):
            trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            trunk.enable_input_require_grads()
        if checkpoint is None:
            adapter_config = LoraConfig(
                r=self.model_config['r'], lora_alpha=self.model_config['alpha'],
                lora_dropout=self.model_config['lora_dropout'], bias='none',
                target_modules=TARGET_MODULES,
            )
            self.backbone = get_peft_model(trunk, adapter_config, autocast_adapter_dtype=True)
        else:
            self.backbone = PeftModel.from_pretrained(
                trunk, str(checkpoint / 'adapter'), is_trainable=True, autocast_adapter_dtype=True,
                local_files_only=True,
            )
        for parameter in self.backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        width = self.hidden_size * len(self.layers)
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size, device=self.device, dtype=torch.float32)
            for _ in self.layers
        ])
        self.plain = nn.Linear(width, len(labels), device=self.device, dtype=torch.float32)
        self.centered = nn.Linear(width, len(labels), device=self.device, dtype=torch.float32)
        if checkpoint is not None:
            heads = load_file(str(checkpoint / 'heads.safetensors'))
            for name in ('norms', 'plain', 'centered'):
                state = {key.removeprefix(name + '.'): value for key, value in heads.items()
                         if key.startswith(name + '.')}
                getattr(self, name).load_state_dict(state)

    def encode(self, texts):
        # Bound the tokenizer's temporary Encoding objects for the expanded pool.
        # Padding is still performed per physical forward, preserving token IDs.
        texts = list(texts)
        encoded = []
        for start in range(0, len(texts), 2048):
            encoded.extend(self.tokenizer(texts[start:start + 2048], truncation=True,
                                          max_length=self.max_len, return_attention_mask=False)['input_ids'])
        return encoded

    def representation(self, tokens):
        batch = self.tokenizer.pad({'input_ids': tokens}, return_tensors='pt').to(self.device)
        use_amp = self.device.type in {'mps', 'cuda'} and self.compute_dtype != torch.float32
        with torch.autocast(self.device.type, dtype=self.compute_dtype, enabled=use_amp):
            hidden = self.backbone(**batch, output_hidden_states=True, use_cache=False).hidden_states
        mask = batch['attention_mask'].unsqueeze(-1).float()
        mask[:, 0] = 0
        count = mask.sum(dim=1).clamp(min=1)
        return torch.cat([norm((hidden[layer].float() * mask).sum(dim=1) / count)
                          for layer, norm in zip(self.layers, self.norms)], dim=1)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(directory / 'adapter'), safe_serialization=True)
        heads = {}
        for name in ('norms', 'plain', 'centered'):
            for key, value in getattr(self, name).state_dict().items():
                heads[f'{name}.{key}'] = value.detach().cpu().contiguous()
        save_file(heads, str(directory / 'heads.safetensors'))
        metadata = {'labels': self.labels, 'model_config': self.model_config,
                    'hidden_size': self.hidden_size,
                    'base_config_sha256': self.base_config_sha256,
                    'base_model_type': self.backbone.config.model_type}
        (directory / 'model_config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    @torch.no_grad()
    def predict_outputs(self, tokens, batch_size, keep_features=False):
        """Extract both heads from one representation pass, retaining input order.

        raw_centered_logits are evaluated BEFORE group centering. With bias b,
        a query group's centered logits are u_i - mean_group(u) + b. This permits
        each sparse cohort to form its own groups without rerunning the backbone.
        Gaussian fitting additionally needs the actual fold-specific features.
        """
        self.eval()
        plain = np.empty((len(tokens), len(self.labels)), dtype=np.float32)
        centered = np.empty_like(plain)
        features = (np.empty((len(tokens), self.hidden_size * len(self.layers)), dtype=np.float32)
                    if keep_features else None)
        order = np.argsort([len(row) for row in tokens], kind='stable')
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            z = self.representation([tokens[index] for index in batch])
            plain[batch] = self.plain(z).float().cpu().numpy()
            centered[batch] = self.centered(z).float().cpu().numpy()
            if features is not None:
                features[batch] = z.float().cpu().numpy()
        return dict(plain_logits=plain, raw_centered_logits=centered,
                    centered_bias=self.centered.bias.detach().float().cpu().numpy(), features=features)

    @torch.no_grad()
    def predict(self, tokens, batch_size):
        return self.predict_outputs(tokens, batch_size)['plain_logits']
