"""Train one member of the binary Kimi K3 vs GLM Flash specialist."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .paired_attribution import text_key
from .sibling_specialist import Specialist, predict, train, write_json


FLASH, KIMI = LABELS = ('glm-5.3-flash', 'kimi-k3')
SHORT_WORDS = 100


def prepare_rows(training, public, holdout):
    if list(holdout.columns) != ['id', 'text']:
        raise ValueError('Holdout must contain query IDs and bodies only')
    queries = pd.concat([public, holdout], ignore_index=True)
    columns = ['id', 'text', 'model', 'n_words']
    for frame, keys in ((training, columns), (queries, ['id', 'text'])):
        if not frame.id.is_unique or frame[keys].isna().any().any():
            raise ValueError('Rows must have unique IDs and complete required fields')
    keys = text_key(training.text)
    words = training.text.str.split().str.len()
    if (set(training.model) != set(LABELS) or not words.eq(training.n_words).all()
            or not words.between(1, SHORT_WORDS - 1).all() or not keys.is_unique
            or not set(keys).isdisjoint(text_key(queries.text))
            or not set(training.id).isdisjoint(queries.id)):
        raise ValueError('Expected distinct short training passages and no query-body overlap')
    if not holdout.text.str.split().str.len().between(1, SHORT_WORDS - 1).all():
        raise ValueError('Every new source-holdout passage must be short')
    return training.sort_values('id').reset_index(drop=True), queries


def run(args):
    config = json.loads(args.config.read_text())
    config['seed'] = args.seed
    if args.seed not in config['seeds']:
        raise ValueError('Seed must be declared in the study configuration')
    public = pd.read_parquet(args.public, columns=['id', 'text'])
    holdout = pd.read_parquet(args.holdout)
    training, queries = prepare_rows(pd.read_parquet(args.training), public, holdout)
    receipt = dict(seed=args.seed, training_rows=len(training), public_rows=len(public), holdout_rows=len(holdout),
                   class_counts=training.model.value_counts().to_dict(),
                   training_sha256=hashlib.sha256(args.training.read_bytes()).hexdigest(),
                   holdout_sha256=hashlib.sha256(args.holdout.read_bytes()).hexdigest())
    print(receipt, flush=True)
    if args.prepare_only:
        return

    from transformers import AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each seed must run on one assigned T4')
    torch.manual_seed(args.seed)
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    specialist = Specialist(args.base, config, len(LABELS), device)
    train(specialist, tokenizer, training, list(LABELS), config, device, focused=False)
    scores = predict(specialist, tokenizer, queries, list(LABELS), config, device)
    result = queries[['id']].join(pd.DataFrame(scores, columns=LABELS))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f'seed_{args.seed}'
    result.to_parquet(stem.with_suffix('.parquet'), index=False)
    torch.save({name: param.detach().cpu() for name, param in specialist.named_parameters() if param.requires_grad},
               stem.with_suffix('.pt'))
    write_json(stem.with_suffix('.json'), receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'config', 'training', 'public', 'holdout', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
