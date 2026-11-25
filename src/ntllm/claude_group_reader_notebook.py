"""Package the matched Claude assignment loss study for zhuating dual T4s."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/claude_group_20261125/kaggle_train')
CONFIG = Path('configs/provenance_readers.json')
PINS = ['zhuating/ntl-claude-group-input/1', 'zhuating/ntl-heldout-source-audit/2',
        'zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1']
BASE = 'qwen-lm/qwen-3/Transformers/4b-base/1'


WORKER = '''import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from ntllm.claude_group_train import train_group
from ntllm.peer_context_report import read_ids
from ntllm.sibling_specialist import Specialist, predict, write_json

parser = argparse.ArgumentParser()
for key in ('base', 'config', 'sample', 'work', 'output'):
    parser.add_argument('--' + key, type=Path, required=True)
parser.add_argument('--arm', choices=('plain_ce', 'conditional_group'), required=True)
args = parser.parse_args()
if torch.cuda.device_count() != 1 or 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Each matched arm requires exactly one assigned T4')
config = json.loads(args.config.read_text())
labels = json.loads((args.work / 'labels.json').read_text())
sample = read_ids(args.sample)
queries = read_ids(args.work / 'queries.parquet')
if list(queries) != ['text']:
    raise ValueError('Reader inference expects only passage IDs and bodies')
identity = json.loads((args.work / 'input_manifest.json').read_text())
def digest(lines):
    return hashlib.sha256(('\\n'.join(lines) + '\\n').encode()).hexdigest()
for name, rows in [('train', sample), ('query', queries)]:
    if (digest(rows.index) != identity[f'ordered_{name}_ids_sha256']
            or digest(rows.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest()))
            != identity[f'ordered_{name}_body_hashes_sha256']):
        raise ValueError(f'{name} fingerprint differs in {args.arm}')
torch.manual_seed(config['seed'])
tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, padding_side='right')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = Specialist(args.base, config, len(labels), 'cuda:0')
receipt = train_group(model, tokenizer, sample, labels, config, 'cuda:0', args.arm == 'conditional_group')
receipt.update(ordered_train_ids_sha256=identity['ordered_train_ids_sha256'],
               ordered_train_body_hashes_sha256=identity['ordered_train_body_hashes_sha256'],
               ordered_query_ids_sha256=identity['ordered_query_ids_sha256'],
               ordered_query_body_hashes_sha256=identity['ordered_query_body_hashes_sha256'])
write_json(args.output / f'{args.arm}.json', receipt)
scores = predict(model, tokenizer, queries, labels, config, 'cuda:0')
pd.DataFrame(scores, index=queries.index, columns=labels).to_parquet(args.output / f'{args.arm}_logits.parquet')
'''


RUN = '''from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from ntllm.claude_group_train import CLAUDE, frozen_rows
from ntllm.member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan
from ntllm.paired_attribution import text_key
from ntllm.peer_context_report import paired_stats, read_ids, same_ids
from ntllm.sibling_odds_replay import replay_predictions

ARMS = ('plain_ce', 'conditional_group')
WEIGHT = .15
DRAWS = 3000
mount = Path('/kaggle/input')


def asset(slug, name):
    paths = [p for p in mount.rglob(name) if slug in p.parts]
    if len(paths) != 1:
        raise ValueError(f'Expected one {name} from {slug}, found {paths}')
    return paths[0]


def digest(lines):
    return hashlib.sha256(('\\n'.join(map(str, lines)) + '\\n').encode()).hexdigest()


def bodies(frame):
    return digest(frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest()))


def replay_panel(truth, trace, base, scores, strict):
    labels = np.asarray(trace['labels'])
    rows = truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(baseline=base, strict_topic_holdout=strict)
    prior_typography = replay_predictions(trace, truth.text, typography=False)
    protected = trace['glm_winner'] | trace['fable_guard']
    rows['focus_incumbent'] = rows.baseline.isin(CLAUDE)
    rows['focus_eligible'] = rows.focus_incumbent & ~protected & rows.baseline.eq(prior_typography)
    true_class = rows.model.to_numpy()[:, None]
    base_scores = np.where(protected[:, None], trace['glm_logp'], trace['sibling_logp'])
    def top3(values):
        ranks = np.argsort(-values, axis=1, kind='stable')[:, :3]
        return labels[ranks]
    rows['baseline_true_top3'] = (top3(base_scores) == true_class).any(1)
    for arm in ARMS:
        matrix = scores[arm].loc[rows.index]
        choices = top3(matrix.to_numpy())
        rows[arm] = choices[:, 0]
        rows[f'{arm}_true_top3'] = (choices == true_class).any(1)
        full = pd.Series(replay_predictions(trace, truth.text, external_logits=matrix, weight=WEIGHT), index=rows.index)
        rows[f'{arm}_full'] = full
        rows[f'{arm}_claude_only'] = full.where(rows.focus_eligible & full.isin(CLAUDE), rows.baseline)
    return rows


def summarize(rows):
    full = [f'{arm}_full' for arm in ARMS]
    scoped = [f'{arm}_claude_only' for arm in ARMS]
    comparisons = [('baseline', col) for col in [*full, *scoped]] + [ARMS, tuple(full), tuple(scoped)]
    classes = {}
    for model, part in rows.groupby('model'):
        correct = part.baseline.eq(model)
        classes[model] = dict(rows=len(part), baseline_correct=int(correct.sum()), arms={
            col: dict(fixed=int((part[col].eq(model) & ~correct).sum()), harmed=int((part[col].ne(model) & correct).sum()))
            for col in [*full, *scoped]})
    misses = rows.baseline.ne(rows.model) & ~rows.baseline_true_top3
    left, right = (rows[f'{arm}_true_top3'] for arm in ARMS)
    return dict(rows=len(rows), focus_incumbents=int(rows.focus_incumbent.sum()),
                focus_eligible=int(rows.focus_eligible.sum()), paired=[paired_stats(rows, a, b, DRAWS) for a, b in comparisons],
                by_true_class=classes, true_outside_base_top3=int(misses.sum()),
                recovered_top3_on_misses={ARMS[0]: int((misses & left).sum()), ARMS[1]: int((misses & right).sum()),
                                          'union': int((misses & (left | right)).sum())})


if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('Expected two T4 devices for the matched study')
sample_path = asset('ntl-claude-group-input', 'training.parquet')
input_report = json.loads(sample_path.with_name('report.json').read_text())
sample = read_ids(sample_path)
if (len(sample) != input_report['rows'] or digest(sample.index) != input_report['ordered_train_ids_sha256']
        or bodies(sample) != input_report['ordered_train_body_hashes_sha256']):
    raise ValueError('The archived order or training bodies differ from the frozen audit')
_, _, groups = frozen_rows(sample)
actual_sizes = {str(size): sum(len(group) == size for group in groups) for size in (2, 3)}
if {size: count for size, count in actual_sizes.items() if count} != input_report['surviving_by_group_size']:
    raise ValueError('The exact matched group supply changed from the input receipt')
public = read_ids(asset('ntl-0719-public-decision-trace', 'public_truth.parquet'))
source = asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent
held = read_prvsiyan(source, Path('kimi_panel_index.parquet'))
rerun = asset('ntl-sibling-odds-member-trace', 'probe_manifest.json').parent
held_root = asset('ntl-prvsiyan-member-trace', 'probe_manifest.json').parent
primary_trace, primary_base, *_ = read_capture(rerun, public)
held_trace, held_base, *_ = read_capture(held_root, held)
labels = primary_trace['labels']
if len(public) != 798 or public.cohort.eq('evaluation').sum() != 300 or not np.array_equal(labels, held_trace['labels']):
    raise ValueError('Historical C2 or independent trace changed the 18-label order and scope')
queries = pd.concat([public[['text']], held[['text']]], verify_integrity=True)
train_text_keys = text_key(sample.text)
if not train_text_keys.is_unique or train_text_keys.isin(text_key(queries.text)).any():
    raise ValueError('Duplicate training body or exact overlap with scored evaluation text')

base, = [p.parent for p in mount.rglob('4b-base/1/config.json')]
config = Path('study.json')
work = Path('/kaggle/temp/claude_group_reader')
output = Path('/kaggle/working/claude_group_readers')
work.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
manifest = {key: input_report[key] for key in ('ordered_train_ids_sha256', 'ordered_train_body_hashes_sha256')}
manifest.update(ordered_query_ids_sha256=digest(queries.index), ordered_query_body_hashes_sha256=bodies(queries),
                training_rows=len(sample), query_rows=len(queries), exact_evaluation_body_overlap=0)
(work / 'input_manifest.json').write_text(json.dumps(manifest))
(work / 'labels.json').write_text(json.dumps(labels.tolist()))
queries.to_parquet(work / 'queries.parquet')
command = [sys.executable, '-u', 'claude_group_worker.py', '--base', str(base), '--config', str(config),
           '--sample', str(sample_path), '--work', str(work), '--output', str(output)]
environment = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')


def launch(arm, gpu):
    with (output / f'{arm}.log').open('w') as log, subprocess.Popen(
            command + ['--arm', arm], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))) as worker:
        for line in worker.stdout:
            log.write(line)
            print(arm, line.rstrip(), flush=True)
        if worker.wait() != 0:
            raise RuntimeError(f'{arm} failed')


with ThreadPoolExecutor(max_workers=2) as pool:
    tasks = [pool.submit(launch, arm, index) for index, arm in enumerate(ARMS)]
    for task in tasks:
        task.result()
receipts = {arm: json.loads((output / f'{arm}.json').read_text()) for arm in ARMS}
a, b = [receipts[arm] for arm in ARMS]
required = ('schedule_sha256', 'seed', 'scheduled_batches', 'row_exposures', 'matched_group_exposures',
            'matched_row_exposures', 'train_groups_by_size')
if (any(a[key] != b[key] for key in required) or (a['pair_weight'], b['pair_weight']) != (0., .40)
        or a['scheduled_batches'] != 5566 or a['row_exposures'] != 89044
        or a['matched_group_exposures'] != 2 * len(groups)):
    raise ValueError('The two arms did not train with the identical scheduled batches and exposures')
for arm, receipt in receipts.items():
    if (any(receipt[key] != manifest[key] for key in manifest if key.endswith('_sha256'))
            or receipt['optimizer_steps'] + receipt['skipped'] != receipt['scheduled_batches']):
        raise ValueError(f'{arm} input fingerprint or optimizer accounting changed')
scores = {arm: read_ids(output / f'{arm}_logits.parquet') for arm in ARMS}
for arm, values in scores.items():
    same_ids(values.index, queries.index, arm, ordered=True)
    if list(values.columns) != labels.tolist() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f'{arm} emitted invalid scores')
old = replay_panel(public, primary_trace, primary_base, scores, False)
new = replay_panel(held, held_trace, held_base, scores, held.strict_topic_holdout)
audit = json.loads((held_root / 'body_overlap.json').read_text())
known = {identifier for key in EXPOSURE_KEYS for identifier in audit.get(key, [])}
if not known.issubset(held.index):
    raise ValueError('The baseline exposure audit mentions unknown held queries')
old['runtime_body_overlap'] = False
new['runtime_body_overlap'] = new.index.isin(known)
rows = pd.concat([old, new], verify_integrity=True)
panels = dict(c2=old.loc[old.cohort.eq('evaluation')], prvsiyan_895=new, prvsiyan_316=new.loc[new.strict_topic_holdout])
if known:
    panels['prvsiyan_316_without_runtime_overlap'] = panels['prvsiyan_316'].loc[~panels['prvsiyan_316'].runtime_body_overlap]
report = dict(weight=WEIGHT, input_manifest=manifest, training=receipts, audited_group_sizes=actual_sizes,
              runtime_base_overlap=audit, zero_weight_original_verified=True,
              scoped_rule='Require both original and proposed labels to be one of the three studied Claude variants, protect original GLM/Fable and typography-changed calls.',
              panels={})
for name, group in panels.items():
    report['panels'][name] = {}
    for length, part in [('all', group), ('short', group.loc[group.n_words.lt(100)])]:
        focused = part.model.isin(CLAUDE)
        subgroups = [('all_truth', part), ('target_claude_truth', part.loc[focused]),
                     ('other_truth', part.loc[~focused]), ('target_claude_incumbent', part.loc[part.focus_incumbent])]
        report['panels'][name][length] = {key: summarize(subset) for key, subset in subgroups if len(subset)}
rows.to_parquet(output / 'predictions.parquet')
changes = [f'{arm}_{scope}' for arm in ARMS for scope in ('full', 'claude_only')]
rows.loc[rows[changes].ne(rows.baseline, axis=0).any(axis=1)].to_parquet(output / 'changed_rows.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Completed exact same-base replay', len(rows), flush=True)
'''


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    modules = ('claude_group_train', 'exact_pair_train', 'sibling_specialist', 'paired_attribution',
               'sibling_odds_replay', 'peer_context_report', 'member_evidence', 'member_evidence_transfer')
    for module in modules:
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<embedded {module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    setup += f"Path('study.json').write_text({CONFIG.read_text()!r})\n"
    compile(WORKER, '<Claude group GPU worker>', 'exec')
    setup += f"Path('claude_group_worker.py').write_text({WORKER!r})\n"
    metadata = dict(id='zhuating/ntl-claude-group-readers', title='NTL Claude Group Readers',
                    code_file='claude-group-readers.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=True, enable_internet=False, kernel_sources=PINS, model_sources=[BASE])
    notebook = dict(cells=[cell(setup), cell(RUN)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
