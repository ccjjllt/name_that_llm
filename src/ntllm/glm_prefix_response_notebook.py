"""Build a paired Low/task and Low/neutral GLM experiment on pinned assets."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from .atlas_notebook import cell, save
from .glm_prefix_hybrid import replace_once, source
from .heldout_source_audit import KIMI_EVAL_INDEX
from .member_trace_notebook import PINNED


AUDIT = 'zhuating/ntl-heldout-source-audit/1'
PLAN = 'zhuating/ntl-glm-prefix-response-panel'
SCORES = 'zhuating/ntl-glm-prefix-response-scores'
LOW = '[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{user}<|assistant|><think></think>'
BINARY_SHA256 = 'a9e071e1529b46f2c79b11a1c0466a83e897bfd01bd0d3fc28f3a859626f25b0'


def notebook(setup, run):
    return {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}


def build_panel(output):
    prior = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup = ("from pathlib import Path\nimport base64\n"
             f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({prior!r}))\n")
    run = '''import hashlib
import json
import pandas as pd

mount = Path('/kaggle/input')
train_path, = [p for p in mount.rglob('train.csv') if p.parent.name == 'name-that-llm']
truth_path, = mount.rglob('candidate_truth.parquet')
pool_truth = pd.read_parquet(truth_path, columns=['id', 'model', 'source_dataset', 'prompt_id', 'n_words'])
pool_queries = pd.read_parquet(truth_path.with_name('candidate_queries.parquet'), columns=['id', 'text'])
if not pool_truth.id.equals(pool_queries.id) or not pool_queries.id.is_unique:
    raise ValueError('The audited source query and label IDs differ')
source_name = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'
prior = pd.read_parquet('prior_kimi.parquet', columns=['source_dataset', 'prompt_id'])
previous = prior.loc[prior.source_dataset.eq(source_name), 'prompt_id']
held = pool_truth.loc[pool_truth.source_dataset.eq(source_name + '/13') & ~pool_truth.prompt_id.isin(previous)]
held = held.drop(columns='source_dataset').join(pool_queries.set_index('id').text, on='id').assign(cohort='heldout')

official = pd.read_csv(train_path, dtype={'id': str, 'prompt_id': str},
                       usecols=['id', 'text', 'model', 'prompt_id'])
classes = set(official.model)
groups = official.groupby('prompt_id').agg(rows=('id', 'size'), models=('model', 'nunique'))
complete = groups.index[groups.rows.eq(18) & groups.models.eq(18)]
order = sorted(complete, key=lambda prompt: hashlib.sha256(('glm-response/1/' + prompt).encode()).digest())
fit = official[official.prompt_id.isin(order[:32])].assign(cohort='calibration')
fit = fit.assign(n_words=fit.text.str.split().str.len())
if len(fit) != 576 or len(held) != 316 or held.prompt_id.nunique() != 23 or set(held.model) != classes:
    raise ValueError('Unexpected scope for the two frozen scoring panels')
panel = pd.concat([fit, held], ignore_index=True)
if not panel.id.is_unique or panel.text.isna().any() or not panel.n_words.eq(panel.text.str.split().str.len()).all():
    raise ValueError('Rows must have unique IDs, complete bodies and matching length')
out = Path('/kaggle/working/prefix_response_panel')
out.mkdir(exist_ok=True)
panel[['id', 'text']].to_parquet(out / 'queries.parquet', index=False)
panel.drop(columns='text').to_parquet(out / 'truth.parquet', index=False)
counts = {}
for name, part in panel.groupby('cohort'):
    counts[name] = dict(rows=len(part), prompts=part.prompt_id.nunique(),
                        classes=part.model.value_counts().to_dict(),
                        short=part.loc[part.n_words.lt(100), 'model'].value_counts().to_dict())
manifest = dict(rows=len(panel), counts=counts,
                ordered_ids_sha256=hashlib.sha256(('\\n'.join(panel.id) + '\\n').encode()).hexdigest(),
                input_body_sha256=[hashlib.sha256(body.encode()).hexdigest() for body in panel.text])
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(counts, flush=True)
'''
    metadata = {
        'id': PLAN, 'title': 'NTL GLM Prefix Response Panel', 'code_file': 'prefix-response-panel.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['name-that-llm'], 'kernel_sources': [AUDIT],
    }
    save(output, metadata, notebook(setup, run))


def paired_pipeline(original):
    guard = '    n_rows = np.array([len(ids) for ids, _ in seqs])\n'
    changed = replace_once(original, guard, '''    if a.max_text_tokens != 160 or '{user}' not in a.prefix:
        raise ValueError('Paired run requires the Low task prefix and a 160-token control')
    if len(seqs) != len(texts):
        raise ValueError('Task tokenization changed the input count')
    neutral_path = tk + '.paired-neutral'
    run([segrun, 'tokenize', '--model', segs[0], '--texts', tx, '--out', neutral_path,
         '--prefix', a.prefix.replace('{user}', ''), '--max-text-tokens', '32'], log)
    neutral = segio.read_tokens(neutral_path)
    if len(neutral) != len(seqs):
        raise ValueError('Neutral tokenization changed the input count')
    for (task_ids, task_pre), (neutral_ids, neutral_pre) in zip(seqs, neutral, strict=True):
        body = neutral_ids[neutral_pre:]
        if not 0 < len(body) <= 32 or not np.array_equal(task_ids[task_pre:task_pre + len(body)], body):
            raise ValueError('The two conditions changed the observed answer token IDs')
    extra = df.copy()
    df[a.id_col] = 'task:' + df[a.id_col].astype(str)
    extra[a.id_col] = 'neutral:' + extra[a.id_col].astype(str)
    df = pd.concat([df, extra], ignore_index=True)
    texts = texts + texts
    seqs = seqs + neutral
    study_rows = len(seqs)
    view_rows = study_rows // 2
    anchors = np.linspace(0, view_rows - 1, 4, dtype=int)
    anchors = np.r_[anchors, anchors + view_rows]
    repeated = np.r_[anchors, np.roll(anchors, 1)].tolist()
    seqs.extend([seqs[i] for i in repeated])
    n_rows = np.array([len(ids) for ids, _ in seqs])
''')
    changed = replace_once(changed, '    ok = [i for i, s in enumerate(all_scores) if s is not None]\n', '''    # Re-score a few exact token sequences in different slots in the same segment passes.
    probe_indices = set(repeated) | set(range(study_rows, len(seqs)))
    slots = {row: [chunk, pos % timing['seq_batch']] for chunk, ids in enumerate(chunks)
             for pos, row in enumerate(ids) if row in probe_indices}
    repeated_scores = []
    for copy, original in enumerate(repeated, study_rows):
        left, right = all_scores[original], all_scores[copy]
        result = dict(original=original, repeated=copy, view='task' if original < view_rows else 'neutral',
                      original_requested_position=slots.get(original), repeat_requested_position=slots.get(copy))
        valid = (left is not None and right is not None and left.shape == right.shape and len(left) > 0
                 and np.isfinite(left).all() and np.isfinite(right).all())
        if valid:
            delta = np.abs(left - right)
            result.update(tokens=len(left), logp_mean_abs=float(delta[:, 0].mean()),
                          logp_max_abs=float(delta[:, 0].max()), entropy_max_abs=float(delta[:, 1].max()),
                          ranks_changed=int(np.count_nonzero(delta[:, 3])))
        repeated_scores.append(dict(result, comparable=bool(valid)))
    timing['repeated_slot_probe'] = repeated_scores
    timing['probe_positions'] = int(n_rows[study_rows:].sum())
    ok = [i for i, s in enumerate(all_scores[:study_rows]) if s is not None]
''')
    compile(changed, 'run_pipeline.py', 'exec')
    return changed


def build_scores(output, parent_metadata, panel_version):
    parent = json.loads(parent_metadata.read_text())
    directory = parent_metadata.parent
    code_file = directory / parent['code_file']
    original = json.loads(code_file.read_text())
    payloads = [m[1] for c in original['cells'] if
                (m := re.search(r'^V1_PAYLOAD = "([A-Za-z0-9+/=]+)"$', source(c), re.MULTILINE))]
    payload, = payloads
    packed = base64.b64decode(payload, validate=True)
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
        files = {n: archive.extractfile(n).read().decode() for n in ('run_pipeline.py', 'segio.py', 'features.py')}
    files['run_pipeline.py'] = paired_pipeline(files['run_pipeline.py'])
    script_hash = hashlib.sha256(files['run_pipeline.py'].encode()).hexdigest()
    setup = ("from pathlib import Path\n"
             + f"control_prefix = {LOW!r}\nexpected_bin = {BINARY_SHA256!r}\nrunner_sha256 = {script_hash!r}\n")
    setup += ''.join(f'Path({name!r}).write_text({body!r})\n' for name, body in files.items())
    run = '''import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
    raise RuntimeError('Offline prefix research only')
mount = Path('/kaggle/input')
panel, = [p.parent for p in mount.rglob('queries.parquet') if p.parent.name == 'prefix_response_panel']
query = pd.read_parquet(panel / 'queries.parquet', columns=['id', 'text'])
sealed = json.loads((panel / 'manifest.json').read_text())
if (len(query) != 892 or len(query) != sealed['rows']
        or hashlib.sha256(('\\n'.join(query.id) + '\\n').encode()).hexdigest() != sealed['ordered_ids_sha256']
        or [hashlib.sha256(x.encode()).hexdigest() for x in query.text] != sealed['input_body_sha256']):
    raise ValueError('Mounted query rows differ from the frozen prefix experiment')
def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

candidates = [p for p in mount.rglob('ntl-segrun') if p.parent.name == 'ntl-bin'
              and p.with_name('lib').joinpath('libcublasLt.so.13').is_file() and sha(p) == expected_bin]
if not candidates:
    raise RuntimeError('Unable to locate the original Sibling Odds GLM binary')
engine = sorted(candidates)[0]
scratch = Path('/kaggle/temp/prefix-response')
scratch.mkdir(parents=True, exist_ok=True)
bin_dir = scratch / 'bin'
bin_dir.mkdir(exist_ok=True)
executable = bin_dir / 'ntl-segrun'
shutil.copyfile(engine, executable)
os.chmod(executable, 0o755)
env = dict(os.environ, LD_LIBRARY_PATH=str(engine.with_name('lib')) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
query_file = scratch / 'queries.csv'
query.to_csv(query_file, index=False)
output = Path('/kaggle/working/prefix_response_scores')
output.mkdir(exist_ok=True)
layouts = [('flash', 'flash_iq3xxs-seg*-of-06-*.gguf', 6), ('glm53', 'glm53_iq3xxs-seg*-of-19-*.gguf', 19)]
deadline = time.monotonic() + 27500
results = {}
for tag, pattern, count in layouts:
    weights = scratch / tag
    weights.mkdir(exist_ok=True)
    paths = sorted(mount.rglob(pattern))
    if len(paths) != count:
        raise ValueError(f'{tag}: expected {count} pinned GGUF segments, found {len(paths)}')
    for file in paths:
        (weights / file.name).symlink_to(file)
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError('The paired scoring deadline expired')
    subprocess.run([sys.executable, '-u', 'run_pipeline.py', '--bin', str(bin_dir), '--segs', str(weights/'*.gguf'),
                    '--csv', str(query_file), '--out', str(output), '--tag', tag, '--work', str(scratch),
                    '--prefix', control_prefix, '--max-text-tokens', '160', '--seq-batch', '8',
                    '--state-dtype', 'f32', '--budget-s', str(remaining), '--prefetch'],
                   env=env, check=True)
    pool = pd.read_csv(output / f'{tag}_features.csv', dtype={'id': str})
    expected = ['task:'+x for x in query.id] + ['neutral:'+x for x in query.id]
    if list(pool.id) != expected:
        raise ValueError(f'{tag}: actual completed rows differ from the fixed two views')
    with np.load(output / f'{tag}_scores.npz', allow_pickle=False) as raw:
        offsets = raw['offsets']
        lengths = np.diff(offsets).reshape(2, len(query)).T
        prefixes = raw['n_prefix'].reshape(2, len(query)).T
        tokens = raw['tokens']
        for a, b in zip(range(len(query)), range(len(query), 2*len(query)), strict=True):
            if not np.array_equal(tokens[offsets[a]:offsets[a]+lengths[a, 1]], tokens[offsets[b]:offsets[b+1]]):
                raise ValueError(f'{tag}: saved token identity check failed for {a}')
    # The exact per-row ceiling includes the second prefix, even for short rows.
    extra = lengths[:, 1] + prefixes[:, 1]
    same_cost_task = lengths[:, 0] - extra
    timing = json.loads((output / f'{tag}_timing.json').read_text())
    results[tag] = dict(task_body_tokens=int(lengths[:, 0].sum()), neutral_body_tokens=int(lengths[:, 1].sum()),
                        task_prefix_tokens=int(prefixes[:, 0].sum()), neutral_prefix_tokens=int(prefixes[:, 1].sum()),
                        positions_for_two_views=int((lengths + prefixes).sum()),
                        repeated_slot_probe=timing['repeated_slot_probe'],
                        probe_positions=timing['probe_positions'],
                        sequence_batch_initial=timing['seq_batch'],
                        sequence_batch_reduced_to=timing.get('seq_batch_reduced_to'),
                        can_match_total_token_cost=int((same_cost_task >= lengths[:, 1]).sum()))
manifest = dict(rows=sealed['rows'], ordered_ids_sha256=sealed['ordered_ids_sha256'], binary_sha256=expected_bin,
                runner_sha256=runner_sha256, task_prefix=control_prefix, models=results)
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(results, flush=True)
'''
    for code in (setup, run):
        compile(code, '<glm-prefix-response>', 'exec')
    pins = [ref for ref in parent['dataset_sources']
            if ref.startswith('jamesliebel/glm53-iq3xxs-segs-') or ref.startswith('jamesliebel/glm53-flash-iq3xxs-segs-')]
    if len(pins) != 3:
        raise ValueError('Expected three original GLM asset pins')
    metadata = {
        'id': SCORES, 'title': 'NTL GLM Prefix Response Scores', 'code_file': 'prefix-response-scores.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': False, 'dataset_sources': pins,
        'kernel_sources': [f'{PLAN}/{panel_version}', 'estopadilla/ntl-glm-prefix-ab/2',
                           'estopadilla/ntl-glm-likelihood-train/6'],
        'docker_image': parent['docker_image'], 'docker_image_pinning_type': 'original',
        'machine_shape': parent['machine_shape'],
    }
    save(output, metadata, notebook(setup, run))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('panel', 'scores'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel-version', type=int, default=1)
    parser.add_argument('--parent', type=Path, default=PINNED)
    args = parser.parse_args()
    if args.stage == 'panel':
        build_panel(args.output)
    else:
        build_scores(args.output, args.parent, args.panel_version)
