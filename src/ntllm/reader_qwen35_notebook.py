"""Add a frozen Qwen3.5 READER readout to the public Padilla v5 notebook."""

import argparse
import json
from pathlib import Path


FIT = "        refined_lp = post_model.predict_log_probability(base_lp, query_df.text.fillna('').astype(str))\n"
CURRENT = "        current_sub = pd.read_parquet(OUT)\n"
WRITE = "        current_sub['model'] = current_preds\n"

READOUT_CELL = '''\
import signal

reader_output = Path('/kaggle/temp/reader_submission')
reader_logits_path = reader_output / 'reader_qwen35_test.npz'
reader_head = next(Path('/kaggle/input').rglob('reader_qwen35_head.npz'))
if not FAST_COMMIT:
    import torch
    # Preserve the full GLM window and reserve time for the existing Apex postprocessing.
    remaining = NB_T0 + 9 * 3600 - time.time() - 100
    if remaining > 0:
        reader_cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                      f'--nproc_per_node={torch.cuda.device_count()}', 'reader_qwen35.py',
                      '--test', V1_TEST_CSV, '--head', str(reader_head), '--view', READER_VIEW,
                      '--output', str(reader_output)]
        reader_proc = subprocess.Popen(reader_cmd, start_new_session=True,
                                       env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
        try:
            reader_status = reader_proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(reader_proc.pid, signal.SIGKILL)
            reader_status = reader_proc.wait()
        if reader_status:
            reader_logits_path.unlink(missing_ok=True)
        print('READER status:', reader_status, flush=True)
'''

BLEND = '''\
        if reader_logits_path.is_file():
            with np.load(reader_logits_path, allow_pickle=False) as reader_pack:
                reader_labels = list(reader_pack['labels'])
                reader_logits = reader_pack['logits']
                if (np.array_equal(reader_pack['ids'], query_df.id.to_numpy(dtype=str))
                        and reader_logits.shape == refined_lp.shape and set(reader_labels) == set(labels)):
                    reader_logits = reader_logits[:, [reader_labels.index(label) for label in labels]]
                    refined_lp = 0.80 * refined_lp + 0.20 * reader_logits
                    print('Fused READER into Apex before typography', flush=True)
'''

MARGIN_BLEND = '''\
        try:
            import signal
            from reader_margin import rank_reader_rows

            reader_ids = query_df.id.to_numpy(dtype=str)
            if not np.array_equal(current_sub.id.to_numpy(dtype=str), reader_ids):
                raise ValueError('V1 and READER input IDs differ')
            glm_names = ['glm-5.3', 'glm-5.3-flash']
            glm_winners = np.isin(current_sub.model.to_numpy(), glm_names)
            v7_rows = (np.isin(np.asarray(labels)[base_lp.argmax(axis=1)], glm_names)
                       & current_sub.model.eq('claude-fable-5').to_numpy()
                       & READER_RESTORE_FABLE)
            ranked = rank_reader_rows(refined_lp, glm_winners, v7_rows)
            reader_rows = np.sort(ranked[:len(query_df) // 2])
            remaining = NB_T0 + 9 * 3600 - time.time() - 100
            print(f'READER margin route: {len(reader_rows)}/{len(query_df)} passages, {remaining:.0f}s available', flush=True)
            if len(reader_rows) and remaining > 0:
                import torch

                reader_output = Path('/kaggle/temp/reader_margin_submission')
                reader_output.mkdir(parents=True, exist_ok=True)
                reader_input = reader_output / 'test_subset.csv'
                reader_logits_path = reader_output / 'reader_qwen35_test.npz'
                reader_logits_path.unlink(missing_ok=True)
                query_df.iloc[reader_rows][['id', 'text']].to_csv(reader_input, index=False)
                reader_head = next(Path('/kaggle/input').rglob('reader_qwen35_head.npz'))
                reader_cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                              f'--nproc_per_node={torch.cuda.device_count()}', 'reader_qwen35.py',
                              '--test', str(reader_input), '--head', str(reader_head), '--view', READER_VIEW,
                              '--max-tokens', '384', '--output', str(reader_output)]
                remaining = NB_T0 + 9 * 3600 - time.time() - 100
                if remaining > 0:
                    reader_proc = subprocess.Popen(reader_cmd, start_new_session=True,
                                                   env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
                    try:
                        reader_status = reader_proc.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(reader_proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        reader_proc.wait()
                        reader_status = 'timeout'
                    print('READER margin status:', reader_status, flush=True)
                    if reader_status:
                        reader_logits_path.unlink(missing_ok=True)
                    else:
                        with np.load(reader_logits_path, allow_pickle=False) as reader_pack:
                            reader_labels = list(reader_pack['labels'])
                            reader_logits = reader_pack['logits']
                            if (not np.array_equal(reader_pack['ids'], reader_ids[reader_rows])
                                    or reader_logits.shape != (len(reader_rows), len(labels))
                                    or len(reader_labels) != len(labels) or set(reader_labels) != set(labels)
                                    or not np.isfinite(reader_logits).all()):
                                raise ValueError('READER subset scores are invalid or misaligned')
                            columns = [reader_labels.index(label) for label in labels]
                            refined_lp[reader_rows] = 0.80 * refined_lp[reader_rows] + 0.20 * reader_logits[:, columns]
                            print('Fused aligned READER margin subset before typography', flush=True)
        except Exception as reader_error:
            print(f'READER margin fallback to original Apex: {reader_error}', flush=True)
'''

RESTORE = '''\
        starts_glm = np.isin(np.asarray(labels)[base_lp.argmax(axis=1)], ['glm-5.3', 'glm-5.3-flash'])
        restore_fable = starts_glm & current_sub.model.eq('claude-fable-5').to_numpy()
        current_preds[restore_fable] = 'claude-fable-5'
'''


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(original, metadata_path, folder, view, restore_fable, margin_half=False):
    original_notebook = json.loads(original.read_text())
    cells = original_notebook["cells"]
    apex_position = next(i for i, cell in enumerate(cells) if FIT in "".join(cell["source"]))
    apex = "".join(cells[apex_position]["source"])
    if margin_half:
        if apex.count(CURRENT) != 1:
            raise ValueError('Apex no longer has the expected original V1 output read')
        apex = apex.replace(CURRENT, '', 1)
        apex = apex.replace(FIT, FIT + CURRENT + MARGIN_BLEND, 1)
    else:
        apex = apex.replace(FIT, FIT + BLEND, 1)
    if restore_fable:
        if WRITE not in apex:
            raise ValueError("Apex no longer has the expected final assignment")
        apex = apex.replace(WRITE, RESTORE + WRITE, 1)
    cells[apex_position]["source"] = apex.splitlines(keepends=True)
    source = Path(__file__).with_name("reader_qwen35.py").read_text()
    reader_cell = code_cell("%%writefile reader_qwen35.py\n" + source)
    if margin_half:
        helper = Path(__file__).with_name('reader_margin.py').read_text()
        cells[apex_position:apex_position] = [reader_cell, code_cell('%%writefile reader_margin.py\n' + helper),
                                            code_cell(f'READER_VIEW = {view!r}\nREADER_RESTORE_FABLE = {restore_fable!r}\n')]
    else:
        cells[apex_position:apex_position] = [reader_cell, code_cell(f"READER_VIEW = {view!r}\n" + READOUT_CELL)]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    meta = json.loads(metadata_path.read_text())
    if margin_half:
        meta.update(id='zhuating/ntl-0716-reader-qwen35-margin50', title='NTL 0716 Reader Qwen35 Margin50',
                    code_file='ntl_0716_reader_qwen35_margin50.ipynb')
    else:
        meta.update(id="zhuating/ntl-0716-reader-qwen35", title="NTL 0716 Reader Qwen35",
                    code_file="ntl_0716_reader_qwen35.ipynb")
    meta["kernel_sources"] = [source for source in meta["kernel_sources"] if not source.startswith("zhuating/")]
    meta["kernel_sources"].append("zhuating/ntl-reader-qwen35-head")
    reader_model = "shelterw/qwen3.5/Transformers/qwen3.5-9b/1"
    if reader_model not in meta["model_sources"]:
        meta["model_sources"].append(reader_model)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / meta["code_file"]).write_text(json.dumps(original_notebook))
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--view", choices=("dc", "dc_ac"), default="dc_ac")
    parser.add_argument("--restore-fable", action="store_true")
    parser.add_argument('--margin-half', action='store_true', help='score the lowest-margin eligible half of the batch')
    args = parser.parse_args()
    build(args.original, args.metadata, args.folder, args.view, args.restore_fable, args.margin_half)
