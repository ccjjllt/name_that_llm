"""Add the cross-family trajectory classifier to the sanpier v2 notebook."""

import argparse
import json
from pathlib import Path


TRAJECTORIES = '/kaggle/temp/author-trajectories'
MODULE = Path(__file__).resolve().parents[1] / 'src/ntllm/sibling_specialist.py'


def replace_once(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError(f'Expected one source block starting with {original!r}')
    return source.replace(original, replacement, 1)


def code_cell(source):
    return {'cell_type': 'code', 'metadata': {}, 'source': source, 'execution_count': None, 'outputs': []}


def attach_specialist(notebook, module_source=None):
    cells = notebook['cells']
    for cell in cells:
        if isinstance(cell['source'], list):
            cell['source'] = ''.join(cell['source'])
    infer_index = next(i for i, cell in enumerate(cells) if cell['source'].startswith('%%writefile infer.py'))
    infer_cell = cells[infer_index]
    infer = infer_cell['source']
    infer = replace_once(
        infer,
        '    def featurize(enc_, norms, hid):\n',
        '    def featurize(enc_, norms, hid, trajectory_split=None):\n'
        '        if trajectory_split is not None:\n'
        '            from sibling_specialist import pool_trajectory\n'
        '            trajectory_depths = (4, 12, 20, 24)\n'
        '            trajectory = np.empty((len(enc_), 2 * len(trajectory_depths) * hid), dtype=np.float16)\n',
    )
    infer = replace_once(
        infer,
        '                mask = att.unsqueeze(-1).float()\n',
        '                if trajectory_split is not None:\n'
        '                    trajectory[b] = pool_trajectory(hs, att, trajectory_depths)\n'
        '                mask = att.unsqueeze(-1).float()\n',
    )
    infer = replace_once(
        infer,
        '        return Z_\n',
        '        if trajectory_split is not None:\n'
        f'            os.makedirs({TRAJECTORIES!r}, exist_ok=True)\n'
        f'            np.save(os.path.join({TRAJECTORIES!r}, spec["name"] + "_" + trajectory_split + "_trajectory.npy"), trajectory)\n'
        '        return Z_\n',
    )
    infer = replace_once(
        infer,
        '        Z = featurize(enc, norms, hid)\n'
        '        if train_texts is not None and ai == len(adapters) - 1:\n'
        '            FEATS = (Z.copy(), featurize(enc_tr, norms, hid))\n',
        '        gaussian_pass = train_texts is not None and ai == len(adapters) - 1\n'
        '        collect_trajectory = gaussian_pass and spec["name"] in {"q8v5", "q14v5"}\n'
        '        Z = featurize(enc, norms, hid, "test" if collect_trajectory else None)\n'
        '        if gaussian_pass:\n'
        '            Ztr = featurize(enc_tr, norms, hid, "train" if collect_trajectory else None)\n'
        '            FEATS = Z.copy(), Ztr\n',
    )
    infer_cell['source'] = infer
    specialist = MODULE.read_text() if module_source is None else module_source
    cells.insert(infer_index, code_cell('%%writefile sibling_specialist.py\n' + specialist))
    final = next(cell for cell in cells if 'final=apply_postprocessing(' in cell['source'])
    final['source'] = replace_once(
        final['source'],
        '                              strength=0.1, corrected=False, penalty=0.05)\n',
        '                              strength=0.1, corrected=False, penalty=0.05)\n'
        '    from sibling_specialist import refine_submission\n'
        '    final=refine_submission(train_df,query_df,OUT,V1_OUT/"post-glm.npz",'
        f'{TRAJECTORIES!r})\n',
    )
    return notebook


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('source', type=Path)
    cli.add_argument('destination', type=Path)
    arguments = cli.parse_args()
    patched = attach_specialist(json.loads(arguments.source.read_text()))
    arguments.destination.write_text(json.dumps(patched, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
