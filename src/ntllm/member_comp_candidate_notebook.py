"""Build a competition candidate that points only Comp Qwen3-8B at its pinned weights."""

import argparse
import hashlib
import json
from pathlib import Path

from .atlas_notebook import save
from .glm_prefix_hybrid import cell_index, source
from .member_comp_notebook import pin_qwen8
from .member_trace_notebook import PINNED


KERNEL = 'zhuating/ntl-sibling-odds-comp-candidate'
DEFAULT_OUTPUT = Path('runs/member_evidence_20261125/kaggle_comp_candidate')


def build(output, pinned_path):
    pinned = json.loads(pinned_path.read_text())
    original = json.loads((pinned_path.parent / pinned['code_file']).read_text())
    comp_index = cell_index(original, '%%writefile comp_stack.py')
    cell = original['cells'][comp_index]
    old = source(cell)
    patched = pin_qwen8(old)
    compile(patched.split('\n', 1)[1], 'comp_stack.py', 'exec')
    cell['source'] = patched
    metadata = dict(pinned)
    metadata.pop('id_no', None)
    metadata.update(id=KERNEL, title='NTL Sibling Odds Comp Candidate', code_file='comp-candidate.ipynb',
                    is_private=True, docker_image_pinning_type='original')
    save(output, metadata, original)
    manifest = dict(parent=pinned['id'], original_source_sha256=hashlib.sha256(old.encode()).hexdigest(),
                    changed_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                    edited_cell=comp_index, pinned_inputs=sum(len(metadata[key]) for key in
                        ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources')))
    (output.parent / 'comp_candidate_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pinned', type=Path, default=PINNED)
    args = parser.parse_args()
    build(args.output, args.pinned)
