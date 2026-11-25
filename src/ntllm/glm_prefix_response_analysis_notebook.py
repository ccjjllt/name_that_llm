"""Build the offline CPU report over the frozen GLM prefix scoring run."""

import argparse
import base64
import io
import json
from pathlib import Path
import re
import tarfile

from .atlas_notebook import save
from .glm_prefix_response_notebook import PINNED, PLAN, SCORES, notebook, source


OUTPUT = Path('runs/glm_prefix_response_20261125/kaggle_analysis')


def build(output, pinned, panel_version, scores_version):
    original = json.loads((pinned.parent / json.loads(pinned.read_text())['code_file']).read_text())
    capsule, = [match[1] for cell in original['cells']
                if (match := re.search(r'^V1_PAYLOAD = "([A-Za-z0-9+/=]+)"$', source(cell), re.M))]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(capsule)), mode='r:gz') as archive:
        features = archive.extractfile('features.py').read().decode()
        settings = archive.extractfile('ll_model.json').read().decode()
    setup = ("from pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n"
             f"Path('prefix_response_features.py').write_text({features!r})\n"
             f"Path('prefix_reference.json').write_text({settings!r})\n")
    for name in ('glm_prefix_response_oof', 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    run = """
from ntllm.glm_prefix_response_oof import run

mount = Path('/kaggle/input')
panel, = [path.parent for path in mount.rglob('truth.parquet') if path.parent.name == 'prefix_response_panel']
scores, = [path.parent for path in mount.rglob('flash_scores.npz')
           if path.parent.name == 'prefix_response_scores']
run(panel, scores, Path('prefix_reference.json'), Path('/kaggle/working/prefix_response_analysis'))
"""
    for code in (setup, run):
        compile(code, '<prefix-response-analysis>', 'exec')
    metadata = {
        'id': 'zhuating/ntl-glm-prefix-response-analysis',
        'title': 'NTL GLM Prefix Response Analysis',
        'code_file': 'prefix-response-analysis.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': False,
        'enable_internet': False,
        'kernel_sources': [f'{PLAN}/{panel_version}', f'{SCORES}/{scores_version}'],
    }
    save(output, metadata, notebook(setup, run))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--parent', type=Path, default=PINNED)
    parser.add_argument('--panel-version', type=int, default=1)
    parser.add_argument('--scores-version', type=int, default=1)
    args = parser.parse_args()
    build(args.output, args.parent, args.panel_version, args.scores_version)
