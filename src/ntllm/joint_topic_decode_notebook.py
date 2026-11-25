"""Package the CPU joint decoding diagnostic using already archived text affinities."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save


AFFINITY = Path('runs/peer_retrieval_20261124/lexical/similarity.npz')
OUTPUT = Path('runs/joint_topic_20261125/kaggle_audit')


def build(output, affinity):
    modules = ('joint_topic_decode', 'peer_collision', 'peer_context_report', 'sibling_odds_replay')
    setup = ("import base64\nfrom pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in modules:
        code = Path(__file__).with_name(name + '.py').read_text()
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    payload = base64.b64encode(affinity.read_bytes()).decode()
    setup += f"Path('lexical_affinity.npz').write_bytes(base64.b64decode({payload!r}))\n"
    run = """
from ntllm.joint_topic_decode import run

mount = Path('/kaggle/input')
trace, = [p.parent for p in mount.rglob('public_truth.parquet') if (p.parent / 'stage_scores.npz').is_file()]
run(trace, Path('lexical_affinity.npz'), Path('/kaggle/working/joint_topic'))
"""
    for code in (setup, run):
        compile(code, '<joint-topic>', 'exec')
    metadata = dict(id='zhuating/ntl-joint-topic-decode', title='NTL Joint Topic Decode', code_file='joint-topic.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=['zhuating/ntl-0719-public-decision-trace/1'])
    notebook = {'cells': [cell(setup), cell(run)], 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--affinity', type=Path, default=AFFINITY)
    args = parser.parse_args()
    build(args.output, args.affinity)
