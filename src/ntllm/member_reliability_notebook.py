"""Package frozen group-level member reliability on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .candidate_choice_notebook import MODULES
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/member_reliability_20261125/kaggle')
KERNEL = 'zhuating/ntl-member-reliability-transfer'
PINS = ('zhuating/ntl-sibling-odds-member-trace/1', 'zhuating/ntl-0719-public-decision-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2',
        'zhuating/ntl-lumibee-typography-transfer/1')


def build(output):
    modules = ('member_reliability', *[name for name in MODULES if name != 'candidate_choice'])
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for name in modules:
        source = Path(__file__).with_name(name + '.py').read_text()
        compile(source, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({source!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    execute = """from threadpoolctl import threadpool_limits
from ntllm.member_reliability import run

def asset(slug, file):
    result, = [path for path in Path('/kaggle/input').rglob(file) if slug in path.parts]
    return result

with threadpool_limits(limits=2):
    run(asset('ntl-sibling-odds-member-trace', 'member_scores.npz').parent,
        asset('ntl-0719-public-decision-trace', 'public_truth.parquet').parent,
        asset('ntl-prvsiyan-member-trace', 'member_scores.npz').parent,
        asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent,
        Path('kimi_panel_index.parquet'),
        asset('ntl-lumibee-typography-transfer', 'predictions.parquet'),
        Path('/kaggle/working/member_reliability'))
"""
    for source in (setup, execute):
        compile(source, '<member reliability notebook>', 'exec')
    metadata = dict(id=KERNEL, title='NTL Member Reliability Transfer', code_file='member-reliability.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, kernel_sources=list(PINS))
    save(output, metadata, dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
