"""Package the four-GPT topic-centering experiment for private Kaggle CPU."""

import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .gpt_topic_orthogonal import ENAI, PRV
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/gpt_topic_orthogonal_20261125/kaggle')
KERNEL = 'zhuating/ntl-gpt-topic-orthogonal-audit'
PINS = ('zhuating/ntl-0719-public-decision-trace/1', 'zhuating/ntl-sibling-odds-member-trace/1',
        'zhuating/ntl-prvsiyan-member-trace/1', 'zhuating/ntl-heldout-source-audit/2')


def build():
    setup = ("import base64\nfrom pathlib import Path\n"
             "Path('ntllm').mkdir(exist_ok=True)\nPath('ntllm/__init__.py').touch()\n")
    for name in ('gpt_topic_orthogonal', 'heldout_source_audit', 'matched_request_supply', 'member_evidence',
                 'member_evidence_transfer', 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(name + '.py').read_text()
        compile(code, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({code!r})\n"
    encoded = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('prior_kimi.parquet').write_bytes(base64.b64decode({encoded!r}))\n"
    execute = '''from threadpoolctl import threadpool_limits
from ntllm.gpt_topic_orthogonal import run

with threadpool_limits(limits=4):
    run(Path('/kaggle/input'), Path('prior_kimi.parquet'), Path('/kaggle/working/gpt_topic_orthogonal'))
'''
    metadata = dict(id=KERNEL, title='NTL GPT Topic Orthogonal Audit', code_file='gpt-topic-orthogonal.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, competition_sources=['name-that-llm'],
                    dataset_sources=[ENAI, PRV], kernel_sources=list(PINS))
    notebook = dict(cells=[cell(setup), cell(execute)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(OUTPUT, metadata, notebook)


if __name__ == '__main__':
    build()
