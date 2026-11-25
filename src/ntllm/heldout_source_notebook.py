"""Package the public-source overlap audit to run on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import DATASETS, EVAL_FILES, KIMI_EVAL_INDEX


KERNEL = 'zhuating/ntl-heldout-source-audit'
OUTPUT = Path('runs/heldout_source_20261125/kaggle_audit')


def build(output):
    script = Path(__file__).with_name('heldout_source_audit.py').read_text()
    setup = "import base64\nfrom pathlib import Path\n"
    setup += f"Path('heldout_source_audit.py').write_text({script!r})\n"
    setup += 'prior = []\n'
    for number, path in enumerate(EVAL_FILES):
        name = f"earlier_queries_{number}.parquet"
        payload = base64.b64encode(Path(path).read_bytes()).decode()
        setup += f"prior.append(Path({name!r}))\nprior[-1].write_bytes(base64.b64decode({payload!r}))\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    process = """
from pathlib import Path

from heldout_source_audit import run

run(Path('/kaggle/input'), prior, Path('/kaggle/working/source_audit'), Path('kimi_panel_index.parquet'))
"""
    for source in (setup, process):
        compile(source, '<heldout-source-audit>', 'exec')
    metadata = dict(
        id=KERNEL, title='NTL Heldout Source Audit', code_file='heldout-sources.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_internet=False, enable_gpu=False, enable_tpu=False,
        dataset_sources=list(DATASETS), competition_sources=['name-that-llm'],
        kernel_sources=['zhuating/ntl-0719-public-decision-trace/1'], model_sources=[],
    )
    notebook = {'cells': [cell(setup), cell(process)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 2}
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    build(parser.parse_args().out)
