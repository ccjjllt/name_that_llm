"""Package the 96-passage StyleDistance precision timing on Kaggle CPU."""

import argparse
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/style_precision_20261125/kaggle_audit')
WEIGHTS = 'zhuating/ntl-styledistance-weights/2'


def build(output):
    source = Path(__file__).with_name('style_precision.py').read_text()
    compile(source, '<precision benchmark module>', 'exec')
    setup = cell(f"from pathlib import Path\nPath('style_precision.py').write_text({source!r})\n")
    run = cell('''import os
import subprocess
import sys

mount = Path('/kaggle/input')
official, = [p for p in mount.rglob('train.csv') if p.parent.name == 'name-that-llm']
encoders, = [p for p in mount.rglob('encoders') if 'ntl-styledistance-weights' in p.parts
             and (p / 'style/config.json').is_file()]
output = Path('/kaggle/working/style_precision')
env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
command = ('from pathlib import Path; from style_precision import run; '
           f'run(Path({str(official)!r}), Path({str(encoders / "style")!r}), Path({str(output)!r}))')
subprocess.run([sys.executable, '-u', '-c', command], env=env, check=True)
''')
    metadata = dict(id='zhuating/ntl-style-precision', title='NTL Style Precision',
                    code_file='style-precision.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'], kernel_sources=[WEIGHTS])
    notebook = dict(cells=[setup, run], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
