"""Package the fixed Lightning versus Kiro PB115 factorial on Kaggle CPU."""

import argparse
import hashlib
import json
from pathlib import Path

from .atlas_notebook import cell, save


OUTPUT = Path('runs/lightning_pb115_20261125/kaggle_audit')
SOURCE = Path('runs/research_frontier_20261125/lightning_0724')
SOURCE_SHA256 = 'a5f6addedffab0d750cc2ef20fbe51af1dd6b7851343525512a3896dcea12983'
RESIDUAL_SHA256 = 'ea2c6bd5a9c15446f719f66b1af19f6f70a03b88c641b76f0e9c98ec9c4f5624'
FEATURES = 'lumibee/ntl-prefixb-full-115/1'
NATIVE = 'zhuating/ntl-glm-likelihood-dynamics/2'
KIRO = 'zhuating/ntl-kiro-pb115-oof/1'


def extract_residual(directory):
    metadata = json.loads((directory / 'kernel-metadata.json').read_text())
    verification = json.loads((directory / 'verified_source.json').read_text())
    original = (directory / metadata['code_file']).read_bytes()
    if (metadata['id'].split('/', 1)[0] != 'lightningv08' or verification['script_version_id'] != 373716275
            or hashlib.sha256(original).hexdigest() != SOURCE_SHA256):
        raise ValueError('Expected the exact downloaded Lightning v53 scoring source')
    marker = '%%writefile pngy_residual.py\n'
    sources = [''.join(item['source']) for item in json.loads(original)['cells'] if item['cell_type'] == 'code']
    body, = [source.removeprefix(marker) for source in sources if source.startswith(marker)]
    if hashlib.sha256(body.encode()).hexdigest() != RESIDUAL_SHA256:
        raise ValueError('Lightning sibling refinement differs from the verified source cell')
    compile(body, '<Lightning pngy_residual>', 'exec')
    return body


def build(output=OUTPUT, source=SOURCE):
    setup = ("from pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for name in ('lightning_pb115_factorial', 'kiro_pb115_oof', 'member_comp_replay',
                 'glm_prefix_response_replay', 'peer_context_report', 'sibling_odds_replay'):
        body = Path(__file__).with_name(name + '.py').read_text()
        compile(body, f'<embedded {name}>', 'exec')
        setup += f"Path('ntllm/{name}.py').write_text({body!r})\n"
    setup += f"Path('pngy_residual.py').write_text({extract_residual(source)!r})\n"
    run = '''from threadpoolctl import threadpool_limits
from ntllm.lightning_pb115_factorial import run

mount = Path('/kaggle/input')

def asset(slug, name):
    matches = [path for path in mount.rglob(name) if slug in path.parts]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} from {slug}, found {matches}')
    return matches[0]

official = asset('name-that-llm', 'train.csv')
native = asset('ntl-glm-likelihood-dynamics', 'glm_dynamics_oof.parquet')
features = asset('ntl-prefixb-full-115', 'paired_token_features.parquet').parent
reference = asset('ntl-kiro-pb115-oof', 'predictions.parquet')
if reference.parent.name != 'kiro_pb115_oof':
    raise ValueError('Expected the original frozen Kiro PB115 comparison file')
with threadpool_limits(limits=2, user_api='blas'):
    run(official, native, features, reference, Path('/kaggle/working/lightning_pb115_factorial'))
'''
    metadata = dict(id='zhuating/ntl-lightning-pb115-factorial', title='NTL Lightning PB115 Factorial',
                    code_file='lightning-pb115-factorial.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    competition_sources=['name-that-llm'], dataset_sources=[FEATURES], kernel_sources=[NATIVE, KIRO])
    notebook = dict(cells=[cell(setup), cell(run)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    build(args.output, args.source)
