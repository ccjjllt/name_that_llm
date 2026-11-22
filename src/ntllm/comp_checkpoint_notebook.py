"""Generate the public 0.716 notebook with the tested Comp checkpoint filter."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

if __package__:
    from .glm_prefix_hybrid import cell_index, replace_once, source
else:
    from glm_prefix_hybrid import cell_index, replace_once, source


NOTEBOOK_FILE = 'ntl_0716_comp_repaired.ipynb'
GPT56_ASSET = 'zhuating/ntl-frozen-native-sibling-heads'
ORIGINAL_MATCH = (
    "        if dir_name or (c.get('model_type') == model_type"
    " and (hidden is None or c.get('hidden_size') == hidden)):\n"
)
WEIGHT_MATCH = (
    "        if dir_name or (c.get('model_type') == model_type"
    " and (hidden is None or c.get('hidden_size') == hidden)"
    " and 'embedding' not in cfg.lower() and 'gguf' not in cfg.lower()"
    " and (glob.glob(os.path.join(os.path.dirname(cfg), '*.safetensors'))"
    " or glob.glob(os.path.join(os.path.dirname(cfg), 'pytorch_model*.bin')))):\n"
)


def repaired_notebook(notebook):
    result = deepcopy(notebook)
    cell = result['cells'][cell_index(result, '%%writefile comp_stack.py\n')]
    original = source(cell)
    start = original.index('def find_checkpoint(')
    end = original.index('def load_truncated_decoder(', start)
    cell['source'] = original[:start] + replace_once(original[start:end], ORIGINAL_MATCH, WEIGHT_MATCH) + original[end:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--metadata-template', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    notebook = repaired_notebook(json.loads(args.base.read_text(encoding='utf-8')))
    metadata = json.loads(args.metadata_template.read_text(encoding='utf-8'))
    metadata['kernel_sources'].remove(GPT56_ASSET)
    metadata.update(id='zhuating/ntl-0716-comp-repaired', title='NTL 0716 Comp Repaired', code_file=NOTEBOOK_FILE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, document in ((args.output_dir / NOTEBOOK_FILE, notebook),
                           (args.output_dir / 'kernel-metadata.json', metadata)):
        path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
