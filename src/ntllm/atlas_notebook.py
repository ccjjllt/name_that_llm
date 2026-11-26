"""Common helpers for building Kaggle notebooks."""

import json


def cell(text):
    compile(text, '<atlas notebook cell>', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': text}


def save(output, metadata, notebook):
    output.mkdir(parents=True, exist_ok=True)
    (output / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + '\n')
