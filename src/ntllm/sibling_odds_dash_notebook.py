"""Build the Sibling Odds v1 competition variant with just the Gemini em-dash recovery."""

from copy import deepcopy
from pathlib import Path
import argparse
import json

from .atlas_notebook import save
from .glm_prefix_hybrid import replace_once, source


REFERENCE = 'ranjeet258/name-that-llm-sibling-odds'
VARIANT = 'zhuating/ntl-sibling-odds-gemini-dash'
TITLE = 'NTL Sibling Odds Gemini Dash'


def patch(original):
    result = deepcopy(original)
    final, = [cell for cell in result['cells'] if '_typo_fixed = 0' in source(cell)]
    text = replace_once(source(final), '        _typo_fixed = 0\n',
                        '        _before_typography = current_preds.copy()\n'
                        '        _typo_fixed = 0\n')
    text = replace_once(text, "        current_sub['model'] = current_preds\n", """        # Restore only Gemini winners changed by spacing or em-dash typography.
        _restore_gemini = (
            (current_preds != _before_typography)
            & np.char.startswith(_before_typography.astype(str), 'gemini-')
            & query_df.text.fillna('').astype(str).str.contains('—', regex=False).to_numpy()
        )
        current_preds[_restore_gemini] = _before_typography[_restore_gemini]
        print('Restored Gemini em-dash decisions:', int(_restore_gemini.sum()), flush=True)
        current_sub['model'] = current_preds
""")
    compile(text, '<Sibling Odds final decision>', 'exec')
    final['source'] = text
    return result


def build(original, metadata, output):
    notebook = json.loads(original.read_text())
    meta = json.loads(metadata.read_text())
    if meta['id'] != REFERENCE:
        raise ValueError('Expected the RanJeet Sibling Odds v1 metadata')
    meta.pop('id_no', None)
    meta.update(id=VARIANT, title=TITLE, code_file='sibling-odds-gemini-dash.ipynb',
                is_private=True, docker_image_pinning_type='original')
    save(output, meta, patch(notebook))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.original, args.metadata, args.output)
