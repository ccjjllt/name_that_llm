"""Package the independent Prvsiyan readout comparison on Kaggle CPU."""

import argparse
import base64
from pathlib import Path

from .atlas_notebook import cell, save
from .heldout_source_audit import KIMI_EVAL_INDEX


OUTPUT = Path('runs/heldout_source_20261125/kaggle_reader_standalone')
SOURCE = 'zhuating/ntl-heldout-source-audit/2'
READERS = 'zhuating/ntl-provenance-readers-heldout/1'


RUN = '''import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from ntllm.member_evidence_transfer import read_prvsiyan, verify_manifest
from ntllm.peer_context_report import paired_stats, read_ids, same_ids

ARMS = ('length_matched', 'provenance_balanced', 'equal_probability_mean')
LEVELS = ('exact', 'same_family_wrong_variant', 'wrong_family')
DRAWS = 3000


def family(model):
    return 'qwen' if model.startswith('qwen') else model.split('-', 1)[0]


def inspect_arm(logits, truth, labels, name):
    values = read_ids(logits / f'{name}_logits.parquet')
    same_ids(values.index, truth.index, name, ordered=True)
    if list(values.columns) != labels or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f'Expected the original finite 18-class logits in {name}')
    return softmax(values.to_numpy(dtype=np.float64), axis=1)


def add_predictions(rows, labels, name, probabilities):
    ranks = np.argsort(-probabilities, axis=1, kind='stable')[:, :3]
    classes = np.asarray(labels)[ranks]
    families = np.asarray([family(label) for label in labels])[ranks]
    choices = classes[:, 0]
    rows[name] = choices
    rows[f'{name}_family'] = families[:, 0]
    rows[f'{name}_confidence'] = probabilities.max(1)
    rows[f'{name}_top3'] = [list(row) for row in classes]
    rows[f'{name}_true_top3'] = (classes == rows.model.to_numpy()[:, None]).any(1)
    rows[f'{name}_family_top3'] = (families == rows.true_family.to_numpy()[:, None]).any(1)


def transitions(panel, before, after):
    def state(arm):
        return np.select([panel[arm].eq(panel.model).to_numpy(dtype=bool),
                          panel[f'{arm}_family'].eq(panel.true_family).to_numpy(dtype=bool)],
                         LEVELS[:2], default=LEVELS[2])

    table = pd.crosstab(state(before), state(after)).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    return table.to_dict(orient='index')


def summary(panel):
    def coverage(group):
        length, provenance = [group[f'{arm}_true_top3'] for arm in ARMS[:2]]
        family_length, family_provenance = [group[f'{arm}_family_top3'] for arm in ARMS[:2]]
        return dict(
            exact_top3={arm: int(group[f'{arm}_true_top3'].sum()) for arm in ARMS},
            family_top3={arm: int(group[f'{arm}_family_top3'].sum()) for arm in ARMS},
            exact_union=int((length | provenance).sum()), family_union=int((family_length | family_provenance).sum()),
            provenance_only_exact=int((provenance & ~length).sum()), length_only_exact=int((length & ~provenance).sum()),
        )

    by_class = {}
    for model, group in panel.groupby('model'):
        correct = {arm: int(group[arm].eq(model).sum()) for arm in ARMS}
        by_class[model] = dict(rows=len(group), correct=correct, recall={arm: count / len(group) for arm, count in correct.items()},
                               family_correct={arm: int(group[f'{arm}_family'].eq(group.true_family).sum()) for arm in ARMS},
                               top3=coverage(group))
    family_panel = panel.assign(model=panel.true_family)
    comparisons = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))
    paired = [dict(exact=paired_stats(panel, before, after, DRAWS),
                   family=paired_stats(family_panel, before + '_family', after + '_family', DRAWS),
                   level_transitions=transitions(panel, before, after)) for before, after in comparisons]
    disagree = panel[panel.length_matched.ne(panel.provenance_balanced)]
    conf_prefer = disagree.provenance_balanced_confidence.gt(disagree.length_matched_confidence)
    conf_tied = disagree.provenance_balanced_confidence.eq(disagree.length_matched_confidence)
    chosen = disagree.provenance_balanced.where(conf_prefer, disagree.length_matched)
    return dict(rows=len(panel), prompts=int(panel.prompt_id.nunique()), represented_labels=len(by_class),
                top3=coverage(panel), paired=paired, by_true_class=by_class,
                disagreement=dict(rows=len(disagree), exact_when_more_confident=int((chosen.eq(disagree.model) & ~conf_tied).sum()),
                                  confidence_ties=int(conf_tied.sum())))


def asset(slug, name):
    files = [path for path in Path('/kaggle/input').rglob(name) if slug in path.parts]
    if len(files) != 1:
        raise ValueError(f'Expected one {name} from {slug}, found {files}')
    return files[0]


source = asset('ntl-heldout-source-audit', 'candidate_truth.parquet').parent
readers = asset('ntl-provenance-readers-heldout', 'provenance_balanced_logits.parquet').parent
truth = read_prvsiyan(source, Path('kimi_panel_index.parquet'))
verify_manifest(readers / 'manifest.json', truth)
labels = sorted(truth.model.unique())
vectors = {arm: inspect_arm(readers, truth, labels, arm) for arm in ARMS[:2]}
vectors[ARMS[2]] = (vectors[ARMS[0]] + vectors[ARMS[1]]) / 2.
rows = truth[['model', 'prompt_id', 'n_words', 'strict_topic_holdout']].assign(true_family=truth.model.map(family))
for arm, probabilities in vectors.items():
    add_predictions(rows, labels, arm, probabilities)
rows['true_class_top3_union'] = rows.length_matched_true_top3 | rows.provenance_balanced_true_top3
rows['true_family_top3_union'] = rows.length_matched_family_top3 | rows.provenance_balanced_family_top3
panels = {'prvsiyan_895': rows, 'prvsiyan_316': rows.loc[rows.strict_topic_holdout]}
manifest = json.loads((readers / 'manifest.json').read_text())
report = dict(manifest={key: manifest[key] for key in ('rows', 'ordered_ids_sha256')},
              ensemble='Unfitted equal mean of the two 18-class probability vectors.',
              scope='Standalone classification. Same-base fusion and candidate acceptance are measured separately.',
              panels={name: {length: summary(part) for length, part in [('all', panel), ('short', panel[panel.n_words.lt(100)])]}
                      for name, panel in panels.items()})
output = Path('/kaggle/working/reader_standalone')
output.mkdir(parents=True, exist_ok=True)
rows.to_parquet(output / 'predictions.parquet')
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\\n')
print('Verified reader alignment and scored', len(rows), 'rows across', len(labels), 'labels', flush=True)
'''


def build(output=OUTPUT):
    setup = ("import base64\nfrom pathlib import Path\nPath('ntllm').mkdir(exist_ok=True)\n"
             "Path('ntllm/__init__.py').touch()\n")
    for module in ('member_evidence_transfer', 'member_evidence', 'peer_context_report', 'sibling_odds_replay'):
        code = Path(__file__).with_name(module + '.py').read_text()
        compile(code, f'<embedded {module}>', 'exec')
        setup += f"Path('ntllm/{module}.py').write_text({code!r})\n"
    index = base64.b64encode(Path(KIMI_EVAL_INDEX).read_bytes()).decode()
    setup += f"Path('kimi_panel_index.parquet').write_bytes(base64.b64decode({index!r}))\n"
    metadata = dict(id='zhuating/ntl-reader-standalone', title='NTL Reader Standalone',
                    code_file='reader-standalone.ipynb', language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=False, enable_internet=False, kernel_sources=[SOURCE, READERS])
    notebook = dict(cells=[cell(setup), cell(RUN)], metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=2)
    save(output, metadata, notebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
