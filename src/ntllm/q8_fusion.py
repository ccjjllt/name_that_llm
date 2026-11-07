"""Select a small fixed fusion family on inner prompts, then lock it for outer use.

Existing Q8 outer OOF may be diagnosed, but cannot masquerade as these nested
selection inputs: its Gaussian fit and backbone usually already saw inner dev.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from scipy.optimize import minimize_scalar

from .q8_data import file_sha
from .q8_inner import validate_training_evidence, content_identity, validate_split_manifest


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inference_core_options(options):
    """Stress membership may vary; grouping and Gaussian inference must stay fixed."""
    return {key: value for key, value in options.items()
            if key not in ('seeds', 'eval_cohorts', 'imbalance_suite')}


def candidate_recipes(allow_exclusion=False):
    """Eleven preregistered weights; no outer-data search or per-cohort tuning."""
    branch = 'gaussian' if allow_exclusion else 'gaussian_noninjective'
    values = [dict(name='plain', p=1., c=0., g=0.), dict(name='centered_fallback', p=0., c=1., g=0.),
              dict(name=branch+'_fallback', p=0., c=0., g=1.)]
    for c in (0., .35, 1.):
        for g in (0., .22, 1.):
            if c == 0 and g == 0:
                continue
            values.append(dict(name=f'p_c{c:g}_g{g:g}', p=1., c=c, g=g))
    for value in values:
        value.update(gaussian_branch=branch, allow_class_exclusion=bool(allow_exclusion),
                     inactive_rule='Use plain when no requested branch is active; inactive relation contributions are zero.')
    return values


def mix_branches(branches, recipe):
    p = np.asarray(branches['plain_lp'], dtype=float)
    c_active = np.asarray(branches['centered_active'], dtype=bool)
    g_active = np.asarray(branches['gaussian_active'], dtype=bool)
    for name, weight in (("centered_lp",recipe['c']),(recipe['gaussian_branch']+'_lp',recipe['g'])):
        if weight and name not in branches:
            raise ValueError('A selected fusion branch is unavailable: '+name)
    c = np.asarray(branches.get('centered_lp', np.zeros_like(p)), dtype=float)
    g = np.asarray(branches.get(recipe['gaussian_branch']+'_lp', np.zeros_like(p)), dtype=float)
    if p.ndim != 2 or p.shape[1] != 18 or c.shape != p.shape or g.shape != p.shape:
        raise ValueError('Fusion branches must be aligned 18-class log probabilities.')
    score = recipe['p']*p + recipe['c']*np.where(c_active[:,None],c,0) + recipe['g']*np.where(g_active[:,None],g,0)
    active_mass = recipe['p'] + recipe['c']*c_active + recipe['g']*g_active
    score[active_mass == 0] = p[active_mass == 0]
    if not np.isfinite(score).all():
        raise ValueError('Fusion received nonfinite branches.')
    return log_softmax(score, axis=1)


def _metrics(rows, lp, labels):
    y = rows.model.map({label:i for i,label in enumerate(labels)}).to_numpy()
    predicted = np.asarray(lp).argmax(1)
    recalls = {label: float(np.mean(predicted[y==i]==i)) for i,label in enumerate(labels) if np.any(y==i)}
    return dict(rows=len(rows), represented_classes=len(recalls), balanced_accuracy=float(np.mean(list(recalls.values()))),
                balanced_accuracy_18=float(np.mean(list(recalls.values()))) if len(recalls)==18 else None,
                nll=float(-lp[np.arange(len(y)),y].mean()), per_class_recall=recalls)


def _probability_metrics(rows, lp, labels):
    """Probability quality is separate from the BA used to choose weights."""
    result = _metrics(rows, lp, labels)
    y = rows.model.map({label:i for i,label in enumerate(labels)}).to_numpy()
    loss = -lp[np.arange(len(y)), y]
    result['balanced_nll'] = float(np.mean([loss[y == i].mean() for i in np.unique(y)]))
    result['brier'] = float(np.square(np.exp(lp) - np.eye(len(labels))[y]).sum(1).mean())
    return result


def _calibration_spec():
    return dict(objective='balanced NLL over represented classes in the branch-active full inner dev rows',
                temperature_bounds=[.05,20.], optimizer='bounded scalar minimization of log temperature',
                log_temperature_tolerance=1e-6, maximum_iterations=200,
                scope='One scalar each for plain, centered, gaussian, gaussian_noninjective; no class/group/layout-specific parameters.',
                inactive_rule='Fit relation temperatures on active rows only; use calibrated plain for unavailable standalone relations.',
                gaussian_scope='Post-hoc scaling of cached Gaussian log probabilities; internal Gaussian inference temperatures remain fixed.',
                weight_selection='The unchanged eleven-candidate family is selected only after all independent temperatures are frozen.')


def _branch_active(branches, name):
    n = len(branches['plain_lp'])
    active = (np.ones(n,dtype=bool) if name == 'plain' else
              np.asarray(branches['centered_active' if name == 'centered' else 'gaussian_active'],dtype=bool))
    values = np.asarray(branches[name+'_lp'],dtype=float)
    if active.shape != (n,) or values.shape != (n,18) or not np.isfinite(values).all():
        raise ValueError('Temperature calibration requires aligned finite branch scores and activity masks.')
    return active


def _validate_calibration(calibration):
    if isinstance(calibration,(str,Path)):
        calibration = json.loads(Path(calibration).read_text())
    if (calibration.get('policy') != 'independent_branch_temperature_v1' or
            calibration.get('specification') != _calibration_spec() or
            calibration.get('calibration_sha256') != _digest({k:v for k,v in calibration.items() if k!='calibration_sha256'})):
        raise ValueError('Calibration is not an unchanged declared independent-temperature fit.')
    if set(calibration['branches']) != {'plain','centered','gaussian','gaussian_noninjective'}:
        raise ValueError('All four branches need explicit calibration evidence.')
    for branch in calibration['branches'].values():
        temperature = branch['temperature']
        if not np.isfinite(temperature) or not .05 <= temperature <= 20.:
            raise ValueError('A branch temperature is outside the declared positive interval.')
    return calibration


def apply_branch_calibration(branches, calibration):
    """No labels are needed to apply an already frozen branch calibration."""
    calibration = _validate_calibration(calibration)
    result = dict(branches)
    for name,record in calibration['branches'].items():
        active = _branch_active(branches,name)
        values = np.zeros_like(np.asarray(branches[name+'_lp'],dtype=float))
        values[active] = log_softmax(np.asarray(branches[name+'_lp'])[active]/record['temperature'],axis=1)
        result[name+'_lp'] = values
    return result


def _standalone_recipes():
    return dict(plain=dict(p=1,c=0,g=0,gaussian_branch='gaussian'),
                centered=dict(p=0,c=1,g=0,gaussian_branch='gaussian'),
                gaussian=dict(p=0,c=0,g=1,gaussian_branch='gaussian'),
                gaussian_noninjective=dict(p=0,c=0,g=1,gaussian_branch='gaussian_noninjective'))


def _source_group_scores(rows, lp, labels, source_intervention):
    if not source_intervention or source_intervention.get('status') != 'available':
        return {}
    from .q8_evaluation import score_intervention_groups
    return score_intervention_groups(rows, labels, np.exp(lp), source_intervention)


def _standalone_report(rows, branches, calibrated, labels, source_intervention=None, evaluation=None):
    result = {name:dict(raw=_probability_metrics(rows,mix_branches(branches,recipe),labels),
                      calibrated=_probability_metrics(rows,mix_branches(calibrated,recipe),labels),
                      active_rows=int(_branch_active(branches,name).sum()),
                      fallback_rows=int((~_branch_active(branches,name)).sum()))
            for name,recipe in _standalone_recipes().items()}
    evaluation = evaluation or {}
    for name, record in result.items():
        # Availability is checkpoint-level; an available relation branch can
        # still have no active rows in a singleton-only evaluation cohort.
        available = evaluation.get('branch_availability', {}).get(name)
        if available is None and record['active_rows']:
            available = True
        record.update(available=available,
                      score_scope=('plain' if name == 'plain' else
                                   'branch_with_plain_fallback' if record['active_rows'] else
                                   'plain_fallback_only'))
    centered = result['centered']
    trained = evaluation.get('head_cache', {}).get(
        'centered_trained', evaluation.get('branch_availability', {}).get('centered'))
    centered.update(trained=trained,
                    training_evidence=evaluation.get('provenance', {}).get('centered_training_evidence'))
    if not centered['active_rows']:
        centered['interpretation'] = (
            'No effective centered training is recorded. These values are entirely plain fallbacks, not centered-head quality.'
            if trained is False else
            'This cohort has no active centered rows. These values are entirely plain fallbacks, not centered-head quality.')
    else:
        centered['interpretation'] = (
            'Centered scores on active relation rows and plain fallback elsewhere; this is not an isolated estimate of the centered training benefit.')
    if not source_intervention or source_intervention.get('status') != 'available':
        return result
    for name, recipe in _standalone_recipes().items():
        for scale, values in (('raw', branches), ('calibrated', calibrated)):
            groups = _source_group_scores(rows, mix_branches(values, recipe), labels, source_intervention)
            if groups:
                result[name][scale]['by_source_intervention_label_coverage'] = groups
    return result


def _load_cohorts(directory, split, role):
    validate_split_manifest(split)
    directory = Path(directory)
    index = json.loads((directory/'evaluation.json').read_text())
    if not all((directory/name).is_file() for name in ('fit_rows.parquet','query_rows.parquet')):
        raise ValueError('This cache lacks explicit fit/query membership artifacts; ordinary outer OOF cannot establish inner holdout.')
    if index['labels'] != split['labels']:
        raise ValueError('Fusion labels differ from the split.')
    if index.get('provenance',{}).get('split_sha256') != split['split_sha256'] or index.get('provenance',{}).get('role') != role:
        raise ValueError('Evaluation provenance does not bind this split and held-out role.')
    if index['options']['grouping_protocol'] != 'train_fold':
        raise ValueError('The nested protocol fixes grouping transforms on the fit side.')
    fit = pd.read_parquet(directory/'fit_rows.parquet')
    if not fit.id.is_unique or set(fit.id.astype(str)) != set(split['inner_fit']['ids']):
        raise ValueError('Gaussian/grouping fit rows are not the exact inner fit set. Ordinary outer OOF is not nested calibration.')
    if content_identity(fit.drop(columns='fold')) != split['inner_fit']['content_sha256']:
        raise ValueError('Gaussian/grouping fit contents changed.')
    query = pd.read_parquet(directory/'query_rows.parquet')
    if not query.id.is_unique or set(query.id.astype(str)) != set(split[role]['ids']):
        raise ValueError('Evaluation rows do not match the declared split role.')
    if content_identity(query.drop(columns='fold')) != split[role]['content_sha256']:
        raise ValueError('Held-out evaluation text, prompt or label changed.')
    cohorts = {}
    for name, entry in index['cohorts'].items():
        if not entry.get('available'):
            continue
        folder = directory/name
        rows = pd.read_parquet(folder/'rows.parquet')
        if not rows.id.is_unique or not set(rows.id.astype(str)) <= set(split[role]['ids']):
            raise ValueError('A cohort contains rows outside its held-out role.')
        expected=query.loc[query.id.isin(rows.id)].drop(columns='fold')
        if content_identity(rows.drop(columns='fold')) != content_identity(expected):
            raise ValueError('A cohort changed held-out text, prompt or label.')
        if name == 'full' and set(rows.id.astype(str)) != set(split[role]['ids']):
            raise ValueError('The full cohort must contain the complete held-out role.')
        with np.load(folder/'branches.npz', allow_pickle=False) as cache:
            branches = {key:cache[key].copy() for key in cache.files}
        cohorts[name] = (rows, branches)
    if 'full' not in cohorts:
        raise ValueError('Full development cohort is required for the prespecified selection objective.')
    return cohorts, index


def fit_temperature_records(rows, branches, labels, *, fit_scope="full inner dev"):
    """Fit the declared scalar objective on supplied rows; caller owns split evidence."""
    y = rows.model.map({label:i for i,label in enumerate(labels)}).to_numpy()
    records = {}
    for name in _standalone_recipes():
        active = _branch_active(branches,name)
        target = y[active]
        values = np.asarray(branches[name+'_lp'],dtype=float)[active]
        if not len(target):
            records[name] = dict(temperature=1.,active_rows=0,represented_classes=0,
                                 status='inactive: identity temperature; standalone falls back to calibrated plain')
            continue
        counts = np.bincount(target,minlength=len(labels))
        weights = 1./counts[target]
        weights /= weights.sum()
        def objective(log_temperature):
            lp = log_softmax(values/np.exp(log_temperature),axis=1)
            return float(-np.dot(weights,lp[np.arange(len(target)),target]))
        lower,upper = np.log(.05),np.log(20.)
        result = minimize_scalar(objective,bounds=(lower,upper),method='bounded',
                                 options=dict(xatol=1e-6,maxiter=200))
        if not result.success or not np.isfinite(result.fun):
            raise ValueError('The predeclared one-dimensional calibration did not converge: '+name)
        # Include the exact identity and interval endpoints; ties prefer no change.
        points = [0.,float(result.x),lower,upper]
        best = min(points,key=objective)
        temperature = float(np.clip(np.exp(best),.05,20.))
        records[name] = dict(temperature=temperature,active_rows=int(active.sum()),
                             represented_classes=int((counts>0).sum()),
                             active_ids_sha256=_digest(sorted(rows.loc[active,'id'].astype(str))),
                             class_counts={label:int(counts[i]) for i,label in enumerate(labels)},
                             balanced_nll_before=objective(0.),balanced_nll_after=objective(np.log(temperature)),
                             boundary=bool(np.isclose(temperature,.05) or np.isclose(temperature,20.)),
                             optimizer_evaluations=int(result.nfev),status=f'fitted on {fit_scope} active rows')
    return records


def fit_branch_calibration(inner_evaluation_dir, split_manifest, training_evidence):
    """Fit four independent scalar temperatures; never inspect outer caches."""
    evidence = validate_training_evidence(training_evidence,split_manifest)
    cohorts,index = _load_cohorts(inner_evaluation_dir,split_manifest,'inner_dev')
    if index.get('provenance',{}).get('model_identity_sha256') != evidence['model_identity_sha256']:
        raise ValueError('Development calibration does not bind the declared checkpoint.')
    rows,branches = cohorts['full']
    labels = split_manifest['labels']
    records = fit_temperature_records(rows, branches, labels)
    directory = Path(inner_evaluation_dir)
    calibration = dict(policy='independent_branch_temperature_v1',specification=_calibration_spec(),
                       split_sha256=split_manifest['split_sha256'],labels=labels,
                       model_identity_sha256=evidence['model_identity_sha256'],
                       inference_options_sha256=_digest(index['options']),
                       source_evaluation_sha256=file_sha(directory/'evaluation.json'),
                       full_dev_rows_sha256=file_sha(directory/'full/rows.parquet'),
                       full_dev_branches_sha256=file_sha(directory/'full/branches.npz'),branches=records,
                       interpretation='Development labels fit temperatures and choose weights sequentially; development scores are optimistic. Outer evaluation is separate, conditional on these predeclared choices.')
    calibration['calibration_sha256'] = _digest(calibration)
    return calibration


def _bind_calibration(calibration, inner_evaluation_dir, split, index, evidence):
    calibration = _validate_calibration(calibration)
    directory = Path(inner_evaluation_dir)
    expected = dict(split_sha256=split['split_sha256'],labels=split['labels'],
                    model_identity_sha256=evidence['model_identity_sha256'],
                    inference_options_sha256=_digest(index['options']),
                    source_evaluation_sha256=file_sha(directory/'evaluation.json'),
                    full_dev_rows_sha256=file_sha(directory/'full/rows.parquet'),
                    full_dev_branches_sha256=file_sha(directory/'full/branches.npz'))
    if any(calibration.get(key)!=value for key,value in expected.items()):
        raise ValueError('Supplied calibration is not bound to these actual inner-dev scores, split and checkpoint.')
    return calibration


def select_recipe(inner_evaluation_dir, split_manifest, training_evidence, out, *, allow_exclusion=False):
    """Original v1 selection, retained unchanged for old recipe replay."""
    return _select_recipe(inner_evaluation_dir,split_manifest,training_evidence,out,
                          allow_exclusion=allow_exclusion)


def select_calibrated_recipe(inner_evaluation_dir, split_manifest, training_evidence, out, *,
                             allow_exclusion=False, calibration=None):
    """v2: fit once, then pass returned recipe['calibration'] to the second family."""
    if calibration is None:
        calibration = fit_branch_calibration(inner_evaluation_dir,split_manifest,training_evidence)
    return _select_recipe(inner_evaluation_dir,split_manifest,training_evidence,out,
                          allow_exclusion=allow_exclusion,calibration=calibration)


def _select_recipe(inner_evaluation_dir, split_manifest, training_evidence, out, *,
                   allow_exclusion=False, calibration=None, selection_branches=None,
                   crossfit_evidence=None):
    """Select by full-inner-dev BA only; stress cohorts are disclosed, not retuned."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'recipe.json').exists():
        raise FileExistsError('A chosen recipe is locked; do not overwrite it after looking at outer scores.')
    evidence = validate_training_evidence(training_evidence, split_manifest)
    cohorts, index = _load_cohorts(inner_evaluation_dir, split_manifest, 'inner_dev')
    if index.get('provenance',{}).get('model_identity_sha256') != evidence['model_identity_sha256']:
        raise ValueError('Development branches do not bind the declared checkpoint.')
    raw_cohorts = cohorts
    standalone = None
    if calibration is not None:
        calibration = _bind_calibration(calibration,inner_evaluation_dir,split_manifest,index,evidence)
        cohorts = {name:(rows,apply_branch_calibration(branches,calibration))
                   for name,(rows,branches) in raw_cohorts.items()}
        standalone = {name:_standalone_report(rows,branches,cohorts[name][1],split_manifest['labels'], index.get('source_intervention'), index)
                      for name,(rows,branches) in raw_cohorts.items()}
    recipes = candidate_recipes(allow_exclusion)
    candidates = []
    for recipe in recipes:
        score = _metrics if calibration is None else _probability_metrics
        scores = {}
        for name, (rows, branches) in cohorts.items():
            lp = mix_branches(branches, recipe)
            scores[name] = score(rows, lp, split_manifest['labels'])
            groups = _source_group_scores(rows, lp, split_manifest['labels'], index.get('source_intervention'))
            if groups:
                scores[name]['by_source_intervention_label_coverage'] = groups
        candidate = dict(recipe=recipe, cohorts=scores)
        if selection_branches is not None:
            candidate['selection_score'] = _probability_metrics(
                raw_cohorts['full'][0], mix_branches(selection_branches, recipe), split_manifest['labels'])
        candidates.append(candidate)
    # Fixed list order breaks ties. Do not optimize NLL or sparse layouts after seeing outer results.
    winner = max(range(len(candidates)), key=lambda i:candidates[i].get(
        'selection_score', candidates[i]['cohorts']['full'])['balanced_accuracy'])
    locked = dict(policy='inner_prompt_fusion_v1', split_sha256=split_manifest['split_sha256'],
                  data_governance=index.get('data_governance', dict(status='unavailable')),
                  validation_scope=index.get('validation_scope', dict(kind='unrecorded')),
                  source_intervention=index.get('source_intervention', dict(status='unrecorded')),
                  actual_source_intervention_exposure=index.get('actual_source_intervention_exposure'),
                  outer_fold=split_manifest['outer_fold'], labels=split_manifest['labels'],
                  recipe=recipes[winner], candidate_family_sha256=_digest(recipes),
                  selection=dict(objective='full inner-development balanced accuracy', tie_break='earliest preregistered candidate',
                                 development_ids_sha256=_digest(sorted(split_manifest['inner_dev']['ids'])),
                                 candidate_count=len(recipes)),
                  training_evidence=evidence,
                  model_identity_sha256=evidence['model_identity_sha256'],
                  inference_options_sha256=_digest(index['options']),
                  inference_core_options=inference_core_options(index['options']),
                  inference_core_options_sha256=_digest(inference_core_options(index['options'])),
                  source_evaluation_sha256=file_sha(Path(inner_evaluation_dir)/'evaluation.json'),
                  role='locked_inner_selection', allow_class_exclusion=bool(allow_exclusion),
                  interpretation='Noninjective family is the primary comparison. Exclusion-enabled family is sensitivity to an unguaranteed class-uniqueness assumption; neither is a leaderboard claim.')
    if calibration is not None:
        locked.update(policy='calibrated_inner_v2',calibration=calibration)
        _write(out/'calibration.json',calibration)
    if selection_branches is not None:
        if calibration is None or crossfit_evidence is None:
            raise ValueError('Cross-fitted selection requires final full-dev calibration and split evidence.')
        locked.update(policy='crossfit_calibrated_inner_v3', crossfit=crossfit_evidence)
        locked['selection']['objective'] = 'Full inner-dev BA from prompt-cross-fitted temperature predictions'
    locked['recipe_sha256'] = _digest(locked)
    diagnostic = dict(candidates=candidates, selected_index=winner,
                      data_governance=locked['data_governance'], validation_scope=locked['validation_scope'],
                      source_intervention=locked['source_intervention'],
                      actual_source_intervention_exposure=locked['actual_source_intervention_exposure'],
                      stress_cohorts='Overlapping descriptive scenarios, not independent confirmations.')
    if standalone is not None:
        diagnostic['standalone_branches'] = standalone
        diagnostic['development_interpretation'] = 'Same dev labels fit scalar temperatures and choose weights; these are optimistic fitting diagnostics, not held-out quality estimates.'
    if selection_branches is not None:
        diagnostic.update(crossfit=crossfit_evidence,
                          development_interpretation='selection_score uses temperature fits excluding each row\'s prompt. Cohort scores use the final full-dev temperature refit and are fitting diagnostics. Choosing weights still uses these 24 prompts; this is not an unbiased estimate of selected quality.')
    _write(out/'candidate_scores.json',diagnostic)
    _write(out/'recipe.json',locked)
    return locked


def apply_recipe(recipe, outer_evaluation_dir, split_manifest, out):
    """Apply the locked rule once without observing labels to choose weights."""
    if isinstance(recipe,(str,Path)):
        recipe=json.loads(Path(recipe).read_text())
    if recipe.get('recipe_sha256') != _digest({k:v for k,v in recipe.items() if k!='recipe_sha256'}):
        raise ValueError('Locked recipe changed.')
    if recipe['split_sha256'] != split_manifest['split_sha256']:
        raise ValueError('Locked recipe belongs to another split.')
    if recipe['policy'] not in ('inner_prompt_fusion_v1','calibrated_inner_v2','crossfit_calibrated_inner_v3'):
        raise ValueError('Unknown locked fusion protocol.')
    calibration = None
    if recipe['policy'] in ('calibrated_inner_v2','crossfit_calibrated_inner_v3'):
        calibration = _validate_calibration(recipe['calibration'])
        if any(calibration[key] != recipe[key] for key in ('split_sha256','labels','model_identity_sha256','inference_options_sha256')):
            raise ValueError('Locked calibration and fusion recipe have different identities.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'application.json').exists():
        raise FileExistsError('Outer application already exists; preserve the first locked evaluation.')
    cohorts,index = _load_cohorts(outer_evaluation_dir, split_manifest, 'outer_test')
    recorded_source_scope = recipe.get('source_intervention') or {}
    if recorded_source_scope.get('status') == 'available':
        from .q8_evaluation import shared_intervention_scope
        if shared_intervention_scope([recorded_source_scope, index.get('source_intervention')]) is None:
            raise ValueError('Outer evaluation lacks the source-intervention scope used for inner selection.')
        if (recorded_source_scope.get('arm_variant') is not None and
                recorded_source_scope['arm_variant'] != index['source_intervention'].get('arm_variant')):
            raise ValueError('Inner selection and outer evaluation must use the same source-intervention arm.')
    recorded_governance = recipe.get('data_governance') or {}
    outer_governance = index.get('data_governance') or {}
    if recorded_governance.get('status') == 'available' and (
            outer_governance.get('status') != 'available' or any(
                recorded_governance.get(key) != outer_governance.get(key)
                for key in ('manifest_sha256', 'data_sha256', 'rows_sha256'))):
        raise ValueError('Outer evaluation must retain the governance binding used for inner selection.')
    if index.get('provenance',{}).get('model_identity_sha256') != recipe['model_identity_sha256']:
        raise ValueError('Outer branches came from a different checkpoint; do not refit after selection.')
    if recipe.get('inference_core_options_sha256') is not None:
        expected = recipe['inference_core_options_sha256']
        if (_digest(recipe.get('inference_core_options')) != expected or
                _digest(inference_core_options(index['options'])) != expected):
            raise ValueError('Outer grouping/Gaussian settings differ from inner selection.')
    elif _digest(index['options']) != recipe['inference_options_sha256']:
        raise ValueError('Legacy recipes require identical recorded evaluation options; select a new locked recipe to add stress scenarios.')
    reports={}
    standalone={}
    for name,(rows,branches) in cohorts.items():
        if calibration is not None:
            calibrated = apply_branch_calibration(branches,calibration)
            standalone[name] = _standalone_report(rows,branches,calibrated,recipe['labels'], index.get('source_intervention'), index)
            branches = calibrated
        lp=mix_branches(branches,recipe['recipe'])
        score = _metrics if calibration is None else _probability_metrics
        reports[name]=score(rows,lp,recipe['labels'])
        groups = _source_group_scores(rows, lp, recipe['labels'], index.get('source_intervention'))
        if groups:
            reports[name]['by_source_intervention_label_coverage'] = groups
        predictions=rows[['id','prompt_id','model','fold','n_words','group_size']].copy()
        predictions['pred_model']=np.asarray(recipe['labels'])[lp.argmax(1)]
        for label,values in zip(recipe['labels'],np.exp(lp).T):
            predictions['prob__'+label]=values
        predictions.to_parquet(out/(name+'_predictions.parquet'),index=False)
    report=dict(recipe_sha256=recipe['recipe_sha256'], split_sha256=split_manifest['split_sha256'],
                data_governance=index.get('data_governance', dict(status='unavailable')),
                validation_scope=index.get('validation_scope', dict(kind='unrecorded')),
                source_intervention=index.get('source_intervention', dict(status='unrecorded')),
                actual_source_intervention_exposure=index.get('actual_source_intervention_exposure'),
                selected_recipe=recipe['recipe'], cohorts=reports,
                applied_evaluation_options=index['options'],
                stress_membership_options_changed=(_digest(index['options']) != recipe['inference_options_sha256']),
                scope='Outer prompt evaluation after locked inner selection; overlapping sparse cohorts are sensitivity diagnostics.')
    if calibration is not None:
        report.update(policy=recipe['policy'],calibration_sha256=calibration['calibration_sha256'],
                      standalone_branches=standalone)
    _write(out/'application.json',report)
    return report
