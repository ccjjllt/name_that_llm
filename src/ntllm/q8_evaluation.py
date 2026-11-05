"""Fold-local Q8 branch evaluation on fixed, text-clustered answer cohorts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULTS = dict(grouping_protocol="train_fold", seeds=[42, 43], cluster_threshold=.15, cluster_cap=18,
                cluster_min_df=2, cluster_max_df=.05, gaussian_d=128,
                gaussian_temperature=3.0, gaussian_noninjective=True)
WEIGHTS = dict(plain=2.0, centered=.70, gaussian=.44)
WORD_BINS = ("<60", "60-99", "100-159", ">=160")


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def capped_agglo(similarity, cap=18, threshold=.15):
    """Published size-capped average linkage; the inputs contain no labels."""
    n = len(similarity)
    if not n:
        return np.empty(0, dtype=int)
    affinity = similarity.astype(np.float32).copy()
    np.fill_diagonal(affinity, -np.inf)
    size, alive, groups = np.ones(n, int), np.ones(n, bool), np.arange(n)
    while True:
        valid = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        scores = np.where(valid, affinity, -np.inf)
        i, j = divmod(int(np.argmax(scores)), n)
        if not np.isfinite(scores[i, j]) or scores[i, j] < threshold:
            break
        updated = (size[i] * affinity[i] + size[j] * affinity[j]) / (size[i] + size[j])
        affinity[i], affinity[:, i] = updated, updated
        affinity[i, i] = -np.inf
        affinity[j], affinity[:, j] = -np.inf, -np.inf
        size[i] += size[j]
        alive[j] = False
        groups[groups == j] = i
    return np.unique(groups, return_inverse=True)[1]


def fit_grouping_vectorizer(texts, config=None):
    """Fit text-only TF-IDF and retain evidence of the actual fit input."""
    options = {**DEFAULTS, **(config or {})}
    texts = list(texts)
    record = dict(rows=len(texts), sklearn_version=sklearn.__version__,
                  texts_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False,
                                                       separators=(',', ':')).encode()).hexdigest(),
                  vocabulary_size=0, vocabulary_sha256=None, idf_sha256=None, fallback=None)
    try:
        vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                     min_df=options["cluster_min_df"], max_df=options["cluster_max_df"])
        vectorizer.fit(texts)
        record.update(vocabulary_size=len(vectorizer.vocabulary_),
                      vocabulary_sha256=hashlib.sha256(json.dumps(
                          sorted(vectorizer.vocabulary_.items()), separators=(',', ':')).encode()).hexdigest(),
                      idf_sha256=hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest())
    except ValueError as error:
        vectorizer, record['fallback'] = None, str(error)
    return dict(vectorizer=vectorizer, diagnostics=record)


def recover_groups(texts, config=None, *, fit_texts=None, fitted=None, return_diagnostics=False):
    """Use an explicit training-fold or unlabelled query-transductive fit.

    A failed training fit never falls back to fitting the query cohort.
    """
    options = {**DEFAULTS, **(config or {})}
    protocol, texts = options['grouping_protocol'], list(texts)
    if protocol == 'train_fold':
        if fitted is None:
            if fit_texts is None:
                raise ValueError('train_fold grouping needs training-fold fit_texts or its fitted vectorizer.')
            fitted = fit_grouping_vectorizer(fit_texts, options)
    elif protocol == 'query_transductive':
        if fit_texts is not None or fitted is not None:
            raise ValueError('query_transductive fits this query cohort; do not supply a training fit.')
        fitted = fit_grouping_vectorizer(texts, options)
    else:
        raise ValueError('grouping_protocol must be train_fold or query_transductive.')
    vectorizer, record = fitted['vectorizer'], dict(fitted['diagnostics'])
    if vectorizer is None:
        groups, zero_rows = np.arange(len(texts)), len(texts)
    else:
        features = vectorizer.transform(texts)
        zero_rows = int((features.getnnz(axis=1) == 0).sum())
        groups = capped_agglo((features @ features.T).toarray(), options['cluster_cap'],
                              options['cluster_threshold'])
    record.update(protocol=protocol, query_rows=len(texts), zero_feature_query_rows=zero_rows)
    if return_diagnostics:
        return groups, record
    return groups, record['fallback']


def make_cohorts(frame, labels, seeds=(42, 43)):
    """Truth defines stress-sample membership only, never inferred groups."""
    cohorts = [("full", np.arange(len(frame)), dict(kind="full", seed=None))]
    rates = dict(zip(labels, np.linspace(.2, 1., len(labels))))
    for seed in seeds:
        for kind, maximum in (("one_per_prompt", 1), ("sparse1_3", 3), ("sparse1_6", 6)):
            rng, selected = np.random.default_rng(seed), []
            for positions in frame.groupby("prompt_id", sort=True).indices.values():
                count = 1 if maximum == 1 else min(len(positions), int(rng.integers(1, maximum + 1)))
                selected.extend(rng.choice(positions, count, replace=False).tolist())
            cohorts.append((f"{kind}_seed{seed}", np.sort(selected),
                            dict(kind=kind, seed=int(seed), maximum_answers_per_prompt=maximum)))
        rng = np.random.default_rng(seed)
        keep = rng.random(len(frame)) < frame.model.map(rates).to_numpy()
        cohorts.append((f"class_imbalanced_seed{seed}", np.flatnonzero(keep),
                        dict(kind="class_imbalanced", seed=int(seed), class_retention_probability=rates,
                             interpretation="Prespecified label-dependent stress sampling; not an estimate of hidden-test priors.")))
    return cohorts


def grouping_diagnostics(rows):
    group_sizes = rows.groupby("inferred_group").size()
    n_prompts = rows.groupby("inferred_group").prompt_id.nunique()
    n_labels = rows.groupby("inferred_group").model.nunique()
    inferred_pairs = int(sum(n * (n - 1) // 2 for n in group_sizes))
    true_pairs = int(sum(n * (n - 1) // 2 for n in rows.prompt_id.value_counts()))
    correct_pairs = int(sum(n * (n - 1) // 2 for n in rows.groupby(["inferred_group", "prompt_id"]).size()))
    return dict(rows=len(rows), true_prompts=int(rows.prompt_id.nunique()),
                inferred_groups=len(group_sizes), active_rows=int(rows.group_size.ge(2).sum()),
                mixed_prompt_rows=int(rows.inferred_group.isin(n_prompts.index[n_prompts.gt(1)]).sum()),
                repeated_true_label_rows=int(rows.inferred_group.isin(group_sizes.index[n_labels < group_sizes]).sum()),
                inferred_pairs=inferred_pairs, true_pairs=true_pairs, correct_pairs=correct_pairs,
                pair_precision=correct_pairs / inferred_pairs if inferred_pairs else None,
                pair_recall=correct_pairs / true_pairs if true_pairs else None,
                group_size_counts={str(k): int(v) for k, v in group_sizes.value_counts().sort_index().items()})


def _metrics(rows, labels, probability, baseline=None):
    if not len(rows):
        return dict(rows=0, balanced_accuracy=None, nll=None, balanced_nll=None, brier=None,
                    classes_present=0, all_18_classes_present=False, per_class=[])
    truth = rows.model.to_numpy()
    target = pd.Series(truth).map({label: i for i, label in enumerate(labels)}).to_numpy(int)
    prediction = probability.argmax(1)
    correct = prediction == target
    nll = -np.log(np.clip(probability[np.arange(len(rows)), target], 1e-30, 1))
    per_class = []
    for i, label in enumerate(labels):
        mask = target == i
        per_class.append(dict(model=label, rows=int(mask.sum()),
                              recall=float(correct[mask].mean()) if mask.any() else None))
    result = dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()),
                  balanced_accuracy=float(pd.Series(correct).groupby(truth).mean().mean()),
                  balanced_accuracy_scope="Mean recall over labels present in this slice; pool fold predictions before final scenario scoring.",
                  classes_present=int(np.unique(target).size), all_18_classes_present=np.unique(target).size == 18,
                  nll=float(nll.mean()), balanced_nll=float(pd.Series(nll).groupby(truth).mean().mean()),
                  brier=float((np.square(probability).sum(1) - 2 * probability[np.arange(len(rows)), target] + 1).mean()),
                  per_class=per_class)
    if baseline is not None:
        first = baseline.argmax(1)
        before = first == target
        baseline_ba = float(pd.Series(before).groupby(truth).mean().mean())
        result["paired_vs_plain"] = dict(changed=int((first != prediction).sum()),
                                         rescued=int((~before & correct).sum()), lost=int((before & ~correct).sum()),
                                         baseline_ba=baseline_ba, delta_ba=result["balanced_accuracy"] - baseline_ba)
    return result


def score_view(rows, labels, log_probability, plain_probability):
    probability = np.exp(log_probability)
    report = _metrics(rows, labels, probability, plain_probability)
    lengths = pd.cut(rows.n_words, [-1, 59, 99, 159, np.inf], labels=WORD_BINS)
    report["by_word_count"] = {}
    for name in WORD_BINS:
        mask = lengths.eq(name).to_numpy()
        report["by_word_count"][name] = _metrics(rows.loc[mask], labels, probability[mask], plain_probability[mask])
    report["by_inferred_group_size"] = {}
    for size in sorted(rows.group_size.unique()):
        mask = rows.group_size.eq(size).to_numpy()
        report["by_inferred_group_size"][str(size)] = _metrics(rows.loc[mask], labels, probability[mask], plain_probability[mask])
    return report


def _prediction_frame(rows, labels, log_probability):
    result = rows[["id", "prompt_id", "n_words", "fold", "inferred_group", "group_size"]].copy()
    probability = np.exp(log_probability)
    result["true_model"] = rows.model.to_numpy()
    result["pred_model"] = np.asarray(labels)[probability.argmax(1)]
    for i, label in enumerate(labels):
        result[f"prob__{label}"] = probability[:, i]
    return result


def evaluate_cohorts(fit_frame, query_frame, labels, plain_logits, raw_centered_logits,
                     centered_bias, output_dir, *, query_features=None, fit_features=None,
                     centered_trained=True, provenance=None, config=None):
    """Evaluate one held-out fold; source model provenance remains caller supplied.

    For a linear head u=Wz+b, W(z-mean(z))+b = u-mean(u)+b. This
    computes centered scores without storing every high-dimensional feature.
    Gaussian alone requires same-checkpoint fit/query features, fit on train only.
    """
    options = {**DEFAULTS, **(config or {})}
    if options['grouping_protocol'] not in ('train_fold', 'query_transductive'):
        raise ValueError('grouping_protocol must be train_fold or query_transductive.')
    if not 2 <= options["cluster_cap"] <= 18 or not 0 < options["cluster_threshold"] <= 1:
        raise ValueError("Clustering requires cap 2..18 and a positive cosine threshold <=1.")
    if options["gaussian_temperature"] <= 0 or not np.isfinite(options["gaussian_temperature"]):
        raise ValueError("Gaussian temperature must be finite and positive.")
    labels = list(labels)
    fit, query = fit_frame.reset_index(drop=True).copy(), query_frame.reset_index(drop=True).copy()
    if len(labels) != 18 or len(set(labels)) != 18 or not len(fit) or not len(query):
        raise ValueError("Evaluation needs nonempty fit/query rows and the 18-label order.")
    for frame in (fit, query):
        if not frame.id.is_unique or not frame.model.isin(labels).all():
            raise ValueError("Evaluation rows need unique IDs and known model labels.")
        if "n_words" not in frame:
            frame["n_words"] = frame.text.str.split().str.len()
    if set(fit.model) != set(labels) or query.fold.nunique() != 1:
        raise ValueError("Fit data must contain all classes and query rows exactly one checkpoint fold.")
    if set(fit.id) & set(query.id) or set(fit.prompt_id) & set(query.prompt_id):
        raise ValueError("Fit/query IDs and true prompts must be disjoint.")
    plain_logits = np.asarray(plain_logits, dtype=np.float64)
    if plain_logits.shape != (len(query), len(labels)) or not np.isfinite(plain_logits).all():
        raise ValueError("Plain logits must be finite and aligned with query rows and labels.")
    if centered_trained:
        raw_centered_logits = np.asarray(raw_centered_logits, dtype=np.float64)
        centered_bias = np.asarray(centered_bias, dtype=np.float64)
        if (raw_centered_logits.shape != plain_logits.shape or centered_bias.shape != (len(labels),)
                or not np.isfinite(raw_centered_logits).all() or not np.isfinite(centered_bias).all()):
            raise ValueError("Trained centered logits and head bias are required and must be finite.")
    if (fit_features is None) != (query_features is None):
        raise ValueError("Gaussian evaluation requires both same-checkpoint fit and query features.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fit[["id", "prompt_id", "text", "model", "n_words", "fold"]].to_parquet(output / "fit_rows.parquet", index=False)
    query[["id", "prompt_id", "text", "model", "n_words", "fold"]].to_parquet(output / "query_rows.parquet", index=False)
    np.savez_compressed(
        output / "head_logits.npz", plain_logits=plain_logits,
        raw_centered_logits=(np.asarray(raw_centered_logits, dtype=np.float64)
                             if raw_centered_logits is not None else np.empty((0, len(labels)))),
        centered_bias=(np.asarray(centered_bias, dtype=np.float64)
                       if centered_bias is not None else np.empty(0)),
        centered_trained=np.asarray(bool(centered_trained)),
        ids=query.id.astype(str).to_numpy(dtype=str), labels=np.asarray(labels, dtype=str))
    head_cache = dict(file="head_logits.npz", query_rows_file="query_rows.parquet",
                      centered_trained=bool(centered_trained),
                      source="Caller-supplied affine logits and head bias from the checkpoint declared in provenance.")
    fit_scope = dict(rows=len(fit), prompts=int(fit.prompt_id.nunique()), query_fold=int(query.fold.iloc[0]),
                     train_query_prompt_overlap=0, fit_rows_file="fit_rows.parquet",
                     fit_ids_sha256=hashlib.sha256("\n".join(fit.id.astype(str)).encode()).hexdigest(),
                     interpretation="All Gaussian statistics use these fit rows only; feature checkpoint training history is supplied in provenance, not inferred from disjoint IDs.")
    gaussian_available = fit_features is not None
    prm, projected = None, None
    if gaussian_available:
        from . import gaussian_head as gaussian
        fit_features, query_features = np.asarray(fit_features), np.asarray(query_features)
        if (fit_features.ndim != 2 or query_features.ndim != 2 or len(fit_features) != len(fit)
                or len(query_features) != len(query) or fit_features.shape[1] != query_features.shape[1]
                or not np.isfinite(fit_features).all() or not np.isfinite(query_features).all()):
            raise ValueError("Gaussian fit/query features must be finite, aligned and share feature coordinates.")
        targets = fit.model.map({label: i for i, label in enumerate(labels)}).to_numpy(int)
        prm = gaussian.fit(fit_features, targets, fit.prompt_id.to_numpy(),
                           dict(red="wpca", d=int(options["gaussian_d"]), se_shrink="lw", sp_shrink="lw",
                                lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
        prm["T1"] = prm["Tg"] = float(options["gaussian_temperature"])
        projected = gaussian.transform(query_features, prm)
        np.savez_compressed(output / "gaussian_fit.npz", **{k: v for k, v in prm.items() if isinstance(v, np.ndarray)})
        _json(output / "gaussian_fit.json", {k: v for k, v in prm.items() if not isinstance(v, np.ndarray)})
        np.save(output / "gaussian_query_projection.npy", projected)
    else:
        for filename in ("gaussian_fit.npz", "gaussian_fit.json", "gaussian_query_projection.npy"):
            (output / filename).unlink(missing_ok=True)
    availability = dict(plain=True, centered=bool(centered_trained), gaussian=gaussian_available)
    unavailable = {}
    if not centered_trained:
        unavailable["centered"] = "No effective centered training; the head is excluded from inference views."
    if not gaussian_available:
        unavailable["gaussian"] = "Same-checkpoint fit/query feature arrays were not supplied."
    fitted_grouping = (fit_grouping_vectorizer(fit.text.tolist(), options)
                       if options['grouping_protocol'] == 'train_fold' else None)
    if fitted_grouping is not None:
        fitted_grouping['diagnostics']['fit_ids_sha256'] = fit_scope['fit_ids_sha256']
    report = dict(scope="One held-out checkpoint fold; grouping uses unlabelled text under the declared vocabulary-fit protocol.",
                  labels=labels, provenance=provenance or {}, fit_scope=fit_scope,
                  branch_availability=availability, unavailable_reasons=unavailable, head_cache=head_cache,
                  standalone_scale="Fixed divisor 2 for q8 weighted block: softmax(P + .35*C + .22*G). Inactive relation contributions are zero, so singletons equal plain. This is not the complete-system member divisor M.",
                  full_system_q8_weights=WEIGHTS, options=options,
                  grouping_fit=(fitted_grouping['diagnostics'] if fitted_grouping else None), cohorts={})
    plain_all = log_softmax(plain_logits, axis=1)
    cohorts = make_cohorts(query, labels, options["seeds"])
    requested = options.get("eval_cohorts")
    if requested is not None:
        available = {name for name, _, _ in cohorts} | {spec["kind"] for _, _, spec in cohorts}
        if not isinstance(requested, (list, tuple)) or not requested or not set(requested) <= available:
            raise ValueError(f"eval_cohorts must select cohort names or kinds from {sorted(available)}.")
        cohorts = [(name, positions, spec) for name, positions, spec in cohorts
                   if name in requested or spec["kind"] in requested]
    for name, positions, definition in cohorts:
        if not len(positions):
            report["cohorts"][name] = dict(available=False, reason="The prespecified cohort contains no rows.")
            continue
        rows = query.iloc[positions].reset_index(drop=True).copy()
        groups, group_fit = recover_groups(rows.text.tolist(), options, fitted=fitted_grouping,
                                           return_diagnostics=True)
        if options['grouping_protocol'] == 'query_transductive':
            group_fit['fit_ids_sha256'] = hashlib.sha256('\n'.join(rows.id.astype(str)).encode()).hexdigest()
        fallback = group_fit['fallback']
        sizes = np.bincount(groups)[groups]
        rows["inferred_group"], rows["group_size"] = groups, sizes
        grouped = sizes >= 2
        p = plain_all[positions]
        c, g, g_noninjective = np.zeros_like(p), np.zeros_like(p), np.zeros_like(p)
        centered_active = grouped & bool(centered_trained)
        gaussian_active = grouped & gaussian_available
        if centered_trained:
            u = raw_centered_logits[positions]
            for group in np.unique(groups[centered_active]):
                part = groups == group
                c[part] = log_softmax(u[part] - u[part].mean(0, keepdims=True) + centered_bias, axis=1)
        if gaussian_active.any():
            dense = np.unique(groups[gaussian_active], return_inverse=True)[1]
            g[gaussian_active] = gaussian.scores(None, dense, prm, U=projected[positions[gaussian_active]])
            if options["gaussian_noninjective"]:
                g_noninjective[gaussian_active] = gaussian.scores(
                    None, dense, prm, cfg=dict(inj=False, repel=False), U=projected[positions[gaussian_active]])
        checked_branches = [(c, centered_active), (g, gaussian_active)]
        if gaussian_available and options["gaussian_noninjective"]:
            checked_branches.append((g_noninjective, gaussian_active))
        for values, active in checked_branches:
            if not np.isfinite(values).all() or not np.allclose(np.exp(values[active]).sum(1), 1, atol=1e-6):
                raise FloatingPointError("Nonfinite or unnormalized active branch log-probabilities.")
        membership_signature = hashlib.sha256(json.dumps(dict(scenario=definition, rows=[
            [str(row.id), row.text] for row in rows.itertuples()
        ]), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        signature = hashlib.sha256(json.dumps(dict(scenario=definition, protocol=options['grouping_protocol'],
            vectorizer_fit=group_fit, cluster_threshold=options['cluster_threshold'],
            cluster_cap=options['cluster_cap'], rows=[
            [str(row.id), row.text, int(row.inferred_group)] for row in rows.itertuples()
        ]), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        cohort_dir = output / name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(cohort_dir / "rows.parquet", index=False)
        branches = dict(plain_lp=p, centered_lp=c, gaussian_lp=g,
                        centered_active=centered_active, gaussian_active=gaussian_active)
        if gaussian_available and options["gaussian_noninjective"]:
            branches["gaussian_noninjective_lp"] = g_noninjective
            np.save(cohort_dir / "gaussian_query_projection.npy", projected[positions])
        elif gaussian_available:
            np.save(cohort_dir / "gaussian_query_projection.npy", projected[positions])
        else:
            (cohort_dir / "gaussian_query_projection.npy").unlink(missing_ok=True)
        np.savez_compressed(cohort_dir / "branches.npz", **branches)
        views = dict(plain=p)
        if centered_trained:
            views["centered_with_plain_fallback"] = np.where(centered_active[:, None], c, p)
            views["plain_centered"] = log_softmax(p + .35 * c, axis=1)
        if gaussian_available:
            views["plain_gaussian"] = log_softmax(p + .22 * g, axis=1)
        if centered_trained and gaussian_available:
            views["plain_centered_gaussian"] = log_softmax(p + .35 * c + .22 * g, axis=1)
            if options["gaussian_noninjective"]:
                views["plain_centered_gaussian_noninjective"] = log_softmax(p + .35 * c + .22 * g_noninjective, axis=1)
        for previous in cohort_dir.glob("*_predictions.parquet"):
            if previous.name.removesuffix("_predictions.parquet") not in views:
                previous.unlink()
        cohort_report = dict(available=True, scenario=definition, cohort_signature=signature,
                             cohort_membership_signature=membership_signature, grouping_fit=group_fit,
                             cluster_fallback=fallback, grouping=grouping_diagnostics(rows), views={})
        for view, lp in views.items():
            if not np.isfinite(lp).all():
                raise FloatingPointError(f"Nonfinite {view} prediction.")
            cohort_report["views"][view] = score_view(rows, labels, lp, np.exp(p))
            _prediction_frame(rows, labels, lp).to_parquet(cohort_dir / f"{view}_predictions.parquet", index=False)
        cohort_report["centered_active_only"] = (
            score_view(rows.loc[centered_active], labels, c[centered_active], np.exp(p[centered_active]))
            if centered_trained else dict(available=False, reason=unavailable["centered"]))
        manifest = dict(labels=labels, ids=rows.id.astype(str).tolist(), cohort_signature=signature,
                        cohort_membership_signature=membership_signature,
                        scenario=definition, branch_availability=availability, unavailable_reasons=unavailable,
                        available_views=list(views), provenance=provenance or {},
                        fit_scope={**fit_scope, "fit_rows_file": "../fit_rows.parquet"},
                        head_cache={**head_cache, "file": "../head_logits.npz", "query_rows_file": "../query_rows.parquet"},
                        cluster=dict(protocol=options['grouping_protocol'],
                                     threshold=options["cluster_threshold"], cap=options["cluster_cap"],
                                     min_df=options["cluster_min_df"], max_df=options["cluster_max_df"],
                                     vocabulary_scope=("training-fold text only" if fitted_grouping is not None
                                                       else "this cohort's unlabelled query text"),
                                     vectorizer_fit=group_fit, fallback=fallback),
                        gaussian=dict(temperature=options["gaussian_temperature"], requested_dimension=options["gaussian_d"],
                                      fitted_dimension=prm["info"]["d"] if gaussian_available else None,
                                      inj=True, repel=True, noninjective_sensitivity=bool(options["gaussian_noninjective"]),
                                      fit_parameters_file="../gaussian_fit.npz" if gaussian_available else None,
                                      assumption="At-most-once class evidence is a published modeling assumption, not a guarantee about inferred clusters."),
                        standalone_scale=report["standalone_scale"], full_system_q8_weights=WEIGHTS)
        _json(cohort_dir / "manifest.json", manifest)
        _json(cohort_dir / "metrics.json", cohort_report)
        report["cohorts"][name] = cohort_report
    _json(output / "evaluation.json", report)
    return report


def summarize_runs(run_dirs):
    """Pool current-index views across distinct folds; never average fold BA."""
    from .compare_lora import paired_interval, infer_grouping_protocol
    directories = [Path(path) / "evaluation" for path in run_dirs]
    indexes = [json.loads((path / "evaluation.json").read_text()) for path in directories]
    if not indexes:
        raise ValueError("At least one evaluated run is required.")
    protocol_evidence = []
    for directory, item in zip(directories, indexes):
        evidence = []
        for name, cohort in item['cohorts'].items():
            if cohort.get('available'):
                manifest = json.loads((directory / name / 'manifest.json').read_text())
                evidence.append(infer_grouping_protocol(item, manifest))
        if not evidence:
            evidence = [infer_grouping_protocol(item, {})]
        if len({value['protocol'] for value in evidence}) != 1:
            raise ValueError('Grouping protocols within one evaluation index are inconsistent or unrecorded.')
        protocol_evidence.append(dict(run=str(directory.parent.resolve()),
                                      protocol=evidence[0]['protocol'],
                                      inferred=any(value['inferred'] for value in evidence),
                                      sources=sorted({value['source'] for value in evidence})))
    labels, first = indexes[0]["labels"], indexes[0]
    conditions = lambda item, protocol: (item["labels"], {
                               **{key: value for key, value in item["options"].items()
                                  if key not in ("eval_cohorts", "seeds")}, 'grouping_protocol': protocol}, item["standalone_scale"],
                               item.get("provenance", {}).get("model_kind"))
    if any(conditions(item, evidence['protocol']) != conditions(first, protocol_evidence[0]['protocol'])
           for item, evidence in zip(indexes, protocol_evidence)):
        raise ValueError("Run labels or declared evaluation conditions differ.")
    signatures = [item.get("provenance", {}).get("condition_signature") for item in indexes]
    if len({value for value in signatures if value}) > 1:
        raise ValueError("Training-condition signatures differ; summarize each condition separately.")
    folds = [item["fit_scope"]["query_fold"] for item in indexes]
    if len(set(folds)) != len(folds):
        raise ValueError("More than one run was supplied for a checkpoint fold.")
    report = dict(scope="Pooled named-view OOF on each fixed scenario; seeds remain separate sensitivity analyses.",
                  labels=labels, folds=sorted(folds), complete_five_folds=set(folds) == set(range(5)),
                  training_condition_verified=bool(signatures) and all(signatures),
                  condition_signature=next((value for value in signatures if value), None),
                  grouping_protocol=protocol_evidence[0]['protocol'],
                  grouping_protocol_evidence=protocol_evidence,
                  evaluation_options=conditions(first, protocol_evidence[0]['protocol'])[1],
                  run_dirs=[str(Path(path).resolve()) for path in run_dirs], cohorts={})
    for name in sorted(set().union(*(item["cohorts"] for item in indexes))):
        entries = [item["cohorts"].get(name, {}) for item in indexes]
        missing = [fold for fold, entry in zip(folds, entries) if not entry.get("available")]
        if missing:
            report["cohorts"][name] = dict(complete=False, missing_cohort_folds=missing, views={})
            continue
        if any(entry["scenario"] != entries[0]["scenario"] for entry in entries):
            raise ValueError(f"{name}: scenario definitions differ across folds.")
        frames = [pd.read_parquet(path / name / "rows.parquet") for path in directories]
        for fold, frame in zip(folds, frames):
            if not frame.fold.eq(fold).all():
                raise ValueError(f"{name}: cohort rows do not belong to the declared checkpoint fold.")
        rows = pd.concat(frames, ignore_index=True)
        if not rows.id.is_unique or rows.groupby("prompt_id").fold.nunique().max() != 1:
            raise ValueError(f"{name}: duplicate query IDs or prompts spanning checkpoint folds.")

        def probabilities(view):
            parts = []
            for path, frame in zip(directories, frames):
                pred = pd.read_parquet(path / name / f"{view}_predictions.parquet").set_index("id")
                if not pred.index.is_unique or set(pred.index) != set(frame.id):
                    raise ValueError(f"{name}/{view}: prediction IDs do not match cohort rows.")
                pred = pred.loc[frame.id]
                if not np.array_equal(pred[["prompt_id", "fold", "true_model", "n_words"]].to_numpy(),
                                      frame[["prompt_id", "fold", "model", "n_words"]].to_numpy()):
                    raise ValueError(f"{name}/{view}: prediction labels or fold metadata differ.")
                parts.append(pred[[f"prob__{label}" for label in labels]].to_numpy(float))
            values = np.concatenate(parts)
            if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-7):
                raise ValueError(f"{name}/{view}: invalid probability rows.")
            return values / values.sum(1, keepdims=True)

        if any("plain" not in entry.get("views", {}) for entry in entries):
            raise ValueError(f"{name}: every evaluated fold needs the plain reference view.")
        plain = probabilities("plain")
        combined = dict(complete=report["complete_five_folds"], scenario=entries[0]["scenario"], rows=len(rows), views={})
        missing_centered = [fold for fold, item in zip(folds, indexes)
                            if not item["branch_availability"].get("centered", False)]
        if missing_centered:
            combined["centered_active_only"] = dict(complete=False, missing_trained_head_folds=missing_centered)
        else:
            masks, centered_parts = [], []
            for path, frame in zip(directories, frames):
                with np.load(path / name / "branches.npz", allow_pickle=False) as branches:
                    mask, values = branches["centered_active"], branches["centered_lp"]
                    if (mask.dtype.kind != "b" or mask.shape != (len(frame),)
                            or values.shape != (len(frame), len(labels)) or not np.isfinite(values).all()
                            or not np.allclose(np.exp(values[mask]).sum(1), 1, atol=1e-7)):
                        raise ValueError(f"{name}: invalid saved centered branch or active mask.")
                    masks.append(mask)
                    centered_parts.append(values[mask])
            active = np.concatenate(masks)
            lp = np.concatenate(centered_parts)
            result = score_view(rows.loc[active], labels, lp, plain[active])
            result["complete_five_folds"] = report["complete_five_folds"]
            result["descriptive_prompt_bootstrap_95pct_vs_plain"] = (paired_interval(
                rows.loc[active].assign(true_model=rows.loc[active, "model"]),
                np.asarray(labels)[plain[active].argmax(1)], np.asarray(labels)[lp.argmax(1)]) if active.any() else None)
            combined["centered_active_only"] = result
        for view in sorted(set().union(*(entry["views"] for entry in entries))):
            absent = [fold for fold, entry in zip(folds, entries) if view not in entry["views"]]
            if absent:
                combined["views"][view] = dict(complete=False, missing_view_folds=absent)
                continue
            p = probabilities(view)
            result = score_view(rows, labels, np.log(np.clip(p, 1e-300, 1)), plain)
            result["complete_five_folds"] = report["complete_five_folds"]
            result["descriptive_prompt_bootstrap_95pct_vs_plain"] = paired_interval(
                rows.assign(true_model=rows.model), np.asarray(labels)[plain.argmax(1)], np.asarray(labels)[p.argmax(1)])
            combined["views"][view] = result
        report["cohorts"][name] = combined
    return report


def main():
    import argparse
    from datetime import datetime, timezone
    from uuid import uuid4
    parser = argparse.ArgumentParser(description="Pool saved multihead evaluation views across checkpoint folds.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    report = summarize_runs(args.runs)
    output = args.output_dir / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_evaluation_summary_{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "summary.json", report)
    print(json.dumps(dict(complete_five_folds=report["complete_five_folds"], cohorts=len(report["cohorts"]))))
    print(output.resolve())


if __name__ == "__main__":
    main()
