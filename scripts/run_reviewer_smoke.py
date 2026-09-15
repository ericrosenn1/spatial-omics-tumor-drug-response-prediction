"""Execute a deterministic public fixture through maintained numerical interfaces.

No private data, pretrained downloads, manuscript code or prewritten PASS tables
are used. The small settings are a software smoke test, never evidence of model
validity. See smoke_report.json for executed branches, exclusions and real checks.
Run: python scripts/run_reviewer_smoke.py --output local/reviewer_smoke
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
for relative in ["scripts", "spatial_feature_identification_pipeline/code",
    "prediction_modeling_pipeline/teacher_builder/scripts",
    "prediction_modeling_pipeline/spatial_prediction_model_V2/src",
    "prediction_modeling_pipeline/prediction_interpretation_model/scripts",
    "prediction_modeling_pipeline/spatial_transfer_inference_model/scripts"]:
    sys.path.insert(0, str(ROOT / relative))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from reviewer_smoke_fixture import write_visium_fixture
from teacher_expression_contract import raw_counts_to_pseudobulk, select_gene_ids, score_expression_artifact
from spm_v2 import conditional_validation as cv
from spm_v2.feature_reference import FeatureReference
from spm_v2.model_training import make_xgb_pipeline, select_features_training_only
from spm_v2.predictor_bundle import SpatialPredictorBundle, save_bundle, load_bundle
from _alignment_contract import fit_reference, save_reference
from alignment_bundle import apply_atlas


def module(name, relative):
    """Import the candidate's maintained file, retaining an auditable source path."""
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t" if path.suffix == ".tsv" else ",", index=False)


def run(output, config):
    """Build actual fixture outputs and fail on numerical, identity or reload defects."""
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Smoke output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("scope") != "SMOKE_ONLY_NOT_SCIENTIFIC_VALIDATION":
        raise ValueError("Explicit smoke-only configuration required")
    cv.atomic_json(output / "smoke_config.json", config)
    checks, commands, sources = [], [], {}

    def check(name, condition, detail=""):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    def rejects(name, function):
        try:
            function()
        except (ValueError, FileExistsError):
            check(name, True)
        else:
            check(name, False, "Invalid input was accepted")

    def command(relative, arguments):
        args = [sys.executable, str(ROOT / relative)] + [str(x) for x in arguments]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180)
        log = output / "logs" / (Path(relative).stem + ".log")
        log.parent.mkdir(exist_ok=True)
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        commands.append({"script": relative, "arguments": [str(x) for x in arguments], "exit_code": result.returncode, "log": str(log.relative_to(output))})
        if result.returncode:
            raise RuntimeError(f"{relative} failed; inspect {log}")

    spatial = module("smoke_spatial", "spatial_feature_identification_pipeline/code/02_process_samples.py")
    axes = module("smoke_axes", "spatial_feature_identification_pipeline/code/05_build_multi_axis_transcriptome_labels.py")
    fusion = module("smoke_fusion", "prediction_modeling_pipeline/teacher_builder/scripts/04_fuse_teacher_tables.py")
    pim = module("smoke_pim", "prediction_modeling_pipeline/prediction_interpretation_model/scripts/03_compute_signed_spatial_effects.py")
    dictionary_module = module("smoke_dictionary", "prediction_modeling_pipeline/prediction_interpretation_model/scripts/02_build_feature_and_treatment_dictionary.py")
    precomputed = module("smoke_precomputed", "prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py")
    for value in [spatial, axes, fusion, pim, dictionary_module, precomputed, cv]:
        path = Path(value.__file__).resolve()
        check("candidate_source_import_" + path.stem, path.is_relative_to(ROOT))
        sources[path.relative_to(ROOT).as_posix()] = digest(path)

    print("Smoke: generate raw fixtures and process retained spots", flush=True)
    sections = write_visium_fixture(output / "raw_visium_fixtures", config, spatial.MARKER_PROGRAMS)
    raw_features, pseudobulks, sample_qc = [], [], []
    ordered_genes = ["ENSG00000000002", "ENSG00000000001", "ENSG00000000003", "ENSG99999999999"]
    for section in sections:
        adata, info = spatial.load_sample(section["path"])
        original_barcodes = adata.obs_names.tolist()
        spatial.add_qc_metrics(adata)
        retained = spatial.filter_adata(adata)
        check("low_count_spot_filtered_" + section["sample_id"], retained.n_obs == config["spots_per_section"] - 1)
        check("retained_barcode_subset_" + section["sample_id"], set(retained.obs_names) < set(original_barcodes))
        gene_ids = select_gene_ids(retained.var, retained.var_names)
        bulk = raw_counts_to_pseudobulk(retained.X, gene_ids, ordered_genes)
        reversed_bulk = raw_counts_to_pseudobulk(retained.X[:, ::-1], gene_ids[::-1], ordered_genes)
        check("gene_order_invariant_" + section["sample_id"], bulk == reversed_bulk)
        pseudobulks.append({"sample_id": section["sample_id"], **bulk})
        with warnings.catch_warnings(record=True) as preprocessing_warnings:
            processed = spatial.preprocess_adata(retained, n_top_genes=config["genes"], n_pcs=config["spatial_pcs"])
        check("required_clustering_completed_" + section["sample_id"], "leiden" in processed.obs and "X_pca" in processed.obsm and "X_umap" in processed.obsm)
        processed, score_columns = spatial.score_marker_programs(processed)
        check("marker_programs_computed_" + section["sample_id"], len(score_columns) == 7)
        spatial.assign_spot_labels(processed, score_columns)
        spatial.add_basic_spatial_features(processed)
        row = spatial.build_slide_feature_row(processed, info).iloc[0].to_dict()
        refined = axes.refine_axis_scores(processed.obs[score_columns], processed.obsm["spatial"], processed.obs)
        for score_column in score_columns:
            prefix = "structure_score__" + score_column.removesuffix("_score")
            row.update(axes.summarize_vector(refined[score_column], prefix))
        raw_features.append(row)
        sample_qc.append({"sample_id": section["sample_id"], "loaded_spots": adata.n_obs, "retained_spots": processed.n_obs,
                          "retained_barcode_sha256": hashlib.sha256("\n".join(processed.obs_names).encode()).hexdigest(),
                          "coordinate_units": "synthetic full-resolution image pixels", "status": "PASS",
                          "preprocessing_warnings": json.dumps([str(w.message) for w in preprocessing_warnings])})
        processed.write_h5ad(output / "raw_visium_fixtures" / section["sample_id"] / "processed.h5ad")
    features = pd.DataFrame(raw_features)
    # Preserve absent simple-label fractions as NA, as the maintained extractor does.
    bulk = pd.DataFrame(pseudobulks).set_index("sample_id")
    training_ids = [x["sample_id"] for x in sections if x["is_training"]]
    train_raw = features[features.sample_id.isin(training_ids)].copy().reset_index(drop=True)
    external_raw = features[~features.sample_id.isin(training_ids)].copy().reset_index(drop=True)
    reference = FeatureReference().fit(train_raw)
    transformed = reference.transform(features)
    numeric_columns = [c for c in transformed if c != "sample_id" and pd.api.types.is_numeric_dtype(transformed[c])
                       and transformed.loc[transformed.sample_id.isin(training_ids), c].nunique() > 1]
    training = transformed[transformed.sample_id.isin(training_ids)][["sample_id"]+numeric_columns].reset_index(drop=True)
    external = transformed[~transformed.sample_id.isin(training_ids)][["sample_id"]+numeric_columns].reset_index(drop=True)
    write_table(features, output / "features/raw_features.csv")
    write_table(training, output / "features/model_input_numeric.csv")
    write_table(external, output / "features/external_numeric.csv")
    write_table(pd.DataFrame({"feature": numeric_columns}), output / "features/feature_manifest.csv")
    write_table(pd.DataFrame(sample_qc), output / "features/spot_qc.tsv")
    write_table(bulk.reset_index(), output / "teacher/pseudobulk.tsv")

    print("Smoke: trained-object fixture scoring, fusion and precomputed handoff", flush=True)
    teacher_root = output / "teacher"
    x = bulk.loc[training_ids, ordered_genes]
    labels = (np.arange(len(x)) >= len(x)//2).astype(int)
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(random_state=config["seed"]))
    base.fit(x, labels)
    p = np.clip(base.predict_proba(x)[:, 1], 1e-8, 1-1e-8)
    calibration = LogisticRegression(random_state=config["seed"]).fit(np.log(p/(1-p)).reshape(-1, 1), labels)
    artifact = {"base_model": base, "gene_columns": ordered_genes, "calibration_method": "sigmoid", "calibrator": calibration}
    artifact_path = teacher_root / "expression_trained_fixture.joblib"
    joblib.dump(artifact, artifact_path)
    raw_probability, probability, coverage = score_expression_artifact(joblib.load(artifact_path), x.drop(columns=ordered_genes[-1]))
    check("legitimate_missing_expression_gene", coverage["missing_gene_count"] == 1)
    _, reordered, _ = score_expression_artifact(joblib.load(artifact_path), x.iloc[::-1, ::-1])
    np.testing.assert_array_equal(probability, reordered[::-1])
    check("saved_expression_calibration_order_and_reload", True)
    histology_fixture = make_pipeline(StandardScaler(), LogisticRegression(random_state=config["seed"]+1)).fit(x, 1-labels)
    joblib.dump(histology_fixture, teacher_root / "histology_trained_probability_fixture.joblib")
    hist_probability = joblib.load(teacher_root / "histology_trained_probability_fixture.joblib").predict_proba(x)[:, 1]
    keys = ["fixture agent alpha", "fixture agent beta | fixture agent gamma"]
    expression_rows, histology_rows = [], []
    for j, key in enumerate(keys):
        for i, sample in enumerate(training_ids):
            if i % 3 != 0:
                expression_rows.append({"sample_id": sample, "drug_key": key.upper(), "drug": key,
                    "expression_prob_raw": raw_probability[i], "expression_prob_calibrated": probability[i] if j == 0 else 1-probability[i],
                    "expression_reliability_weight": .7, "expression_teacher_mode": "trained_smoke_fixture"})
            if i % 3 != 1:
                histology_rows.append({"sample_id": sample, "drug_key": key, "drug": key,
                    "histology_prob_calibrated": hist_probability[i] if j == 0 else 1-hist_probability[i],
                    "histology_reliability_weight": .6, "histology_control_factor": 1.})
    expression_rows = pd.DataFrame(expression_rows)
    histology_rows = pd.DataFrame(histology_rows)
    rejects("duplicate_normalized_teacher_key_rejected", lambda: fusion.standardize_expression(pd.concat([expression_rows, expression_rows.iloc[[0]]])))
    fallback = expression_rows.iloc[[0]].copy()
    fallback["expression_teacher_mode"] = "model_artifact_missing_prior_only"
    check("prior_only_not_expression_success", not fusion.standardize_expression(fallback).expression_available.any())
    write_table(expression_rows, teacher_root / "02_expression_teacher/expression_teacher_scores.tsv")
    write_table(histology_rows, teacher_root / "03_histology_teacher/histology_teacher_scores.tsv")
    write_table(pd.DataFrame({"canonical_treatment_key": keys, "prior_prob_responder": [.45, .55], "prior_n": 40,
        "prior_responders": [18, 22], "prior_nonresponders": [22, 18], "prior_source": "seeded_fixture"}), teacher_root / "01_input_validation/treatment_priors.tsv")
    teacher_config = {"output_root": str(teacher_root), "sample_col": "sample_id",
        "spatial_feature_table": str(output / "features/model_input_numeric.csv"), "spatial_feature_manifest": str(output / "features/feature_manifest.csv")}
    # JSON is a strict subset of YAML and uses the same maintained config reader.
    cv.atomic_json(output / "teacher_config.json", teacher_config)
    command("prediction_modeling_pipeline/teacher_builder/scripts/04_fuse_teacher_tables.py", ["--config", output / "teacher_config.json"])
    command("prediction_modeling_pipeline/teacher_builder/scripts/05_build_prediction_ready_teacher.py", ["--config", output / "teacher_config.json"])
    handoff = teacher_root / "05_prediction_ready_teacher"
    fused = pd.read_csv(handoff / "visium_fused_teacher_table.tsv", sep="\t")
    check("all_three_modality_branches", set(fused.modality_used) == {"both", "expression_only", "histology_only"})
    check("teacher_full_keys_and_rows", set(fused.drug_key) == set(keys) and len(fused) == len(training_ids)*len(keys))
    check("teacher_residual_arithmetic", np.allclose(fused.fused_residual_vs_prior, fused.fused_prob_responder-fused.treatment_prior, atol=1e-12, rtol=0))
    files = [handoff / "visium_fused_teacher_table.tsv", handoff / "model_input_numeric.csv", handoff / "feature_manifest.csv"]
    precomputed.prepare(*files, output / "precomputed_handoff", *[digest(p) for p in files])
    for p in files:
        check("precomputed_bytes_preserved_"+p.name, digest(p) == digest(output / "precomputed_handoff" / p.name))

    print("Smoke: training-only selection, durable permutations and predictor reload", flush=True)
    candidates = pd.DataFrame({"drug_key": keys, "support_status": config["scope"]})
    metrics, heldout, importances, bundles, bundle_manifest = [], [], [], [], []
    config_model = {**config, "random_state": config["seed"], "target_col": "fused_residual_vs_prior"}
    pairs = fused[["sample_id", "drug_key", "fused_residual_vs_prior"]].merge(training, on="sample_id", validate="many_to_one")
    fixture_inputs = output / "conditional_fixture_inputs"
    write_table(pairs, fixture_inputs / "03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv")
    write_table(pd.DataFrame({"feature_name": numeric_columns}), fixture_inputs / "03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv")
    write_table(candidates, fixture_inputs / "07_label_shuffle_handoff/tier1_label_shuffle_candidates.tsv")
    # These are explicitly fixture task inputs, not a claimed production registry
    # or Tier 1 discovery result. Maintained preparation binds their bytes, code,
    # environment and stable original task indices into the resumable signatures.
    task_paths = cv.prepare_tasks(argparse.Namespace(**config_model, output_root=output / "conditional_smoke",
        dataset_root=fixture_inputs, step05_root=fixture_inputs, step08_root=fixture_inputs, block_size=2))
    for task_index, key in enumerate(keys):
        sub = pairs[pairs.drug_key == key].reset_index(drop=True)
        task_root = task_paths[task_index].parent
        task = json.loads(task_paths[task_index].read_text(encoding="utf-8"))
        try:
            cv.run_task(task_root / "task.json", stop_after_blocks=1)
        except InterruptedError:
            check("injected_interruption_"+str(task_index), True)
        observed_manifest = task_root / "blocks/observed/manifest.json"
        saved_hash = digest(observed_manifest)
        cv.run_task(task_root / "task.json")
        check("successful_checkpoint_reused_"+str(task_index), digest(observed_manifest) == saved_hash)
        for ids in cv.expected_blocks(task):
            path = task_root / "blocks" / cv.block_name(ids)
            check("verified_numerical_block_"+str(task_index)+"_"+path.name, cv.verify_block(path, task, ids))
            metrics.append(pd.read_parquet(path / "metrics.parquet"))
            heldout.append(pd.read_parquet(path / "predictions.parquet"))
        selected = select_features_training_only(sub[numeric_columns], sub.fused_residual_vs_prior, numeric_columns,
            max_features=config["max_features_per_split"], min_variance=1e-12)
        estimator = make_xgb_pipeline(random_state=config["seed"]+task_index, n_estimators=config["n_estimators"],
            max_depth=config["max_depth"], learning_rate=config["learning_rate"], n_jobs=1, tree_method="hist")
        estimator.fit(sub[selected], sub.fused_residual_vs_prior)
        prior = float(fused.loc[fused.drug_key == key, "treatment_prior"].iloc[0])
        bundle = SpatialPredictorBundle(key, selected, estimator, prior, training_ids, feature_reference=reference, support_status=config["scope"])
        bundle_path = output / "predictors" / f"model_{task_index:02d}.joblib"
        save_bundle(bundle, bundle_path)
        reloaded = load_bundle(bundle_path)
        pd.testing.assert_frame_equal(bundle.predict(external), reloaded.predict(external))
        pd.testing.assert_frame_equal(reloaded.predict(external), reloaded.predict(external_raw, representation="raw_reference"))
        predictions = reloaded.predict(external)
        batch = pd.concat([reloaded.predict(external.iloc[[i]]) for i in range(len(external))], ignore_index=True)
        pd.testing.assert_frame_equal(predictions, batch)
        reordered = reloaded.predict(external.iloc[::-1, ::-1]).iloc[::-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(predictions, reordered)
        explicit_na = external.copy()
        explicit_na.loc[0, selected[0]] = np.nan
        na_predictions = reloaded.predict(explicit_na)
        check("predictor_explicit_na_uses_fitted_imputer_"+str(task_index), na_predictions.n_training_median_imputed_features.iloc[0] == 1)
        pd.testing.assert_frame_equal(na_predictions.iloc[[0]].reset_index(drop=True), reloaded.predict(explicit_na.iloc[[0]]))
        rejects("missing_predictor_feature_"+str(task_index), lambda: reloaded.predict(external.drop(columns=selected[0])))
        rejects("predictor_train_identifier_collision_"+str(task_index), lambda: reloaded.predict(training.iloc[[0]]))
        check("predictor_reload_batch_order_raw_reference_"+str(task_index), True)
        write_table(predictions, output / "predictions" / f"model_{task_index:02d}.tsv")
        write_table(reloaded.contributions(external), output / "predictions" / f"contributions_{task_index:02d}.tsv")
        bundles.append(reloaded)
        bundle_manifest.append({"drug_key": key, "bundle_file": bundle_path.name, "sha256": digest(bundle_path),
                                "n_features": len(selected), "support_status": config["scope"]})
        importances.extend({"drug_key": key, "feature_name": f, "gain_importance": float(w)}
            for f, w in zip(selected, estimator.named_steps["model"].feature_importances_))
    write_table(pd.DataFrame(bundle_manifest), output / "predictors/bundle_manifest.tsv")
    command("prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/15_predict_spatial_features.py", [
        "--bundles", output / "predictors", "--features", output / "features/external_numeric.csv",
        "--output", output / "predictions/all_model_predictions.tsv", "--contributions", output / "predictions/all_feature_contributions.tsv"])
    expected_predictions = pd.concat([b.predict(external) for b in bundles], ignore_index=True)
    pd.testing.assert_frame_equal(expected_predictions, pd.read_csv(output / "predictions/all_model_predictions.tsv", sep="\t"),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    check("fitted_prediction_cli_matches_api", True, "1e-12 tolerance only for CSV float serialization")
    metrics = pd.concat(metrics, ignore_index=True)
    heldout = pd.concat(heldout, ignore_index=True)
    check("all_smoke_split_ids_present", len(metrics) == len(keys)*(config["n_shuffles"]+1)*config["n_repeats"] and not metrics.duplicated(["drug_key", "shuffle_id", "repeat"]).any())
    for identity, group in heldout.groupby(["drug_key", "shuffle_id", "repeat"]):
        row = metrics[(metrics.drug_key == identity[0]) & (metrics.shuffle_id == identity[1]) & (metrics.repeat == identity[2])].iloc[0]
        recomputed = cv.metrics_from_predictions(group.y_true, group.prediction, group.baseline_prediction)
        for name in cv.METRICS:
            check("recomputed_"+str(identity)+name, np.isclose(row[name], recomputed[name], rtol=0, atol=1e-12, equal_nan=True))
    summaries = cv.summarize(metrics[metrics.shuffle_id == -1], metrics[metrics.shuffle_id >= 0], candidates, config_model)
    write_table(metrics, output / "conditional_smoke/split_metrics.tsv")
    write_table(heldout, output / "conditional_smoke/heldout_predictions.tsv")
    write_table(summaries[-1], output / "conditional_smoke/conditional_decisions_SMOKE_ONLY.tsv")

    print("Smoke: maintained interpretation and separate signed alignment", flush=True)
    importance = pd.DataFrame(importances)
    dictionary = dictionary_module.normalize_feature_table(pd.DataFrame({"feature_name": sorted(set(importance.feature_name))}))
    effects = pim.compute_treatment_feature_effects(pairs, importance, dictionary, "fused_residual_vs_prior")
    themes = pim.compute_theme_effects(effects)
    write_table(effects, output / "interpretation/signed_feature_effects.tsv")
    write_table(themes, output / "interpretation/signed_theme_effects.tsv")
    atlas = fit_reference(training, dictionary.feature_name.tolist())
    atlas["effects"] = effects.astype(object).where(pd.notna(effects), None).to_dict("records")
    atlas["treatment_cards"] = [{"drug_key": k} for k in keys]
    atlas["source_estimator_validation_scope"] = config["scope"]
    atlas["predictive_performance_status"] = "NOT_EVALUATED_FOR_THIS_ALIGNMENT_RULE"
    save_reference(atlas, output / "interpretation/alignment_fixture.json")
    alignment, contributions = apply_atlas(output / "interpretation/alignment_fixture.json", external, output / "alignment")
    rerun, _ = apply_atlas(output / "interpretation/alignment_fixture.json", external.iloc[::-1, ::-1], output / "alignment_reordered")
    pd.testing.assert_frame_equal(alignment.sort_values(["sample_id", "drug_key"]).reset_index(drop=True), rerun.sort_values(["sample_id", "drug_key"]).reset_index(drop=True))
    single = [apply_atlas(output / "interpretation/alignment_fixture.json", external.iloc[[i]], output / "alignment_single" / str(i))[0] for i in range(len(external))]
    pd.testing.assert_frame_equal(alignment.sort_values(["sample_id", "drug_key"]).reset_index(drop=True), pd.concat(single).sort_values(["sample_id", "drug_key"]).reset_index(drop=True))
    rejects("alignment_train_identifier_collision", lambda: apply_atlas(output / "interpretation/alignment_fixture.json", training.iloc[[0]], output / "invalid_alignment"))
    rejects("alignment_missing_required_feature", lambda: apply_atlas(output / "interpretation/alignment_fixture.json", external.drop(columns=dictionary.feature_name.iloc[0]), output / "invalid_alignment"))
    missing_alignment = external.copy()
    missing_alignment.loc[0, effects.sort_values("effect_weight", ascending=False).feature_name.iloc[0]] = np.nan
    missing_scores, _ = apply_atlas(output / "interpretation/alignment_fixture.json", missing_alignment, output / "alignment_explicit_na")
    check("alignment_explicit_na_coverage_retained", missing_scores.loc[missing_scores.sample_id == external.sample_id.iloc[0], "observed_weight_fraction"].min() < 1)
    command("prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/alignment_bundle.py", [
        "score", "--bundle", output / "interpretation/alignment_fixture.json", "--feature-table", output / "features/external_numeric.csv", "--output-root", output / "alignment_cli"])
    pd.testing.assert_frame_equal(alignment, pd.read_csv(output / "alignment_cli/sample_treatment_alignment.tsv", sep="\t"), check_exact=False, rtol=1e-12, atol=1e-12)
    check("alignment_cli_matches_api", True, "1e-12 tolerance only for CSV float serialization")
    check("alignment_reload_batch_order", True)
    check("alignment_exact_sample_profile_set", len(alignment) == len(external)*len(keys) and not alignment.duplicated(["sample_id", "drug_key"]).any())
    check("alignment_and_prediction_quantities_separate", "predicted_residual_vs_prior" not in alignment and "signed_spatial_alignment" in alignment)
    write_table(pd.DataFrame(checks), output / "qc_checks.tsv")
    for imported in list(sys.modules.values()):
        filename = getattr(imported, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                sources[path.relative_to(ROOT).as_posix()] = digest(path)
    output_hashes = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    # Only numerical tables enter the reproducibility fingerprint; paths/timestamps
    # and fixture container metadata remain separate provenance fields.
    fingerprint_files = ["features/model_input_numeric.csv", "teacher/pseudobulk.tsv", "conditional_smoke/split_metrics.tsv",
                         "conditional_smoke/heldout_predictions.tsv", "interpretation/signed_feature_effects.tsv",
                         "interpretation/signed_theme_effects.tsv", "alignment/sample_treatment_alignment.tsv"]
    fingerprint = cv.stable_hash({p: output_hashes[p] for p in fingerprint_files})
    report = {"status": "PASS", "scope": config["scope"], "checks_passed": len(checks), "checks_failed": 0,
        "training_sections": len(training), "external_sections": len(external), "teacher_rows": len(fused),
        "teacher_modalities": fused.modality_used.value_counts().to_dict(), "model_bundles": len(bundles),
        "model_prediction_rows": len(external)*len(bundles), "alignment_rows": len(alignment), "signed_feature_effects": len(effects),
        "permutations_per_profile": config["n_shuffles"], "repeated_splits": config["n_repeats"],
        "conditional_smoke_accepted_profiles": int(summaries[-1].validated_for_step10.sum()),
        "executed": ["seeded raw 10x-style MTX and coordinate loading", "retained-spot/gene QC and normalization/log/HVG/scaling; PCA, neighbors, UMAP and Leiden",
            "marker program scoring and physical-neighbor axis refinement", "saved expression base-model/sigmoid calibration fixture",
            "saved trained probability fixture supplying the histology modality interface", "governed fusion Step04 and handoff Step05",
            "hash-bound precomputed three-file handoff interface", "training-only feature selection and fitted XGBoost fixture",
            "durable conditional permutation blocks, injected interruption/resume and prediction-based metric recomputation",
            "maintained signed feature/theme effects", "saved fitted prediction/contributions and separate saved alignment/contributions", "numerical, identity, missing-feature, reload, ordering and batch QC"],
        "excluded": ["real clinical response data and scientific model validation", "original expression/histology cohort training",
            "histology CNN/tile extraction and pretrained image weights", "external gene-set downloads and full architecture extraction",
            "full pooled supervised registry and full treatment-candidate discovery", "production1000-permutation evaluation and manuscript/figure generation"],
        "fixture_cautions": "Teacher targets and trained teacher objects are synthetic; fitted smoke bundles are exported regardless of the smoke conditional decision and never marked independently supported.",
        "production_settings_modified": False, "network_or_private_inputs_required": False,
        "candidate_root": str(ROOT), "python": sys.executable, "sources": sources, "commands": commands,
        "environment_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "scanpy", "anndata", "h5py", "pyarrow", "joblib"]},
        "numerical_fingerprint": fingerprint, "numerical_fingerprint_files": fingerprint_files,
        "elapsed_seconds": time.monotonic()-started, "outputs_sha256": output_hashes}
    cv.atomic_json(output / "smoke_report.json", report)
    print(json.dumps({k: report[k] for k in ["status", "scope", "checks_passed", "teacher_rows", "model_prediction_rows", "alignment_rows", "numerical_fingerprint", "elapsed_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reviewer_smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        with threadpool_limits(limits=1):
            run(args.output, config)
    except Exception as error:
        # Retain partial outputs and an explicit failure instead of leaving a
        # successful-looking output directory. Never replace a prior PASS.
        report_path = args.output / "smoke_report.json"
        if args.output.is_dir() and not report_path.exists():
            cv.atomic_json(report_path, {"status": "FAIL", "scope": config.get("scope"),
                "error_type": type(error).__name__, "error": str(error), "partial_outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
