"""
Script: 05_build_prediction_ready_teacher.py

Purpose:
    Build the final prediction-ready teacher handoff tables for downstream
    spatial response modeling.

Project context:
    This is Step 05 of the governed teacher_builder workflow. It reads the
    spatial feature table from the spatial feature identification pipeline and
    the fused sample-by-treatment teacher labels from Step 04, selects usable
    numeric spatial features, joins teacher labels to sample-level features,
    and writes the canonical model-input and teacher-table files expected by
    downstream spatial prediction workflows.

Scientific role:
    This step is the handoff boundary between teacher construction and response
    prediction. It does not train a model. Instead, it preserves governed teacher
    labels, treatment priors, modality availability, residual targets, label
    quality fields, and numeric spatial covariates in reproducible tables so
    later models can learn from auditable teacher labels.

Documentation polish marker:
    TEACHER_BUILDER_STEP05_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, paths, thresholds,
    schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 05 uses pandas/numpy and shared governance helpers to build the model handoff tables.

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd



# =========================
# Shared governance helper imports
# =========================
# Shared helpers keep config loading, table IO, and feature-manifest logic consistent.

from teacher_governance_lib import (
    load_config,
    cfg_path,
    ensure_dir,
    read_table,
    write_table,
    basic_numeric_feature_manifest,
)




# =========================
# Command-line interface
# =========================
# The governed runner passes the YAML config path into this prediction-ready step.

def parse_args():
    """Parse the governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Prediction-ready teacher workflow
# =========================
# Main workflow: load spatial features and fused teacher labels, select numeric features, join tables, and write handoff outputs.

def main():
    """Build model-input, teacher, feature-manifest, training, and summary handoff tables."""

    args = parse_args()
    cfg = load_config(args.config)



    # =========================
    # Output directory and run configuration
    # =========================
    # Step 05 writes all final handoff files under 05_prediction_ready_teacher.

    # Keep Step 05 outputs under the governed teacher_builder output root.
    out_root = Path(cfg["output_root"])
    out_dir = ensure_dir(out_root / "05_prediction_ready_teacher")

    # The sample column links spatial features to fused teacher rows.
    sample_col = cfg.get("sample_col", "sample_id")



    # =========================
    # Input path resolution
    # =========================
    # The spatial feature table and fused teacher table are the two required Step 05 inputs.

    spatial_path = cfg_path(cfg, "spatial_feature_table")
    manifest_path = cfg_path(cfg, "spatial_feature_manifest")
    # Fused teacher labels from Step 04 are the supervised target handoff.
    teacher_path = out_root / "04_fused_teacher" / "fused_teacher_table.tsv"

    print("")
    print("Prediction ready teacher table")
    print("=" * 70)
    print("spatial_feature_table:", spatial_path)
    print("teacher_path:", teacher_path)



    # =========================
    # Input loading and validation
    # =========================
    # Spatial features and fused teacher labels are loaded before feature selection.

    # Load the spatial feature matrix and governed fused teacher table.
    spatial = read_table(spatial_path)
    teacher = read_table(teacher_path)



    # =========================
    # Sample identifier validation
    # =========================
    # The spatial feature table must contain the configured sample identifier column.

    if sample_col not in spatial.columns:
        raise ValueError(f"Spatial table missing sample column: {sample_col}")



    # =========================
    # Sample de-duplication and test-mode filtering
    # =========================
    # The model-input table is one row per spatial sample, with optional smoke-test truncation.

    # Keep one feature row per sample before joining to many treatment rows.
    spatial = spatial.drop_duplicates(sample_col).copy()

    # Smoke-test mode limits both spatial rows and teacher rows consistently.
    if bool(cfg.get("test_mode", False)):
        keep_samples = spatial[sample_col].astype(str).drop_duplicates().tolist()[: int(cfg.get("test_n_samples", 5))]
        spatial = spatial[spatial[sample_col].astype(str).isin(keep_samples)].copy()
        teacher = teacher[teacher["sample_id"].astype(str).isin(keep_samples)].copy()



    # =========================
    # Numeric feature-selection thresholds
    # =========================
    # Feature inclusion is controlled by minimum nonmissing fraction and optional constant-feature removal.

    # Feature inclusion thresholds are config-driven for reproducible handoffs.
    min_nonmissing = float(cfg.get("feature_min_nonmissing_fraction", 0.2))
    drop_constant = bool(cfg.get("drop_constant_features", True))



    # =========================
    # Feature manifest handling
    # =========================
    # A provided manifest is reused when possible, otherwise a numeric feature manifest is rebuilt.

    # Reuse upstream feature manifests when available, otherwise infer one from the spatial table.
    if manifest_path is not None and manifest_path.exists():
        manifest = read_table(manifest_path)
        if "feature" not in manifest.columns:
            manifest = basic_numeric_feature_manifest(spatial, sample_col, min_nonmissing, drop_constant)
    else:
        manifest = basic_numeric_feature_manifest(spatial, sample_col, min_nonmissing, drop_constant)



    # =========================
    # Numeric spatial feature discovery
    # =========================
    # Candidate feature columns are tested for numeric coercion, nonmissing support, and variability.

    # All non-sample spatial columns are candidates for numeric model features.
    candidate_cols = [c for c in spatial.columns if c != sample_col]
    numeric_cols = []



    # =========================
    # Feature inclusion loop
    # =========================
    # Each candidate spatial feature is independently screened against the configured thresholds.

    for c in candidate_cols:
        # Numeric coercion is used for feature screening without changing original spatial columns.
        s = pd.to_numeric(spatial[c], errors="coerce")
        nonmissing = float(s.notna().mean())
        n_unique = int(s.dropna().nunique())

        # Exclude features that are too sparse for downstream modeling.
        if nonmissing < min_nonmissing:
            continue
        # Constant features carry no sample-level signal and can be dropped when configured.
        if drop_constant and n_unique <= 1:
            continue
        numeric_cols.append(c)



    # =========================
    # Model-input matrix construction
    # =========================
    # The downstream model matrix keeps only the sample ID and selected numeric spatial features.

    # model_input_numeric.csv is the sample-by-feature matrix for downstream models.
    model_input = spatial[[sample_col] + numeric_cols].copy()



    # =========================
    # Numeric coercion
    # =========================
    # Selected features are coerced to numeric values before writing model_input_numeric.csv.

    for c in numeric_cols:
        model_input[c] = pd.to_numeric(model_input[c], errors="coerce")



    # =========================
    # Teacher-feature training table assembly
    # =========================
    # The prediction-ready training table joins every teacher row to the matching sample-level feature row.

    # Join each sample-treatment teacher label to the matching spatial feature vector.
    training = teacher.merge(model_input, left_on="sample_id", right_on=sample_col, how="left", suffixes=("_teacher", ""))



    # =========================
    # Primary output table writing
    # =========================
    # Step 05 writes the canonical files consumed by downstream spatial prediction pipelines.

    # These output filenames are the downstream spatial prediction contract.
    write_table(model_input, out_dir / "model_input_numeric.csv", sep=",")
    write_table(teacher, out_dir / "visium_fused_teacher_table.tsv")
    write_table(training, out_dir / "prediction_ready_training_table.tsv")



    # =========================
    # Final feature manifest annotation
    # =========================
    # The output manifest records which candidate features were retained for modeling.

    # Ensure the feature manifest includes an explicit inclusion flag.
    if "included" not in manifest.columns:
        manifest = basic_numeric_feature_manifest(spatial, sample_col, min_nonmissing, drop_constant)

    if "included" in manifest.columns:
        # Mark exactly the numeric columns retained in model_input_numeric.csv.
        manifest["included"] = manifest["feature"].isin(numeric_cols)
        manifest.loc[manifest["included"], "reason"] = "included"
        manifest.loc[~manifest["included"] & manifest["reason"].isna(), "reason"] = "excluded_by_step05"

    manifest.to_csv(out_dir / "feature_manifest.csv", index=False)



    # =========================
    # Summary and run-config construction
    # =========================
    # Summary outputs record row counts, feature counts, teacher coverage, and retained governance fields.

    summary = pd.DataFrame(
        [
            {
                "n_spatial_rows_raw": len(spatial),
                "n_model_input_samples": model_input[sample_col].nunique(),
                "n_teacher_rows": len(teacher),
                "n_teacher_samples": teacher["sample_id"].nunique(),
                "n_teacher_treatments": teacher["drug_key"].nunique(),
                "n_prediction_ready_rows": len(training),
                "n_prediction_ready_samples": training["sample_id"].nunique(),
                "n_prediction_ready_treatments": training["drug_key"].nunique(),
                "n_features_included": len(numeric_cols),
                "n_features_total_candidates": len(candidate_cols),
                "mean_fused_prob_responder": float(pd.to_numeric(teacher["fused_prob_responder"], errors="coerce").mean()) if len(teacher) else np.nan,
                "mean_fused_residual_vs_prior": float(pd.to_numeric(teacher["fused_residual_vs_prior"], errors="coerce").mean()) if "fused_residual_vs_prior" in teacher.columns and len(teacher) else np.nan,
                "mean_fused_confidence": float(pd.to_numeric(teacher["fused_confidence"], errors="coerce").mean()) if len(teacher) else np.nan,
            }
        ]
    )

    write_table(summary, out_dir / "prediction_ready_summary.tsv")



    # =========================
    # Machine-readable run configuration
    # =========================
    # run_config.json records resolved inputs, output directory, thresholds, and governance columns retained.

    # Retain machine-readable provenance for inputs, thresholds, and governance columns.
    run_config = {
        "config": str(Path(args.config).resolve()),
        "fused_teacher_input": str(teacher_path),
        "spatial_feature_input": str(spatial_path),
        "output_dir": str(out_dir),
        "sample_col": sample_col,
        "feature_min_nonmissing_fraction": min_nonmissing,
        "drop_constant_features": drop_constant,
        "n_numeric_features": len(numeric_cols),
        "n_teacher_rows": len(teacher),
        "n_training_rows": len(training),
        "governance_columns_retained": [
            c
            for c in [
                "fused_prob_responder",
                "fused_residual_vs_prior",
                "treatment_prior",
                "prior_source",
                "prior_n",
                "modality_used",
                "expression_effective_weight",
                "histology_effective_weight",
                "label_quality_flag",
                "label_quality_reason",
                "histology_control_warning",
            ]
            if c in training.columns
        ],
    }

    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")



    # =========================
    # Human-readable prediction-ready summary
    # =========================
    # The text summary is intended for terminal review and GitHub/publication audit.

    lines = [
        "Prediction ready governed teacher summary",
        "",
        f"model input samples: {model_input[sample_col].nunique()}",
        f"numeric features: {len(numeric_cols)}",
        f"teacher rows: {len(teacher)}",
        f"teacher samples: {teacher['sample_id'].nunique()}",
        f"teacher treatments: {teacher['drug_key'].nunique()}",
        f"prediction ready rows: {len(training)}",
        f"prediction ready samples: {training['sample_id'].nunique()}",
        "",
        "Governance columns retained:",
        ", ".join(run_config["governance_columns_retained"]),
        "",
        "Outputs:",
        str(out_dir / "model_input_numeric.csv"),
        str(out_dir / "visium_fused_teacher_table.tsv"),
        str(out_dir / "prediction_ready_training_table.tsv"),
        str(out_dir / "feature_manifest.csv"),
    ]

    # Human-readable summary mirrors the key counts printed to the terminal.
    summary_text = "\n".join(lines)
    (out_dir / "prediction_ready_summary.txt").write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print("DONE")
    print(out_dir)


if __name__ == "__main__":
    main()
