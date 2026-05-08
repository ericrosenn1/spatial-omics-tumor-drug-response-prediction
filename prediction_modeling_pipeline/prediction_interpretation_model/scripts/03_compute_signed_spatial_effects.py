#!/usr/bin/env python
"""
Script:
    03_compute_signed_spatial_effects.py

Description:
    Converts unsigned V2 feature-importance evidence into signed spatial effects
    by correlating strict spatial feature values with V2 residual response targets.
    Positive treatment effects indicate above-prior residual response association.

Instructions:
    Run after Step 02. Inspect signed feature/theme QC before building treatment
    cards or sample-level interpretations.

Source-truth policy:
    Directionality is correlation-based and model-importance-weighted. It is an
    interpretable association screen, not causal proof or treatment guidance.
"""

# =============================================================================
# PIM_DOCS_PATCH: RUN AND MAINTENANCE INSTRUCTIONS
# =============================================================================
# Run numbered scripts through 00_run_prediction_interpretation_model.py unless
# debugging a single step. Treat the V2 full-run root as read-only source truth.
# Every generated .txt report must start with FILEPATH, and terminal summaries
# should remain concise enough for copy/paste debugging.
# =============================================================================


# =============================================================================
# PIM_DOCS_SECTION: imports and dependencies
# =============================================================================
# Keep imports explicit and standard-library-first where practical. The pipeline
# expects local scripts to run from the scripts directory or through the orchestrator.

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import traceback
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from _pim_utils import (
    add_qc,
    choose_col,
    corr_pair,
    effect_label,
    ensure_dir,
    evidence_grade,
    load_prepared_index,
    numeric_series,
    open_folder,
    read_header,
    read_source_table,
    read_table,
    save_output_manifest,
    source_path,
    summarize_examples,
    write_json,
    write_text_report,
    write_tsv,
)


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    Defaults preserve local project paths while allowing explicit overrides."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--target-col", default="fused_residual_vs_prior")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def read_pair_minimal(index_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, str, str, str]:
    """Read only required pair-level columns from the large V2 table.
    Keeps signed-effect and sample-level steps memory-conscious."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    pair_path = source_path(index_df, "v2_pair_level_residual_dataset", prefer_copied=False)
    columns = read_header(pair_path)

    sample_col = choose_col(columns, ["sample_id", "slide_id", "sample"], required=True, label="sample column")
    treatment_col = choose_col(columns, ["drug_key", "treatment_key", "drug", "treatment"], required=True, label="treatment column")

    possible = [sample_col, treatment_col, target_col, "fused_prob_responder", "treatment_prior", "prior_prob_responder"]
    usecols = [c for c in possible if c in columns]
    if target_col not in usecols:
        raise ValueError(f"Pair-level dataset does not contain required target column: {target_col}")

    pair = read_table(pair_path, usecols=usecols)
    if sample_col != "sample_id":
        pair = pair.rename(columns={sample_col: "sample_id"})
        sample_col = "sample_id"
    if treatment_col != "drug_key":
        pair = pair.rename(columns={treatment_col: "drug_key"})
        treatment_col = "drug_key"
    return pair, sample_col, treatment_col, target_col


def load_spatial_features(index_df: pd.DataFrame, feature_names: Sequence[str]) -> tuple[pd.DataFrame, List[str]]:
    """Load requested spatial feature columns from the V2 spatial table.
    Returns only features present in the source table."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    spatial = read_source_table(index_df, "v2_spatial_features_broad_pool")
    sample_col = choose_col(spatial.columns, ["sample_id", "slide_id", "sample"], required=True, label="spatial sample column")
    if sample_col != "sample_id":
        spatial = spatial.rename(columns={sample_col: "sample_id"})

    features = [f for f in feature_names if f in spatial.columns]
    keep = ["sample_id"] + features
    return spatial[keep].copy(), features


def normalize_importance_by_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature importance within each treatment.
    Combines SHAP and gain evidence into comparable treatment-level weights."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    out = df.copy()
    if "mean_abs_shap" in out.columns:
        out["importance_raw"] = pd.to_numeric(out["mean_abs_shap"], errors="coerce")
    else:
        out["importance_raw"] = np.nan
    if "gain_importance" in out.columns:
        out["importance_raw"] = out["importance_raw"].fillna(pd.to_numeric(out["gain_importance"], errors="coerce"))
    out["importance_raw"] = out["importance_raw"].fillna(0.0).clip(lower=0.0)

    totals = out.groupby("drug_key")["importance_raw"].transform("sum")
    out["importance_normalized_within_treatment"] = np.where(totals > 0, out["importance_raw"] / totals, 0.0)
    return out


def compute_treatment_feature_effects(
    merged: pd.DataFrame,
    shap_df: pd.DataFrame,
    feature_dict: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Compute signed feature effects per treatment.
    Weights direction by model importance and feature-target association."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows: List[dict] = []

    feature_meta_cols = [
        "feature_name",
        "feature_label",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]
    meta = feature_dict[[c for c in feature_meta_cols if c in feature_dict.columns]].drop_duplicates("feature_name")

    shap_df = normalize_importance_by_treatment(shap_df)
    shap_df = shap_df.merge(meta, on="feature_name", how="left", suffixes=("", "_dictionary"))

    for drug_key, evidence in shap_df.groupby("drug_key", dropna=False):
        sub = merged[merged["drug_key"].astype(str) == str(drug_key)].copy()
        if sub.empty:
            continue

        for _, ev in evidence.iterrows():
            feature = str(ev["feature_name"])
            if feature not in sub.columns:
                continue

            pearson, n = corr_pair(sub[feature], sub[target_col], method="pearson")
            spearman, _ = corr_pair(sub[feature], sub[target_col], method="spearman")
            abs_corr = abs(pearson) if np.isfinite(pearson) else np.nan
            sign = 0 if not np.isfinite(pearson) else (1 if pearson > 0 else (-1 if pearson < 0 else 0))

            importance = float(ev.get("importance_normalized_within_treatment", 0.0) or 0.0)
            signed_effect = sign * importance * (float(abs_corr) if np.isfinite(abs_corr) else 0.0)

            rows.append({
                "drug_key": drug_key,
                "feature_name": feature,
                "feature_label": ev.get("feature_label", ""),
                "feature_group": ev.get("feature_group", ""),
                "feature_axis": ev.get("feature_axis", ""),
                "biological_theme": ev.get("biological_theme", "other interpretable spatial signal"),
                "interpretation_class": ev.get("interpretation_class", ""),
                "interpretation_note": ev.get("interpretation_note", ""),
                "target_col": target_col,
                "n_samples_with_feature_and_target": n,
                "feature_target_pearson": pearson,
                "feature_target_spearman": spearman,
                "direction_sign": sign,
                "direction_label": effect_label(pearson),
                "abs_feature_target_pearson": abs_corr,
                "importance_raw": ev.get("importance_raw", 0.0),
                "importance_normalized_within_treatment": importance,
                "signed_effect": signed_effect,
                "effect_weight": abs(signed_effect),
                "evidence_grade": evidence_grade(abs_corr, n),
                "v2_gain_importance": ev.get("gain_importance", np.nan),
                "v2_mean_abs_shap": ev.get("mean_abs_shap", np.nan),
                "v2_shap_status": ev.get("shap_status", ""),
                "final_model_rank": ev.get("final_model_rank", ""),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["drug_key", "effect_weight", "importance_normalized_within_treatment"], ascending=[True, False, False])
    return out


def compute_theme_effects(feature_effects: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signed feature effects into biology-theme effects.
    Produces treatment-theme directionality for cards and mechanism atlas."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if feature_effects.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for (drug_key, theme), sub in feature_effects.groupby(["drug_key", "biological_theme"], dropna=False):
        signed_sum = float(pd.to_numeric(sub["signed_effect"], errors="coerce").sum())
        abs_sum = float(pd.to_numeric(sub["effect_weight"], errors="coerce").sum())
        pos_count = int((pd.to_numeric(sub["signed_effect"], errors="coerce") > 0).sum())
        neg_count = int((pd.to_numeric(sub["signed_effect"], errors="coerce") < 0).sum())
        dominant = "sensitivity_associated" if signed_sum > 0 else ("resistance_associated" if signed_sum < 0 else "balanced_or_ambiguous")

        rows.append({
            "drug_key": drug_key,
            "biological_theme": theme,
            "n_features": int(sub["feature_name"].nunique()),
            "positive_feature_count": pos_count,
            "negative_feature_count": neg_count,
            "signed_theme_effect": signed_sum,
            "absolute_theme_effect": abs_sum,
            "dominant_direction_label": dominant,
            "mean_feature_target_pearson": pd.to_numeric(sub["feature_target_pearson"], errors="coerce").mean(),
            "top_features": summarize_examples(sub.sort_values("effect_weight", ascending=False)["feature_name"], 8),
        })

    return pd.DataFrame(rows).sort_values(["drug_key", "absolute_theme_effect"], ascending=[True, False])


def compute_broad_effects(index_df: pd.DataFrame, feature_dict: pd.DataFrame, spatial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute broad sample-level signed effects.
    Summarizes feature and theme directionality for V2 broad residual targets."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        broad_pred = read_source_table(index_df, "broad_residual_test_predictions_long")
        broad_ev = read_source_table(index_df, "broad_residual_feature_evidence_long")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {"target_col", "sample_id", "target"}
    if not required_cols.issubset(set(broad_pred.columns)):
        return pd.DataFrame(), pd.DataFrame()
    if not {"target_col", "feature_name"}.issubset(set(broad_ev.columns)):
        return pd.DataFrame(), pd.DataFrame()

    targets = (
        broad_pred
        .groupby(["target_col", "sample_id"], as_index=False)
        .agg(target=("target", "mean"))
    )

    ev = broad_ev.copy()
    if "mean_abs_shap" in ev.columns:
        ev["importance_raw"] = pd.to_numeric(ev["mean_abs_shap"], errors="coerce")
    else:
        ev["importance_raw"] = np.nan
    if "gain_importance" in ev.columns:
        ev["importance_raw"] = ev["importance_raw"].fillna(pd.to_numeric(ev["gain_importance"], errors="coerce"))
    ev["importance_raw"] = ev["importance_raw"].fillna(0.0).clip(lower=0.0)

    ev_summary = (
        ev.groupby(["target_col", "feature_name"], as_index=False)
        .agg(importance_raw=("importance_raw", "mean"))
    )
    totals = ev_summary.groupby("target_col")["importance_raw"].transform("sum")
    ev_summary["importance_normalized_within_target"] = np.where(totals > 0, ev_summary["importance_raw"] / totals, 0.0)

    meta_cols = [c for c in ["feature_name", "feature_label", "feature_group", "biological_theme", "interpretation_class"] if c in feature_dict.columns]
    meta = feature_dict[meta_cols].drop_duplicates("feature_name")

    rows: List[dict] = []
    spatial_cols = set(spatial.columns)

    for target_col, ev_sub in ev_summary.groupby("target_col"):
        target_sub = targets[targets["target_col"] == target_col].merge(spatial, on="sample_id", how="left")
        for _, ev_row in ev_sub.iterrows():
            feature = str(ev_row["feature_name"])
            if feature not in spatial_cols:
                continue

            pearson, n = corr_pair(target_sub[feature], target_sub["target"], method="pearson")
            spearman, _ = corr_pair(target_sub[feature], target_sub["target"], method="spearman")
            sign = 0 if not np.isfinite(pearson) else (1 if pearson > 0 else (-1 if pearson < 0 else 0))
            abs_corr = abs(pearson) if np.isfinite(pearson) else np.nan
            importance = float(ev_row.get("importance_normalized_within_target", 0.0) or 0.0)

            rows.append({
                "broad_target_col": target_col,
                "feature_name": feature,
                "n_samples_with_feature_and_target": n,
                "feature_target_pearson": pearson,
                "feature_target_spearman": spearman,
                "direction_sign": sign,
                "direction_label": effect_label(pearson, positive_label="higher_feature_higher_target", negative_label="higher_feature_lower_target"),
                "importance_raw": ev_row.get("importance_raw", 0.0),
                "importance_normalized_within_target": importance,
                "signed_effect": sign * importance * (float(abs_corr) if np.isfinite(abs_corr) else 0.0),
                "effect_weight": importance * (float(abs_corr) if np.isfinite(abs_corr) else 0.0),
                "evidence_grade": evidence_grade(abs_corr, n),
            })

    broad_feature = pd.DataFrame(rows)
    if broad_feature.empty:
        return broad_feature, pd.DataFrame()

    broad_feature = broad_feature.merge(meta, on="feature_name", how="left")
    broad_feature = broad_feature.sort_values(["broad_target_col", "effect_weight"], ascending=[True, False])

    theme_rows: List[dict] = []
    for (target_col, theme), sub in broad_feature.groupby(["broad_target_col", "biological_theme"], dropna=False):
        signed_sum = float(pd.to_numeric(sub["signed_effect"], errors="coerce").sum())
        abs_sum = float(pd.to_numeric(sub["effect_weight"], errors="coerce").sum())
        theme_rows.append({
            "broad_target_col": target_col,
            "biological_theme": theme,
            "n_features": int(sub["feature_name"].nunique()),
            "signed_theme_effect": signed_sum,
            "absolute_theme_effect": abs_sum,
            "dominant_direction_label": "higher_feature_higher_target" if signed_sum > 0 else ("higher_feature_lower_target" if signed_sum < 0 else "balanced_or_ambiguous"),
            "top_features": summarize_examples(sub.sort_values("effect_weight", ascending=False)["feature_name"], 8),
        })

    broad_theme = pd.DataFrame(theme_rows).sort_values(["broad_target_col", "absolute_theme_effect"], ascending=[True, False])
    return broad_feature, broad_theme


# =============================================================================
# PIM_DOCS_SECTION: main entry point
# =============================================================================
# The main function wires inputs, output folders, QC checks, reports, and terminal summaries.

def main() -> int:
    """Run the script's command-line workflow.
    Writes outputs, QC checks, summaries, and terminal status messages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    args = parse_args()
    started = dt.datetime.now()
    output_root = Path(args.output_root)
    prepared_root, index_df = load_prepared_index(output_root, Path(args.prepared_input_root) if args.prepared_input_root else None)

    step02_root = output_root / "02_feature_and_treatment_dictionary"
    feature_dict_path = step02_root / "01_feature_dictionary" / "strict_spatial_feature_dictionary.tsv"
    treatment_dict_path = step02_root / "02_treatment_dictionary" / "treatment_dictionary.tsv"

    if not feature_dict_path.exists():
        raise FileNotFoundError(f"Step 02 feature dictionary not found: {feature_dict_path}")
    if not treatment_dict_path.exists():
        raise FileNotFoundError(f"Step 02 treatment dictionary not found: {treatment_dict_path}")

    feature_dict = read_table(feature_dict_path)
    treatment_dict = read_table(treatment_dict_path)

    step_root = output_root / "03_signed_spatial_effects"
    feature_dir = step_root / "01_treatment_feature_effects"
    theme_dir = step_root / "02_treatment_theme_effects"
    broad_dir = step_root / "03_broad_effects"
    matrix_dir = step_root / "04_matrices"
    qc_dir = step_root / "05_qc"
    report_dir = step_root / "06_reports"

    for path in [feature_dir, theme_dir, broad_dir, matrix_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        validated = treatment_dict[treatment_dict.get("label_shuffle_validated", False).astype(str).str.lower().isin(["true", "1", "yes"])] if "label_shuffle_validated" in treatment_dict.columns else pd.DataFrame()
        if validated.empty:
            validated_source = read_source_table(index_df, "integrated_validated_treatment_table")
            validated_keys = sorted(validated_source["drug_key"].dropna().astype(str).unique())
        else:
            validated_keys = sorted(validated["drug_key"].dropna().astype(str).unique())

        shap = read_source_table(index_df, "per_treatment_final_shap_feature_long")
        shap = shap[shap["drug_key"].astype(str).isin(validated_keys)].copy()

        all_needed_features = sorted(set(shap["feature_name"].dropna().astype(str)).union(set(feature_dict["feature_name"].dropna().astype(str))))
        spatial, present_features = load_spatial_features(index_df, all_needed_features)
        pair, sample_col, treatment_col, target_col = read_pair_minimal(index_df, args.target_col)

        pair = pair[pair["drug_key"].astype(str).isin(validated_keys)].copy()
        merged = pair.merge(spatial[["sample_id"] + present_features], on="sample_id", how="left")

        treatment_feature_effects = compute_treatment_feature_effects(merged, shap, feature_dict, target_col)
        treatment_theme_effects = compute_theme_effects(treatment_feature_effects)

        broad_feature_effects, broad_theme_effects = compute_broad_effects(index_df, feature_dict, spatial)

        write_tsv(feature_dir / "signed_treatment_feature_effects.tsv", treatment_feature_effects)
        write_tsv(theme_dir / "signed_treatment_theme_effects.tsv", treatment_theme_effects)
        write_tsv(broad_dir / "signed_broad_feature_effects.tsv", broad_feature_effects)
        write_tsv(broad_dir / "signed_broad_theme_effects.tsv", broad_theme_effects)

        if not treatment_feature_effects.empty:
            feature_matrix = treatment_feature_effects.pivot_table(
                index="drug_key",
                columns="feature_name",
                values="signed_effect",
                aggfunc="sum",
                fill_value=0.0,
            ).reset_index()
            write_tsv(matrix_dir / "treatment_feature_signed_effect_matrix.tsv", feature_matrix)
        else:
            write_tsv(matrix_dir / "treatment_feature_signed_effect_matrix.tsv", pd.DataFrame())

        if not treatment_theme_effects.empty:
            theme_matrix = treatment_theme_effects.pivot_table(
                index="drug_key",
                columns="biological_theme",
                values="signed_theme_effect",
                aggfunc="sum",
                fill_value=0.0,
            ).reset_index()
            write_tsv(matrix_dir / "treatment_theme_signed_effect_matrix.tsv", theme_matrix)
        else:
            write_tsv(matrix_dir / "treatment_theme_signed_effect_matrix.tsv", pd.DataFrame())

        add_qc(qc, "validated_treatments_for_signed_effects", "pass" if len(validated_keys) == 27 else "warn", len(validated_keys), 27, "Validated treatments used for signed interpretation.")
        add_qc(qc, "treatment_feature_effect_rows", "pass" if len(treatment_feature_effects) > 0 else "fail", len(treatment_feature_effects), ">0", "Signed per-treatment feature effects generated.")
        add_qc(qc, "treatment_theme_effect_rows", "pass" if len(treatment_theme_effects) > 0 else "fail", len(treatment_theme_effects), ">0", "Signed per-treatment theme effects generated.")
        add_qc(qc, "spatial_features_available_for_signing", "pass" if len(present_features) >= 139 else "warn", len(present_features), ">=139", "Strict spatial features available in V2 spatial feature table.")
        add_qc(qc, "pair_rows_used_for_validated_treatments", "pass" if len(pair) > 0 else "fail", len(pair), ">0", "Pair-level residual rows used for directionality.")
        add_qc(qc, "broad_feature_effect_rows", "pass" if len(broad_feature_effects) > 0 else "warn", len(broad_feature_effects), ">0", "Broad sample-level signed effects generated from Step 06 predictions.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        treatment_feature_effects = pd.DataFrame()
        treatment_theme_effects = pd.DataFrame()
        broad_feature_effects = pd.DataFrame()
        broad_theme_effects = pd.DataFrame()

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step03_signed_effect_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "target_col": args.target_col,
        "treatment_feature_effect_rows": len(treatment_feature_effects),
        "treatment_theme_effect_rows": len(treatment_theme_effects),
        "broad_feature_effect_rows": len(broad_feature_effects),
        "broad_theme_effect_rows": len(broad_theme_effects),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step03_summary.json", summary)

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 03 REPORT",
        "",
        f"status: {status}",
        f"target_col: {args.target_col}",
        f"prepared_input_root: {prepared_root}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Interpretation rule",
        "For treatment effects, positive signed_effect means the feature is associated with higher fused_residual_vs_prior for that treatment.",
        "In this project, higher fused_residual_vs_prior means above-prior response signal; lower values mean below-prior response signal.",
        "",
        "Outputs",
        f"signed_treatment_feature_effects: {feature_dir / 'signed_treatment_feature_effects.tsv'}",
        f"signed_treatment_theme_effects: {theme_dir / 'signed_treatment_theme_effects.tsv'}",
        f"signed_broad_feature_effects: {broad_dir / 'signed_broad_feature_effects.tsv'}",
        f"signed_broad_theme_effects: {broad_dir / 'signed_broad_theme_effects.tsv'}",
        f"treatment_feature_matrix: {matrix_dir / 'treatment_feature_signed_effect_matrix.tsv'}",
        f"treatment_theme_matrix: {matrix_dir / 'treatment_theme_signed_effect_matrix.tsv'}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Caveat",
        "Directionality is correlation-based and model-importance-weighted. It is an interpretable association screen, not causal proof.",
        "This step does not make treatment recommendations.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step03_signed_spatial_effects_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 03 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"treatment_feature_effect_rows: {len(treatment_feature_effects)}")
    print(f"treatment_theme_effect_rows: {len(treatment_theme_effects)}")
    print(f"broad_feature_effect_rows: {len(broad_feature_effects)}")
    print(f"report: {report_dir / 'step03_signed_spatial_effects_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())

