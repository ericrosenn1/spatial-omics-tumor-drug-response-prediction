#!/usr/bin/env python
"""
Script:
    02_build_feature_and_treatment_dictionary.py

Description:
    Builds readable dictionaries for strict spatial features, biology themes,
    treatment keys, treatment components, and source-column contracts from Step 01
    prepared V2 inputs.

Instructions:
    Run after Step 01 passes. Downstream signed-effect, treatment-card, mechanism
    atlas, and publication-output steps rely on these dictionaries for stable names,
    biological themes, treatment labels, and component classes.

Source-truth policy:
    This step consumes prepared V2 sources only. Treatment component classes are
    reporting labels, not clinical recommendations.
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
import os
from pathlib import Path
import sys
import traceback
from typing import List

import pandas as pd

from _pim_utils import (
    add_qc,
    build_output_manifest,
    choose_col,
    classify_component,
    clean_component,
    ensure_dir,
    humanize_feature,
    infer_feature_group,
    load_prepared_index,
    open_folder,
    parse_treatment_components,
    read_header,
    read_source_table,
    read_table,
    safe_slug,
    save_output_manifest,
    selected_columns,
    source_path,
    summarize_examples,
    write_json,
    write_text_report,
    write_tsv,
)


# =============================================================================
# PIM_DOCS_SECTION: constants and source contracts
# =============================================================================
# Constants define expected files, output names, QC contracts, or reporting rules.

FEATURE_META_COLS = [
    "feature_name",
    "feature_original",
    "feature_group",
    "feature_axis",
    "biological_theme",
    "interpretation_class",
    "interpretation_note",
]


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
    parser.add_argument("--project-root", default=r"D:\Adv_Omics_Fenyo\project")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def normalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize strict feature registry columns.
    Adds readable labels, fallback groups, and treatment-identity screening flags."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    out = df.copy()
    feature_col = choose_col(out.columns, ["feature_name", "feature", "spatial_feature", "model_feature", "variable"], required=True, label="feature column")
    if feature_col != "feature_name":
        out = out.rename(columns={feature_col: "feature_name"})

    for col in FEATURE_META_COLS:
        if col not in out.columns:
            out[col] = ""

    out["feature_name"] = out["feature_name"].astype(str)
    out["feature_original"] = out["feature_original"].where(out["feature_original"].astype(str).str.len() > 0, out["feature_name"])
    out["feature_group"] = out["feature_group"].where(out["feature_group"].astype(str).str.len() > 0, out["feature_name"].map(infer_feature_group))
    out["biological_theme"] = out["biological_theme"].fillna("").replace("", "other interpretable spatial signal")
    out["feature_label"] = out["feature_name"].map(humanize_feature)
    out["strict_biology_feature"] = True

    identity_pattern = r"(?:^drug__|treatment_identity|treatment|drug_key|prior_prob|treatment_prior)"
    out["treatment_identity_like"] = out["feature_name"].str.lower().str.contains(identity_pattern, regex=True, na=False)

    keep = [
        "feature_name",
        "feature_label",
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
        "strict_biology_feature",
        "treatment_identity_like",
    ]

    extra_cols = [c for c in out.columns if c not in keep]
    return out[keep + extra_cols].drop_duplicates("feature_name").reset_index(drop=True)


def add_feature_recurrence(feature_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Attach recurrence and evidence summaries to feature metadata.
    Merges optional V2 evidence sources when they are present."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    out = feature_df.copy()

    for source_id, prefix, preferred_cols in [
        (
            "integrated_recurrent_spatial_feature_table",
            "integrated",
            [
                "feature_name",
                "model_branch_count",
                "total_branch_score",
                "first_detected_branch",
                "step05_registry_score",
                "step06_broad_score",
                "step07_per_treatment_score",
                "step08_curated_score",
                "step09_validated_score",
            ],
        ),
        (
            "label_shuffle_validated_recurrent_spatial_features",
            "label_shuffle_validated",
            [
                "feature_name",
                "validated_treatment_count",
                "observed_selection_count",
                "mean_gain_importance",
                "max_gain_importance",
                "example_validated_treatments",
            ],
        ),
        (
            "curated_recurrent_spatial_features",
            "curated",
            [
                "feature_name",
                "treatment_count",
                "tier1_treatment_count",
                "tier2_treatment_count",
                "total_score",
                "mean_score",
                "max_score",
                "example_treatments",
            ],
        ),
        (
            "broad_residual_feature_evidence_summary",
            "broad",
            [
                "feature_name",
                "target_count",
                "selection_count",
                "mean_gain_importance",
                "mean_abs_shap",
                "max_abs_shap",
            ],
        ),
    ]:
        try:
            df = read_source_table(index_df, source_id, required=False)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            continue

        feat_col = choose_col(df.columns, ["feature_name", "feature"], required=False)
        if feat_col and feat_col != "feature_name":
            df = df.rename(columns={feat_col: "feature_name"})

        keep = [c for c in preferred_cols if c in df.columns]
        if "feature_name" not in keep:
            continue

        tmp = df[keep].drop_duplicates("feature_name").copy()
        rename = {c: f"{prefix}_{c}" for c in keep if c != "feature_name"}
        tmp = tmp.rename(columns=rename)
        out = out.merge(tmp, on="feature_name", how="left")

    return out


def build_feature_dictionaries(feature_dict: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature-group and biology-theme dictionaries.
    Summarizes strict spatial features for downstream reporting."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    group_rows: List[dict] = []
    for group, sub in feature_dict.groupby("feature_group", dropna=False):
        themes = sub["biological_theme"].astype(str)
        group_rows.append({
            "feature_group": group,
            "n_features": int(sub["feature_name"].nunique()),
            "n_biology_themes": int(themes.nunique()),
            "dominant_biology_theme": themes.value_counts().index[0] if len(themes) else "",
            "example_features": summarize_examples(sub["feature_name"], 8),
        })
    group_df = pd.DataFrame(group_rows).sort_values(["n_features", "feature_group"], ascending=[False, True])

    theme_rows: List[dict] = []
    for theme, sub in feature_dict.groupby("biological_theme", dropna=False):
        feature_groups = sub["feature_group"].astype(str)
        row = {
            "biological_theme": theme,
            "n_features": int(sub["feature_name"].nunique()),
            "n_feature_groups": int(feature_groups.nunique()),
            "dominant_feature_group": feature_groups.value_counts().index[0] if len(feature_groups) else "",
            "example_features": summarize_examples(sub["feature_name"], 10),
        }
        for col in [
            "integrated_model_branch_count",
            "integrated_total_branch_score",
            "label_shuffle_validated_validated_treatment_count",
            "curated_treatment_count",
            "broad_target_count",
        ]:
            if col in sub.columns:
                row[f"{col}_sum"] = pd.to_numeric(sub[col], errors="coerce").sum()
        theme_rows.append(row)
    theme_df = pd.DataFrame(theme_rows).sort_values(["n_features", "biological_theme"], ascending=[False, True])

    return group_df, theme_df


def build_treatment_tables(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build treatment, component, and component-by-treatment tables.
    Combines V2 eligibility, curation, validation, and component metadata."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    source_ids = [
        "v2_treatment_eligibility",
        "curated_treatment_model_table",
        "integrated_validated_treatment_table",
        "tier1_label_shuffle_validated_treatments",
        "final_model_manifest",
    ]

    tables = {}
    all_drugs = set()
    for source_id in source_ids:
        try:
            df = read_source_table(index_df, source_id, required=False)
        except Exception:
            df = pd.DataFrame()
        tables[source_id] = df
        if not df.empty:
            drug_col = choose_col(df.columns, ["drug_key", "treatment_key", "drug", "treatment"], required=False)
            if drug_col:
                all_drugs.update(df[drug_col].dropna().astype(str).tolist())

    treatments = pd.DataFrame({"drug_key": sorted(all_drugs)})

    if treatments.empty:
        return treatments, pd.DataFrame(), pd.DataFrame()

    treatments["treatment_label"] = treatments["drug_key"]
    treatments["treatment_components"] = treatments["drug_key"].map(lambda x: "; ".join(parse_treatment_components(x)))
    treatments["n_treatment_components"] = treatments["drug_key"].map(lambda x: len(parse_treatment_components(x)))
    treatments["component_classes"] = treatments["drug_key"].map(
        lambda x: "; ".join(sorted(set(classify_component(c) for c in parse_treatment_components(x))))
    )

    curated = tables.get("curated_treatment_model_table", pd.DataFrame())
    if not curated.empty and "drug_key" in curated.columns:
        keep = [c for c in [
            "drug_key",
            "interpretation_tier",
            "ready_for_label_shuffle_validation",
            "uses_step05_v2_registry",
            "n_samples",
            "n_rows",
            "target_mean",
            "target_std",
            "test_pearson_mean",
            "test_r2_mean",
            "rmse_improvement_vs_baseline_mean",
            "interpretation_tier_reason",
        ] if c in curated.columns]
        treatments = treatments.merge(curated[keep].drop_duplicates("drug_key"), on="drug_key", how="left")

    validated = tables.get("integrated_validated_treatment_table", pd.DataFrame())
    if not validated.empty and "drug_key" in validated.columns:
        keep = [c for c in [
            "drug_key",
            "observed_test_pearson_mean",
            "observed_test_r2_mean",
            "observed_rmse_improvement_vs_baseline_mean",
            "empirical_p_pearson",
            "fdr_q_pearson",
            "n_null_shuffles",
            "n_samples_total",
            "label_shuffle_validation_status",
            "integrated_interpretation_status",
        ] if c in validated.columns]
        tmp = validated[keep].drop_duplicates("drug_key").copy()
        tmp["label_shuffle_validated"] = True
        treatments = treatments.merge(tmp, on="drug_key", how="left")
    else:
        treatments["label_shuffle_validated"] = False

    if "label_shuffle_validated" not in treatments.columns:
        treatments["label_shuffle_validated"] = False
    treatments["label_shuffle_validated"] = treatments["label_shuffle_validated"].fillna(False).astype(bool)

    component_rows: List[dict] = []
    component_by_treatment_rows: List[dict] = []

    for _, row in treatments.iterrows():
        drug_key = str(row["drug_key"])
        components = parse_treatment_components(drug_key)
        is_validated = bool(row.get("label_shuffle_validated", False))
        tier = row.get("interpretation_tier", "")
        for i, component in enumerate(components, start=1):
            component_class = classify_component(component)
            component_by_treatment_rows.append({
                "drug_key": drug_key,
                "component_index": i,
                "component_name": component,
                "component_class": component_class,
                "label_shuffle_validated_treatment": is_validated,
                "interpretation_tier": tier,
            })
            component_rows.append({
                "component_name": component,
                "component_class": component_class,
                "drug_key": drug_key,
                "label_shuffle_validated_treatment": is_validated,
            })

    component_by_treatment = pd.DataFrame(component_by_treatment_rows)
    if component_rows:
        comp = pd.DataFrame(component_rows)
        component_dictionary = (
            comp.groupby(["component_name", "component_class"], as_index=False)
            .agg(
                treatment_count=("drug_key", "nunique"),
                validated_treatment_count=("label_shuffle_validated_treatment", "sum"),
                example_treatments=("drug_key", lambda s: summarize_examples(s, 5)),
            )
            .sort_values(["validated_treatment_count", "treatment_count", "component_name"], ascending=[False, False, True])
        )
    else:
        component_dictionary = pd.DataFrame()

    return treatments, component_dictionary, component_by_treatment


def build_source_column_contract(prepared_root: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    """Build a source-column contract from Step 01 schemas.
    Documents the input columns available to later interpretation steps."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows: List[dict] = []
    schema_root = prepared_root / "03_source_table_schemas"

    if schema_root.exists():
        for path in sorted(schema_root.glob("*_columns.tsv")):
            try:
                df = read_table(path)
            except Exception:
                continue
            source_id = path.name.replace("_columns.tsv", "")
            for _, row in df.iterrows():
                rows.append({
                    "source_id_from_schema_file": source_id,
                    "schema_file": str(path),
                    "column_index": row.get("column_index", ""),
                    "column_name": row.get("column_name", ""),
                })

    if not rows:
        for _, row in index_df.iterrows():
            sid = str(row["source_id"])
            path = Path(str(row.get("preferred_read_path") or row.get("source_path") or ""))
            if path.exists() and path.suffix.lower() in [".tsv", ".tab", ".csv"]:
                try:
                    cols = read_header(path)
                except Exception:
                    continue
                for i, col in enumerate(cols):
                    rows.append({
                        "source_id_from_schema_file": sid,
                        "schema_file": "",
                        "column_index": i,
                        "column_name": col,
                    })

    return pd.DataFrame(rows)


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

    step_root = output_root / "02_feature_and_treatment_dictionary"
    feature_dir = step_root / "01_feature_dictionary"
    treatment_dir = step_root / "02_treatment_dictionary"
    contract_dir = step_root / "03_contracts"
    report_dir = step_root / "04_reports"

    for path in [feature_dir, treatment_dir, contract_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        strict = read_source_table(index_df, "v2_strict_biology_feature_registry")
        feature_dict = normalize_feature_table(strict)

        try:
            spatial_path = source_path(index_df, "v2_spatial_features_broad_pool", required=False)
            spatial_cols = set(read_header(spatial_path)) if spatial_path else set()
        except Exception:
            spatial_cols = set()
            warnings.append("Could not read broad spatial feature table header.")

        feature_dict["present_in_v2_spatial_features_broad_pool"] = feature_dict["feature_name"].isin(spatial_cols)
        feature_dict = add_feature_recurrence(feature_dict, index_df)

        group_dict, theme_dict = build_feature_dictionaries(feature_dict)
        treatment_dict, component_dict, component_by_treatment = build_treatment_tables(index_df)
        source_contract = build_source_column_contract(prepared_root, index_df)

        write_tsv(feature_dir / "strict_spatial_feature_dictionary.tsv", feature_dict)
        write_tsv(feature_dir / "feature_group_dictionary.tsv", group_dict)
        write_tsv(feature_dir / "biology_theme_dictionary.tsv", theme_dict)
        write_tsv(treatment_dir / "treatment_dictionary.tsv", treatment_dict)
        write_tsv(treatment_dir / "treatment_component_dictionary.tsv", component_dict)
        write_tsv(treatment_dir / "treatment_component_by_treatment.tsv", component_by_treatment)
        write_tsv(contract_dir / "source_column_contract.tsv", source_contract)

        feature_count = int(feature_dict["feature_name"].nunique())
        theme_count = int(feature_dict["biological_theme"].nunique())
        treatment_count = int(treatment_dict["drug_key"].nunique()) if "drug_key" in treatment_dict.columns else 0
        validated_count = int(treatment_dict["label_shuffle_validated"].sum()) if "label_shuffle_validated" in treatment_dict.columns else 0
        identity_hits = int(feature_dict["treatment_identity_like"].sum()) if "treatment_identity_like" in feature_dict.columns else 0
        missing_spatial = int((~feature_dict["present_in_v2_spatial_features_broad_pool"]).sum()) if "present_in_v2_spatial_features_broad_pool" in feature_dict.columns else 0

        add_qc(qc, "strict_feature_dictionary_rows", "pass" if feature_count == 139 else "warn", feature_count, 139, "Strict biology dictionary should carry the Step 05 registry.")
        add_qc(qc, "biology_theme_count", "pass" if theme_count == 11 else "warn", theme_count, 11, "Expected recurrent V2 biology themes.")
        add_qc(qc, "validated_treatment_count", "pass" if validated_count == 27 else "warn", validated_count, 27, "Expected label-shuffle-validated treatments.")
        add_qc(qc, "treatment_identity_features_absent", "pass" if identity_hits == 0 else "fail", identity_hits, 0, "Strict dictionary should not contain treatment identity features.")
        add_qc(qc, "strict_features_present_in_spatial_table", "pass" if missing_spatial == 0 else "warn", missing_spatial, 0, "Strict features should be readable from the V2 spatial feature table for signed interpretation.")
        add_qc(qc, "source_column_contract_rows", "pass" if len(source_contract) > 0 else "warn", len(source_contract), ">0", "Source schemas should be recorded for reproducibility.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(contract_dir / "step02_dictionary_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "feature_dictionary": str(feature_dir / "strict_spatial_feature_dictionary.tsv"),
        "theme_dictionary": str(feature_dir / "biology_theme_dictionary.tsv"),
        "treatment_dictionary": str(treatment_dir / "treatment_dictionary.tsv"),
        "component_dictionary": str(treatment_dir / "treatment_component_dictionary.tsv"),
        "source_column_contract": str(contract_dir / "source_column_contract.tsv"),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step02_summary.json", summary)

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 02 REPORT",
        "",
        f"status: {status}",
        f"prepared_input_root: {prepared_root}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        f"feature_dictionary: {feature_dir / 'strict_spatial_feature_dictionary.tsv'}",
        f"feature_group_dictionary: {feature_dir / 'feature_group_dictionary.tsv'}",
        f"biology_theme_dictionary: {feature_dir / 'biology_theme_dictionary.tsv'}",
        f"treatment_dictionary: {treatment_dir / 'treatment_dictionary.tsv'}",
        f"treatment_component_dictionary: {treatment_dir / 'treatment_component_dictionary.tsv'}",
        f"treatment_component_by_treatment: {treatment_dir / 'treatment_component_by_treatment.tsv'}",
        f"source_column_contract: {contract_dir / 'source_column_contract.tsv'}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Policy",
        "This step consumed Step 01 prepared V2 sources.",
        "This step did not rerun V2 and did not perform model selection.",
        "Treatment component classes are descriptive labels for reporting, not clinical recommendations.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step02_feature_and_treatment_dictionary_report.txt", "\n".join(report_lines))

    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 02 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    if not errors:
        print(f"strict_features: {feature_count}")
        print(f"biology_themes: {theme_count}")
        print(f"treatments_in_dictionary: {treatment_count}")
        print(f"validated_treatments: {validated_count}")
        print(f"component_rows: {len(component_dict)}")
    print(f"report: {report_dir / 'step02_feature_and_treatment_dictionary_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())
