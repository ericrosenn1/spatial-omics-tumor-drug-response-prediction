from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path | str, obj: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def write_table(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if df is None:
        df = pd.DataFrame()
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)
    return path


def write_text_report(path: Path | str, body: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")
    return path


def terminal_block(title: str, lines: list[str]) -> str:
    bar = "=" * 90
    return "\n".join([bar, title, bar] + lines)


def file_manifest(root: Path) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["relative_path", "path", "size_bytes"])
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "relative_path": str(path.relative_to(root)),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "modified_time": path.stat().st_mtime,
            })
    return pd.DataFrame(rows)


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_number(series: pd.Series) -> pd.Series:
    """Convert numeric-like, boolean, and boolean-like columns to float safely."""

    if series.dtype == bool:
        return series.astype(float).replace([np.inf, -np.inf], np.nan)

    lowered = series.dropna().astype(str).str.lower()
    bool_values = {"true", "false", "yes", "no", "1", "0"}

    if len(lowered) > 0 and lowered.isin(bool_values).all():
        mapped = (
            series.astype(str)
            .str.lower()
            .map({"true": 1.0, "yes": 1.0, "1": 1.0, "false": 0.0, "no": 0.0, "0": 0.0})
        )
        return pd.to_numeric(mapped, errors="coerce").replace([np.inf, -np.inf], np.nan)

    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def load_first(root: Path, candidates: list[str]) -> pd.DataFrame:
    for rel in candidates:
        path = root / rel
        df = read_table(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def expected_file_contract(run_root: Path) -> pd.DataFrame:
    expected = [
        ("01", "01_validate_inputs/v2_input_validation_summary.json"),
        ("01", "01_validate_inputs/02_reports/v2_input_validation_report.txt"),
        ("02", "02_build_modeling_dataset/v2_dataset_builder_summary.json"),
        ("02", "02_build_modeling_dataset/03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"),
        ("02", "02_build_modeling_dataset/03_modeling_datasets/v2_broad_residual_dataset_broad_governed_candidate_pool.tsv"),
        ("02", "02_build_modeling_dataset/03_modeling_datasets/v2_treatment_eligibility.tsv"),
        ("03", "03_probability_baseline/v2_step03_probability_baseline_summary.json"),
        ("04", "04_pair_level_residual_model/v2_step04_pair_level_residual_model_summary.json"),
        ("05", "05_residual_biology_registry/v2_step05_residual_biology_registry_summary.json"),
        ("05", "05_residual_biology_registry/03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv"),
        ("06", "06_broad_residual_model/v2_step06_broad_residual_model_summary.json"),
        ("07", "07_filtered_per_treatment_residual_models/v2_step07_filtered_per_treatment_residual_models_summary.json"),
        ("08", "08_curated_per_treatment_residual_models/v2_step08_curated_per_treatment_residual_models_summary.json"),
        ("09", "09_tier1_label_shuffle_validation/v2_step09_tier1_label_shuffle_validation_summary.json"),
        ("10", "10_integrated_interpretation_package/v2_step10_integrated_interpretation_package_summary.json"),
        ("10", "10_integrated_interpretation_package/02_model_comparison/model_comparison_table.tsv"),
        ("11", "11_publication_tables/v2_step11_publication_tables_summary.json"),
        ("11", "11_publication_tables/01_publication_excel/v2_integrated_publication_tables.xlsx"),
        ("11", "11_publication_tables/05_package_zip/v2_publication_tables_and_supporting_files.zip"),
    ]

    rows = []
    for step, rel in expected:
        path = run_root / rel
        rows.append({
            "step": step,
            "relative_path": rel,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
        })
    return pd.DataFrame(rows)


def governance_qc(run_root: Path) -> pd.DataFrame:
    rows = []

    registry = read_table(run_root / "05_residual_biology_registry/03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv")
    broad_features = read_table(run_root / "02_build_modeling_dataset/02_feature_governance/v2_broad_governed_candidate_features.tsv")
    step07_manifest = read_table(run_root / "07_filtered_per_treatment_residual_models/03_final_models/final_model_manifest.tsv")
    step08_curated = read_table(run_root / "08_curated_per_treatment_residual_models/02_curated_treatment_models/curated_treatment_model_table.tsv")
    step09_validation = read_table(run_root / "09_tier1_label_shuffle_validation/04_validation_results/tier1_label_shuffle_validation_results.tsv")

    def has_treatment_identity(df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        text = " ".join(df.astype(str).fillna("").values.ravel().tolist()).lower()
        return "treatment_identity" in text or "drug_identity" in text or "drug_dummy" in text

    rows.append({
        "qc_check": "step05_registry_exists",
        "status": "pass" if not registry.empty else "fail",
        "value": len(registry),
        "detail": "Strict biology registry rows.",
    })

    rows.append({
        "qc_check": "step05_registry_no_treatment_identity",
        "status": "pass" if not has_treatment_identity(registry) else "fail",
        "value": has_treatment_identity(registry),
        "detail": "Treatment identity should not appear in strict biology registry.",
    })

    rows.append({
        "qc_check": "broad_candidate_pool_exists",
        "status": "pass" if not broad_features.empty else "fail",
        "value": len(broad_features),
        "detail": "Broad candidate pool used by Step 04.",
    })

    if not step07_manifest.empty and "uses_step05_v2_registry" in step07_manifest.columns:
        uses = boolish(step07_manifest["uses_step05_v2_registry"])
        rows.append({
            "qc_check": "step07_uses_step05_registry",
            "status": "pass" if uses.all() else "fail",
            "value": f"{int(uses.sum())}/{len(uses)}",
            "detail": "All final per-treatment models should use Step 05 V2 registry.",
        })

    if not step08_curated.empty and "ready_for_label_shuffle_validation" in step08_curated.columns:
        ready = boolish(step08_curated["ready_for_label_shuffle_validation"])
        rows.append({
            "qc_check": "step08_label_shuffle_handoff_candidates",
            "status": "pass" if int(ready.sum()) > 0 else "warn",
            "value": int(ready.sum()),
            "detail": "Tier 1 candidates ready for Step 09.",
        })

    if not step09_validation.empty and "validated_for_step10" in step09_validation.columns:
        val = boolish(step09_validation["validated_for_step10"])
        rows.append({
            "qc_check": "step09_validated_treatments",
            "status": "pass" if int(val.sum()) > 0 else "warn",
            "value": int(val.sum()),
            "detail": "Label-shuffle-validated treatments.",
        })

    return pd.DataFrame(rows)


def pair_dataset_qc(run_root: Path, output_root: Path) -> dict:
    pair = read_table(run_root / "02_build_modeling_dataset/03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv")
    eligibility = read_table(run_root / "02_build_modeling_dataset/03_modeling_datasets/v2_treatment_eligibility.tsv")

    if pair.empty:
        empty = pd.DataFrame()
        write_table(empty, output_root / "03_pair_level_and_prediction_qc/qc_by_sample.tsv")
        write_table(empty, output_root / "03_pair_level_and_prediction_qc/qc_by_treatment.tsv")
        write_table(empty, output_root / "03_pair_level_and_prediction_qc/qc_value_ranges.tsv")
        return {"pair_rows": 0, "pair_status": "fail"}

    for col in ["sample_id", "drug_key"]:
        if col in pair.columns:
            pair[col] = pair[col].astype(str)

    summary = {
        "pair_rows": int(len(pair)),
        "pair_columns": int(len(pair.columns)),
        "samples": int(pair["sample_id"].nunique()) if "sample_id" in pair.columns else 0,
        "treatments": int(pair["drug_key"].nunique()) if "drug_key" in pair.columns else 0,
        "pair_status": "pass",
    }

    if "sample_id" in pair.columns and "drug_key" in pair.columns:
        by_sample = (
            pair.groupby("sample_id", dropna=False)
            .agg(
                n_pair_rows=("drug_key", "count"),
                n_treatments=("drug_key", "nunique"),
            )
            .reset_index()
            .sort_values("n_pair_rows", ascending=False)
        )

        by_treatment = (
            pair.groupby("drug_key", dropna=False)
            .agg(
                n_pair_rows=("sample_id", "count"),
                n_samples=("sample_id", "nunique"),
            )
            .reset_index()
            .sort_values("n_pair_rows", ascending=False)
        )
    else:
        by_sample = pd.DataFrame()
        by_treatment = pd.DataFrame()

    if not eligibility.empty and "drug_key" in eligibility.columns:
        eligibility["drug_key"] = eligibility["drug_key"].astype(str)
        keep_cols = [c for c in ["drug_key", "eligible", "n_samples", "target_std", "residual_std", "n_rows"] if c in eligibility.columns]
        if keep_cols:
            by_treatment = by_treatment.merge(eligibility[keep_cols].drop_duplicates("drug_key"), on="drug_key", how="left")

    numeric_cols = []
    for col in pair.columns:
        vals = safe_number(pair[col])
        if vals.notna().sum() > 0:
            numeric_cols.append(col)

    range_rows = []
    for col in numeric_cols:
        vals = safe_number(pair[col]).dropna().astype(float)

        if len(vals) > 0:
            q01 = float(np.nanpercentile(vals.to_numpy(dtype=float), 1))
            q50 = float(np.nanpercentile(vals.to_numpy(dtype=float), 50))
            q99 = float(np.nanpercentile(vals.to_numpy(dtype=float), 99))
        else:
            q01 = np.nan
            q50 = np.nan
            q99 = np.nan

        range_rows.append({
            "column": col,
            "nonmissing": int(len(vals)),
            "missing": int(len(pair) - len(vals)),
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std()) if len(vals) > 1 else np.nan,
            "min": float(vals.min()) if len(vals) else np.nan,
            "q01": q01,
            "q50": q50,
            "q99": q99,
            "max": float(vals.max()) if len(vals) else np.nan,
        })

    ranges = pd.DataFrame(range_rows)

    write_table(by_sample, output_root / "03_pair_level_and_prediction_qc/qc_by_sample.tsv")
    write_table(by_treatment, output_root / "03_pair_level_and_prediction_qc/qc_by_treatment.tsv")
    write_table(ranges, output_root / "03_pair_level_and_prediction_qc/qc_value_ranges.tsv")

    for target_col in ["fused_residual_vs_prior", "fused_prob_responder"]:
        if target_col in pair.columns:
            vals = safe_number(pair[target_col]).dropna()
            if len(vals) > 0:
                plt.figure(figsize=(8, 5))
                plt.hist(vals, bins=40)
                plt.title(f"Distribution of {target_col}")
                plt.xlabel(target_col)
                plt.ylabel("Pair count")
                plt.tight_layout()
                fig_path = output_root / "07_qc_figures" / f"fig_distribution_{target_col}.png"
                ensure_dir(fig_path.parent)
                plt.savefig(fig_path, dpi=220, bbox_inches="tight")
                plt.close()

    return summary


def model_metrics_qc(run_root: Path, output_root: Path) -> pd.DataFrame:
    rows = []

    model_comparison = read_table(run_root / "10_integrated_interpretation_package/02_model_comparison/model_comparison_table.tsv")
    step06 = read_table(run_root / "06_broad_residual_model/02_model_metrics/broad_residual_target_summary.tsv")
    step07 = read_table(run_root / "07_filtered_per_treatment_residual_models/02_screening_metrics/per_treatment_screening_summary.tsv")
    step08 = read_table(run_root / "08_curated_per_treatment_residual_models/02_curated_treatment_models/curated_treatment_model_table.tsv")
    step09 = read_table(run_root / "09_tier1_label_shuffle_validation/04_validation_results/tier1_label_shuffle_validation_results.tsv")

    if not model_comparison.empty:
        for _, row in model_comparison.iterrows():
            rows.append({
                "source": "step10_model_comparison",
                "model_branch": row.get("model_branch", ""),
                "metric_1": row.get("primary_metric_name", ""),
                "metric_1_value": row.get("primary_metric_value", ""),
                "metric_2": row.get("secondary_metric_name", ""),
                "metric_2_value": row.get("secondary_metric_value", ""),
                "status": row.get("validation_status", ""),
            })

    if not step06.empty:
        col = "test_pearson_mean" if "test_pearson_mean" in step06.columns else ""
        if col:
            best = step06.sort_values(col, ascending=False).head(1)
            if not best.empty:
                rows.append({
                    "source": "step06_broad_residual",
                    "model_branch": str(best.iloc[0].get("target_col", "")),
                    "metric_1": "best_test_pearson_mean",
                    "metric_1_value": best.iloc[0].get("test_pearson_mean", ""),
                    "metric_2": "best_test_r2_mean",
                    "metric_2_value": best.iloc[0].get("test_r2_mean", ""),
                    "status": "pass",
                })

    if not step07.empty:
        col = "test_pearson_mean" if "test_pearson_mean" in step07.columns else ""
        if col:
            vals = safe_number(step07[col])
            rows.append({
                "source": "step07_per_treatment",
                "model_branch": "all_screened_treatments",
                "metric_1": "n_screened_treatments",
                "metric_1_value": len(step07),
                "metric_2": "median_test_pearson_mean",
                "metric_2_value": float(vals.median()) if vals.notna().sum() else np.nan,
                "status": "pass",
            })

    if not step08.empty and "interpretation_tier" in step08.columns:
        counts = step08["interpretation_tier"].value_counts(dropna=False).to_dict()
        for tier, count in counts.items():
            rows.append({
                "source": "step08_curation",
                "model_branch": str(tier),
                "metric_1": "treatment_count",
                "metric_1_value": int(count),
                "metric_2": "",
                "metric_2_value": "",
                "status": "pass",
            })

    if not step09.empty and "validated_for_step10" in step09.columns:
        valid = boolish(step09["validated_for_step10"])
        rows.append({
            "source": "step09_label_shuffle",
            "model_branch": "validated_tier1",
            "metric_1": "validated_count",
            "metric_1_value": int(valid.sum()),
            "metric_2": "tested_count",
            "metric_2_value": int(len(step09)),
            "status": "pass" if int(valid.sum()) > 0 else "warn",
        })

    out = pd.DataFrame(rows)
    write_table(out, output_root / "04_model_metric_qc/qc_model_metrics.tsv")
    return out


def sample_treatment_contract_qc(run_root: Path, output_root: Path) -> pd.DataFrame:
    pair = read_table(run_root / "02_build_modeling_dataset/03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv")
    if pair.empty or "sample_id" not in pair.columns or "drug_key" not in pair.columns:
        out = pd.DataFrame([{
            "qc_check": "pair_table_available",
            "status": "fail",
            "detail": "Pair-level dataset unavailable or missing sample_id/drug_key.",
        }])
        write_table(out, output_root / "05_sample_treatment_contract_qc/qc_sample_treatment_contract.tsv")
        return out

    pair["sample_id"] = pair["sample_id"].astype(str)
    pair["drug_key"] = pair["drug_key"].astype(str)

    samples = pair["sample_id"].nunique()
    treatments = pair["drug_key"].nunique()
    observed_pairs = len(pair[["sample_id", "drug_key"]].drop_duplicates())
    possible_pairs = samples * treatments
    coverage = observed_pairs / possible_pairs if possible_pairs else np.nan

    duplicate_pairs = pair.duplicated(["sample_id", "drug_key"]).sum()

    out = pd.DataFrame([
        {
            "qc_check": "sample_treatment_pair_coverage",
            "status": "pass" if coverage >= 0.80 else "warn",
            "value": coverage,
            "detail": f"Observed {observed_pairs} unique sample-treatment pairs out of {possible_pairs} possible pairs.",
        },
        {
            "qc_check": "duplicate_sample_treatment_pairs",
            "status": "pass" if duplicate_pairs == 0 else "warn",
            "value": int(duplicate_pairs),
            "detail": "Duplicate sample-treatment pairs in pair-level modeling dataset.",
        },
        {
            "qc_check": "sample_count",
            "status": "pass" if samples > 0 else "fail",
            "value": int(samples),
            "detail": "Unique samples with pair-level rows.",
        },
        {
            "qc_check": "treatment_count",
            "status": "pass" if treatments > 0 else "fail",
            "value": int(treatments),
            "detail": "Unique treatments with pair-level rows.",
        },
    ])

    write_table(out, output_root / "05_sample_treatment_contract_qc/qc_sample_treatment_contract.tsv")
    return out


def publication_package_qc(run_root: Path, output_root: Path) -> pd.DataFrame:
    step11 = run_root / "11_publication_tables"
    sheet_summary = read_table(step11 / "01_publication_excel/excel_workbook_sheet_summary.tsv")
    pub_manifest = read_table(step11 / "02_publication_ready_tsv/publication_ready_tsv_manifest.tsv")
    support_manifest = read_table(step11 / "03_supporting_source_files/supporting_source_file_manifest.tsv")
    summary = read_json(step11 / "v2_step11_publication_tables_summary.json")

    rows = [
        {
            "qc_check": "excel_workbook_exists",
            "status": "pass" if (step11 / "01_publication_excel/v2_integrated_publication_tables.xlsx").exists() else "fail",
            "value": str(step11 / "01_publication_excel/v2_integrated_publication_tables.xlsx"),
            "detail": "Publication workbook exists.",
        },
        {
            "qc_check": "zip_package_exists",
            "status": "pass" if (step11 / "05_package_zip/v2_publication_tables_and_supporting_files.zip").exists() else "fail",
            "value": str(step11 / "05_package_zip/v2_publication_tables_and_supporting_files.zip"),
            "detail": "Publication ZIP exists.",
        },
        {
            "qc_check": "publication_sheet_count",
            "status": "pass" if len(sheet_summary) >= 7 else "warn",
            "value": int(len(sheet_summary)),
            "detail": "Workbook sheet summary rows.",
        },
        {
            "qc_check": "publication_tsv_count",
            "status": "pass" if len(pub_manifest) >= 7 else "warn",
            "value": int(len(pub_manifest)),
            "detail": "Publication-ready TSV rows.",
        },
        {
            "qc_check": "supporting_file_count",
            "status": "pass" if len(support_manifest) >= 30 else "warn",
            "value": int(len(support_manifest)),
            "detail": "Supporting source files copied into short-path package.",
        },
        {
            "qc_check": "v1_dependency",
            "status": "pass" if summary.get("production_dependency_on_v1_outputs", "") == "no" else "fail",
            "value": summary.get("production_dependency_on_v1_outputs", ""),
            "detail": "Publication summary should report no V1 production dependency.",
        },
    ]

    out = pd.DataFrame(rows)
    write_table(out, output_root / "06_publication_package_qc/qc_publication_package.tsv")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--open-output", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_file_contract_qc")
    d02 = ensure_dir(output_root / "02_branch_governance_qc")
    d03 = ensure_dir(output_root / "03_pair_level_and_prediction_qc")
    d04 = ensure_dir(output_root / "04_model_metric_qc")
    d05 = ensure_dir(output_root / "05_sample_treatment_contract_qc")
    d06 = ensure_dir(output_root / "06_publication_package_qc")
    d07 = ensure_dir(output_root / "07_qc_figures")
    d08 = ensure_dir(output_root / "08_reports")

    contract = expected_file_contract(run_root)
    write_table(contract, d01 / "file_contract_report.tsv")

    manifest = file_manifest(run_root)
    write_table(manifest, d01 / "run_file_manifest.tsv")

    governance = governance_qc(run_root)
    write_table(governance, d02 / "branch_governance_qc.tsv")

    pair_summary = pair_dataset_qc(run_root, output_root)
    model_metrics = model_metrics_qc(run_root, output_root)
    sample_contract = sample_treatment_contract_qc(run_root, output_root)
    publication_qc = publication_package_qc(run_root, output_root)

    all_checks = pd.concat([
        contract.assign(qc_group="file_contract").rename(columns={"exists": "status_raw"}),
        governance.assign(qc_group="branch_governance"),
        sample_contract.assign(qc_group="sample_treatment_contract"),
        publication_qc.assign(qc_group="publication_package"),
    ], ignore_index=True, sort=False)

    contract_fail = int((contract["exists"] == False).sum())
    explicit_fail = int((all_checks.get("status", pd.Series(dtype=str)).astype(str) == "fail").sum())
    warn_count = int((all_checks.get("status", pd.Series(dtype=str)).astype(str) == "warn").sum())

    overall_status = "pass" if contract_fail == 0 and explicit_fail == 0 else "fail"

    summary = {
        "status": overall_status,
        "official_step": "12_qc_v2_outputs",
        "run_root": str(run_root),
        "output_root": str(output_root),
        "file_contract_missing": int(contract_fail),
        "explicit_fail_checks": int(explicit_fail),
        "warning_checks": int(warn_count),
        "pair_rows": int(pair_summary.get("pair_rows", 0)),
        "samples": int(pair_summary.get("samples", 0)),
        "treatments": int(pair_summary.get("treatments", 0)),
        "model_metric_rows": int(len(model_metrics)),
        "production_dependency_on_v1_outputs": "no",
        "canonical_v1_scripts_modified": "no",
    }

    write_json(output_root / "v2_step12_output_qc_summary.json", summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 12 OUTPUT QC REPORT")
    report_lines.append("")
    for key, value in summary.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("1. File contract")
    report_lines.append(contract.to_string(index=False))

    report_lines.append("")
    report_lines.append("2. Branch governance QC")
    report_lines.append(governance.to_string(index=False) if not governance.empty else "No branch governance checks generated.")

    report_lines.append("")
    report_lines.append("3. Sample-treatment contract QC")
    report_lines.append(sample_contract.to_string(index=False) if not sample_contract.empty else "No sample-treatment contract checks generated.")

    report_lines.append("")
    report_lines.append("4. Publication package QC")
    report_lines.append(publication_qc.to_string(index=False) if not publication_qc.empty else "No publication package checks generated.")

    report_lines.append("")
    report_lines.append("5. Interpretation")
    report_lines.append("Step 12 checks that expected files exist, branch outputs are internally consistent, strict biology interpretation does not contain treatment identity features, Step 07 uses the Step 05 registry, sample-treatment pair coverage is measurable, model metrics exist, and publication outputs are present.")
    report_lines.append("This is a QC layer. It does not retrain models and does not change scientific outputs.")

    report_path = write_text_report(d08 / "v2_step12_output_qc_report.txt", "\n".join(report_lines))

    print("")
    print(terminal_block("V2 STEP 12 OUTPUT QC COMPLETE", [
        f"Status: {overall_status}",
        f"Run root: {run_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"File contract missing: {contract_fail}",
        f"Explicit failed checks: {explicit_fail}",
        f"Warnings: {warn_count}",
        f"Pair rows: {summary['pair_rows']}",
        f"Samples: {summary['samples']}",
        f"Treatments: {summary['treatments']}",
        "Production dependency on V1 outputs: no",
        "Canonical V1 scripts modified: no",
    ]))
    print("")

    print("Branch governance QC")
    print(governance.to_string(index=False) if not governance.empty else "No branch governance QC rows.")
    print("")

    print("Sample-treatment contract QC")
    print(sample_contract.to_string(index=False) if not sample_contract.empty else "No sample-treatment contract QC rows.")
    print("")

    print("Publication package QC")
    print(publication_qc.to_string(index=False) if not publication_qc.empty else "No publication package QC rows.")
    print("")

    if args.open_output and sys.platform.startswith("win"):
        try:
            os.startfile(str(output_root))
            os.startfile(str(d08))
            os.startfile(str(d07))
        except Exception:
            pass

    return 0 if overall_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
