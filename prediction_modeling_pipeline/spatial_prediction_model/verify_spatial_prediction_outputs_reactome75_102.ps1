param(
    [string]$ProjectRoot = "" ,
    [string]$CurrentRunName = "output_run_102_reactome75_20260504",
    [string[]]$PastRunNames = @("output_run_102", "output_run_10")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($ProjectRoot -eq "") {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

$SpatialModelDir = Join-Path $ProjectRoot "prediction_modeling_pipeline\spatial_prediction_model"
$OutputBase = Join-Path $SpatialModelDir "outputs"
$CurrentRoot = Join-Path $OutputBase $CurrentRunName
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $SpatialModelDir "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$VerifierPy = Join-Path $LogDir "verify_spatial_prediction_outputs_$Stamp.py"
$ReportTxt = Join-Path $LogDir "verify_spatial_prediction_outputs_$Stamp.txt"
$StepChecksTsv = Join-Path $LogDir "verify_spatial_prediction_step_checks_$Stamp.tsv"
$CompareTsv = Join-Path $LogDir "verify_spatial_prediction_compare_runs_$Stamp.tsv"
$FlagsTsv = Join-Path $LogDir "verify_spatial_prediction_reasonableness_flags_$Stamp.tsv"

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

if (-not (Test-Path $CurrentRoot)) {
    throw "Current run output folder not found: $CurrentRoot"
}

$PastRunText = ($PastRunNames | ForEach-Object { '"' + $_ + '"' }) -join ", "

$Code = @"
from pathlib import Path
import json
import math
import re
import sys
import traceback

import numpy as np
import pandas as pd

project_root = Path(r"$ProjectRoot")
spatial_model_dir = Path(r"$SpatialModelDir")
output_base = Path(r"$OutputBase")
current_run_name = "$CurrentRunName"
current_root = Path(r"$CurrentRoot")
past_run_names = [$PastRunText]

report_txt = Path(r"$ReportTxt")
step_checks_tsv = Path(r"$StepChecksTsv")
compare_tsv = Path(r"$CompareTsv")
flags_tsv = Path(r"$FlagsTsv")

checks = []
flags = []
summary_lines = []

def add_check(step, check_name, status, observed="", expected="", detail=""):
    checks.append({
        "step": step,
        "check_name": check_name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })

def add_flag(level, category, message, detail=""):
    flags.append({
        "level": level,
        "category": category,
        "message": message,
        "detail": detail,
    })

def exists_file(path):
    return path.exists() and path.is_file() and path.stat().st_size > 0

def read_table(path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 2:
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)

def safe_read(path, step, name, required=True):
    try:
        if not path.exists():
            if required:
                add_check(step, f"{name} exists", "FAIL", "missing", str(path), "")
            else:
                add_check(step, f"{name} exists", "WARN", "missing optional", str(path), "")
            return pd.DataFrame()
        if path.is_file() and path.stat().st_size <= 2:
            if required:
                add_check(step, f"{name} nonempty", "FAIL", "empty", str(path), "")
            else:
                add_check(step, f"{name} nonempty", "WARN", "empty optional", str(path), "")
            return pd.DataFrame()
        df = read_table(path)
        add_check(step, f"{name} load", "PASS", f"{df.shape[0]} rows x {df.shape[1]} cols", str(path), "")
        return df
    except Exception as exc:
        add_check(step, f"{name} load", "FAIL", type(exc).__name__, str(path), str(exc))
        return pd.DataFrame()

def numeric_col(df, col):
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def unique_count(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    return int(df[col].nunique(dropna=True))

def expect_equal(step, check_name, observed, expected, detail=""):
    status = "PASS" if observed == expected else "FAIL"
    add_check(step, check_name, status, observed, expected, detail)

def expect_at_least(step, check_name, observed, threshold, detail=""):
    status = "PASS" if observed >= threshold else "FAIL"
    add_check(step, check_name, status, observed, f">= {threshold}", detail)

def expect_between(step, check_name, observed, low, high, detail=""):
    status = "PASS" if observed >= low and observed <= high else "FAIL"
    add_check(step, check_name, status, observed, f"{low} to {high}", detail)

def maybe_value(df, col, default=np.nan):
    if df.empty or col not in df.columns or len(df) == 0:
        return default
    return df.iloc[0][col]

def read_qc_summary(root):
    path = root / "07_prediction_qc" / "qc_summary.tsv"
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)

def summarize_run(root, name):
    out = {"run_name": name, "path": str(root), "exists": root.exists()}
    qc = read_qc_summary(root)
    if not qc.empty:
        row = qc.iloc[0].to_dict()
        for key in [
            "n_labeled_rows",
            "n_labeled_samples",
            "n_labeled_treatments",
            "n_x_features",
            "n_drug_dummy_features",
            "n_spatial_features_in_x",
            "n_manifest_rows",
            "n_prediction_rows",
            "n_prediction_samples",
            "n_prediction_treatments",
            "required_files_missing",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_pearson",
            "teacher_overlap_mae",
            "teacher_overlap_rmse",
            "teacher_overlap_pearson",
            "prediction_mean_prediction",
            "prediction_median_prediction",
            "prediction_min_prediction",
            "prediction_max_prediction",
            "per_treatment_n_trained",
            "per_treatment_n_skipped",
            "per_treatment_n_failed",
        ]:
            out[key] = row.get(key, np.nan)
        return out

    gm = root / "03_global_model" / "global_model_summary.txt"
    if gm.exists():
        text = gm.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"n_features:\s*(\d+)", text)
        if m:
            out["n_x_features"] = int(m.group(1))
        m = re.search(r"n_drug_dummy_features:\s*(\d+)", text)
        if m:
            out["n_drug_dummy_features"] = int(m.group(1))
        m = re.search(r"test:\s*n=(\d+);\s*mae=([0-9.]+);\s*rmse=([0-9.]+);\s*r2=([0-9.]+);\s*pearson=([0-9.]+)", text)
        if m:
            out["test_n"] = int(m.group(1))
            out["test_mae"] = float(m.group(2))
            out["test_rmse"] = float(m.group(3))
            out["test_r2"] = float(m.group(4))
            out["test_pearson"] = float(m.group(5))
    return out

summary_lines.append("Spatial prediction output verification")
summary_lines.append(f"Current run: {current_run_name}")
summary_lines.append(f"Current root: {current_root}")
summary_lines.append("")

# Step 01
d01 = current_root / "01_input_validation"
required_01 = [
    "input_validation_summary.txt",
    "input_table_shapes.tsv",
    "input_column_report.tsv",
    "required_column_check.tsv",
    "input_duplicate_report.tsv",
    "input_sample_overlap.tsv",
    "input_pair_overlap.tsv",
    "target_summary.tsv",
    "feature_manifest_check.tsv",
    "leakage_column_report.tsv",
    "available_labeled_samples.tsv",
    "validation_issues.tsv",
    "run_config.json",
]
for f in required_01:
    add_check("01", f"{f} exists", "PASS" if exists_file(d01 / f) else "FAIL", str(d01 / f), "present and nonempty", "")

shapes = safe_read(d01 / "input_table_shapes.tsv", "01", "input_table_shapes")
issues = safe_read(d01 / "validation_issues.tsv", "01", "validation_issues")
dupes = safe_read(d01 / "input_duplicate_report.tsv", "01", "input_duplicate_report")
manifest_check = safe_read(d01 / "feature_manifest_check.tsv", "01", "feature_manifest_check")
available_samples = safe_read(d01 / "available_labeled_samples.tsv", "01", "available_labeled_samples")

if not issues.empty and "severity" in issues.columns:
    n_errors = int(issues["severity"].astype(str).str.lower().eq("error").sum())
else:
    n_errors = 0
expect_equal("01", "validation errors", n_errors, 0)

if not dupes.empty and "n_duplicate_rows" in dupes.columns:
    total_dupes = int(pd.to_numeric(dupes["n_duplicate_rows"], errors="coerce").fillna(0).sum())
    expect_equal("01", "duplicate rows across core input keys", total_dupes, 0)

if not shapes.empty:
    def shape_row(table):
        if "table" not in shapes.columns:
            return pd.Series(dtype=object)
        rows = shapes[shapes["table"].astype(str).eq(table)]
        return rows.iloc[0] if len(rows) else pd.Series(dtype=object)

    r = shape_row("model_input_numeric")
    if len(r):
        expect_equal("01", "model_input_numeric rows", int(r["n_rows"]), 102)
        expect_equal("01", "model_input_numeric columns", int(r["n_columns"]), 662)
        expect_equal("01", "model_input_numeric sample count", int(float(r["n_sample_id"])), 102)

    r = shape_row("teacher_table")
    if len(r):
        expect_equal("01", "teacher_table rows", int(r["n_rows"]), 3909)
        expect_equal("01", "teacher_table treatments", int(float(r["n_drug_key"])), 40)

    r = shape_row("training_table")
    if len(r):
        expect_equal("01", "training_table rows", int(r["n_rows"]), 3909)
        expect_equal("01", "training_table treatments", int(float(r["n_drug_key"])), 40)

if not manifest_check.empty:
    expect_equal("01", "feature manifest rows checked", len(manifest_check), 661)
    if "in_model_input_numeric" in manifest_check.columns:
        missing_manifest = int((manifest_check["in_model_input_numeric"].astype(str).str.lower() != "true").sum())
        expect_equal("01", "manifest features missing from model_input_numeric", missing_manifest, 0)

if not available_samples.empty:
    expect_equal("01", "available labeled samples", len(available_samples), 102)

# Step 02
d02 = current_root / "02_modeling_dataset"
required_02 = [
    "modeling_table.tsv",
    "X_features.csv",
    "X_features_spatial_only.csv",
    "y_target.csv",
    "model_feature_manifest.csv",
    "sample_split.tsv",
    "split_assignments.tsv",
    "teacher_target_table.tsv",
    "feature_quality_report.tsv",
    "dataset_build_summary.txt",
]
for f in required_02:
    add_check("02", f"{f} exists", "PASS" if exists_file(d02 / f) else "FAIL", str(d02 / f), "present and nonempty", "")

modeling = safe_read(d02 / "modeling_table.tsv", "02", "modeling_table")
X = safe_read(d02 / "X_features.csv", "02", "X_features")
Xs = safe_read(d02 / "X_features_spatial_only.csv", "02", "X_features_spatial_only")
y = safe_read(d02 / "y_target.csv", "02", "y_target")
sample_split = safe_read(d02 / "sample_split.tsv", "02", "sample_split")
split_assignments = safe_read(d02 / "split_assignments.tsv", "02", "split_assignments")
feature_manifest = safe_read(d02 / "model_feature_manifest.csv", "02", "model_feature_manifest")
unmatched_spatial = safe_read(d02 / "unmatched_spatial_samples.tsv", "02", "unmatched_spatial_samples", required=False)
unmatched_teacher = safe_read(d02 / "unmatched_teacher_rows.tsv", "02", "unmatched_teacher_rows", required=False)

if not modeling.empty:
    expect_equal("02", "modeling rows", len(modeling), 3909)
    expect_equal("02", "modeling samples", unique_count(modeling, "sample_id"), 102)
    expect_equal("02", "modeling treatments", unique_count(modeling, "drug_key"), 40)

if not X.empty:
    expect_equal("02", "X rows", len(X), 3909)
    expect_equal("02", "X feature columns", X.shape[1], 671)
    drug_cols = [c for c in X.columns if str(c).startswith("drug__")]
    expect_equal("02", "drug dummy feature columns", len(drug_cols), 40)
    expect_equal("02", "spatial feature columns in X", X.shape[1] - len(drug_cols), 631)

if not Xs.empty:
    expect_equal("02", "spatial only X rows", len(Xs), 3909)
    expect_equal("02", "spatial only X columns", Xs.shape[1], 631)

if not y.empty:
    expect_equal("02", "y rows", len(y), 3909)

    target_candidates = [
        "fused_prob_responder",
        "target",
        "y",
        "response",
        "prob_responder",
    ]

    ycol = None

    for candidate in target_candidates:
        if candidate in y.columns:
            vals = pd.to_numeric(y[candidate], errors="coerce")
            if vals.notna().sum() > 0 and float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0:
                ycol = candidate
                break

    if ycol is None:
        best_cols = []

        for col in y.columns:
            vals = pd.to_numeric(y[col], errors="coerce")

            if vals.notna().sum() == 0:
                continue

            vmin = float(vals.min())
            vmax = float(vals.max())
            nunique = int(vals.nunique(dropna=True))

            is_index_like = (
                vmin == 0.0 and
                vmax == float(len(y) - 1) and
                nunique == len(y)
            )

            if is_index_like:
                continue

            if vmin >= 0.0 and vmax <= 1.0:
                best_cols.append((col, nunique))

        if best_cols:
            best_cols = sorted(best_cols, key=lambda x: x[1], reverse=True)
            ycol = best_cols[0][0]

    if ycol is None:
        add_check("02", "resolved y target column", "WARN", "unresolved", "probability-like y column", "Verifier could not identify target column in y_target.csv")
    else:
        yy = pd.to_numeric(y[ycol], errors="coerce")
        add_check("02", "resolved y target column", "INFO", ycol, "probability-like target column", "")
        expect_between("02", "target minimum in y", float(yy.min()), 0.0, 1.0)
        expect_between("02", "target maximum in y", float(yy.max()), 0.0, 1.0)

if not sample_split.empty:
    expect_equal("02", "sample split rows", len(sample_split), 102)
    if "split" in sample_split.columns:
        split_counts = sample_split["split"].astype(str).value_counts().to_dict()
        expect_equal("02", "train samples", int(split_counts.get("train", 0)), 81)
        expect_equal("02", "test samples", int(split_counts.get("test", 0)), 21)

if not split_assignments.empty:
    expect_equal("02", "split assignment rows", len(split_assignments), 3909)

# Step 03
d03 = current_root / "03_global_model"
required_03 = [
    "model.joblib",
    "metrics.tsv",
    "predictions_train.tsv",
    "predictions_test.tsv",
    "predictions_all_labeled.tsv",
    "feature_importance.tsv",
    "global_model_summary.txt",
    "run_config.json",
]
for f in required_03:
    add_check("03", f"{f} exists", "PASS" if exists_file(d03 / f) else "FAIL", str(d03 / f), "present and nonempty", "")

metrics = safe_read(d03 / "metrics.tsv", "03", "metrics")
pred_train = safe_read(d03 / "predictions_train.tsv", "03", "predictions_train")
pred_test = safe_read(d03 / "predictions_test.tsv", "03", "predictions_test")
pred_all = safe_read(d03 / "predictions_all_labeled.tsv", "03", "predictions_all_labeled")
feature_importance = safe_read(d03 / "feature_importance.tsv", "03", "feature_importance")

if exists_file(d03 / "model.joblib"):
    model_size_mb = (d03 / "model.joblib").stat().st_size / (1024 * 1024)
    expect_at_least("03", "model.joblib size MB", round(model_size_mb, 2), 10)

expect_equal("03", "train prediction rows", len(pred_train), 3107)
expect_equal("03", "test prediction rows", len(pred_test), 802)
expect_equal("03", "all labeled prediction rows", len(pred_all), 3909)

if not feature_importance.empty:
    expect_equal("03", "feature importance rows", len(feature_importance), 671)

if not metrics.empty:
    split_col = "split" if "split" in metrics.columns else None
    for split_name in ["train", "test", "all_labeled"]:
        row = metrics[metrics[split_col].astype(str).eq(split_name)] if split_col else pd.DataFrame()
        if len(row):
            row = row.iloc[0]
            for metric_name, threshold in [("mae", 0.25), ("rmse", 0.30)]:
                if metric_name in row:
                    value = float(row[metric_name])
                    status = "PASS" if value <= threshold else "WARN"
                    add_check("03", f"{split_name} {metric_name}", status, value, f"<= {threshold}", "")
            for metric_name, threshold in [("r2", 0.50), ("pearson", 0.80)]:
                if metric_name in row:
                    value = float(row[metric_name])
                    status = "PASS" if value >= threshold else "WARN"
                    add_check("03", f"{split_name} {metric_name}", status, value, f">= {threshold}", "")

# Step 04
d04 = current_root / "04_per_treatment_models"
required_04 = [
    "per_treatment_model_summary.tsv",
    "skipped_treatments.tsv",
    "per_treatment_predictions_all.tsv",
    "per_treatment_feature_importance_top.tsv",
    "per_treatment_model_summary.txt",
    "run_config.json",
]
for f in required_04:
    add_check("04", f"{f} exists", "PASS" if exists_file(d04 / f) else "FAIL", str(d04 / f), "present and nonempty", "")

pt = safe_read(d04 / "per_treatment_model_summary.tsv", "04", "per_treatment_model_summary")
skipped = safe_read(d04 / "skipped_treatments.tsv", "04", "skipped_treatments", required=False)
pt_pred = safe_read(d04 / "per_treatment_predictions_all.tsv", "04", "per_treatment_predictions_all", required=False)

if not pt.empty:
    expect_equal("04", "per treatment rows", len(pt), 40)
    status_counts = pt["status"].astype(str).value_counts().to_dict() if "status" in pt.columns else {}
    expect_equal("04", "per treatment trained", int(status_counts.get("trained", 0)), 15)
    expect_equal("04", "per treatment skipped", int(status_counts.get("skipped", 0)), 25)
    expect_equal("04", "per treatment failed", int(status_counts.get("failed", 0)), 0)

    if int(status_counts.get("skipped", 0)) > 0 and "reason" in pt.columns:
        top_reason = pt.loc[pt["status"].astype(str).eq("skipped"), "reason"].astype(str).value_counts().head(1).to_dict()
        add_flag("INFO", "per_treatment_models", "Some treatment models were skipped", str(top_reason))

model_files = list((d04 / "models").glob("*.joblib")) if (d04 / "models").exists() else []
prediction_files = list((d04 / "predictions").glob("*_predictions.tsv")) if (d04 / "predictions").exists() else []
add_check("04", "per treatment model files", "PASS" if len(model_files) == 15 else "FAIL", len(model_files), 15, "")
add_check("04", "per treatment prediction files", "PASS" if len(prediction_files) == 15 else "FAIL", len(prediction_files), 15, "")

# Step 05
d05 = current_root / "05_model_explanation"
required_05 = [
    "global_feature_explanations.tsv",
    "global_native_feature_importance.tsv",
    "global_permutation_importance.tsv",
    "global_feature_group_summary.tsv",
    "global_feature_axis_summary.tsv",
    "global_spatial_vs_drug_summary.tsv",
    "model_explanation_summary.txt",
    "prediction_sanity_metrics.tsv",
    "run_config.json",
]
for f in required_05:
    add_check("05", f"{f} exists", "PASS" if exists_file(d05 / f) else "FAIL", str(d05 / f), "present and nonempty", "")

gfe = safe_read(d05 / "global_feature_explanations.tsv", "05", "global_feature_explanations")
perm = safe_read(d05 / "global_permutation_importance.tsv", "05", "global_permutation_importance", required=False)
svd = safe_read(d05 / "global_spatial_vs_drug_summary.tsv", "05", "global_spatial_vs_drug_summary", required=False)
sanity = safe_read(d05 / "prediction_sanity_metrics.tsv", "05", "prediction_sanity_metrics", required=False)

if not gfe.empty:
    expect_equal("05", "global feature explanations rows", len(gfe), 671)

if not perm.empty:
    expect_equal("05", "permutation importance rows", len(perm), 671)

if not sanity.empty and "metric_value" in sanity.columns:
    test_rows = sanity[sanity["split"].astype(str).eq("test")]
    if len(test_rows):
        test_r2 = float(test_rows.iloc[0]["metric_value"])
        expect_at_least("05", "prediction sanity test r2", round(test_r2, 4), 0.50)

if not svd.empty and "feature_class" in svd.columns and "fraction_of_total_score" in svd.columns:
    row = svd[svd["feature_class"].astype(str).eq("drug_identity")]
    if len(row):
        frac = float(row.iloc[0]["fraction_of_total_score"])
        if frac > 0.90:
            add_flag("WARN", "interpretation", "Global explanation is dominated by drug identity", f"drug_identity fraction={frac:.4f}")
        else:
            add_flag("PASS", "interpretation", "Drug identity does not dominate global explanation", f"drug_identity fraction={frac:.4f}")

fig05 = list((d05 / "figures").glob("*.png")) if (d05 / "figures").exists() else []
expect_at_least("05", "step 05 figures", len(fig05), 3)

# Step 06
d06 = current_root / "06_all_sample_predictions"
required_06 = [
    "all_sample_treatment_predictions.tsv",
    "all_sample_treatment_predictions.csv",
    "prediction_matrix_sample_by_treatment.tsv",
    "prediction_summary_by_sample.tsv",
    "prediction_summary_by_treatment.tsv",
    "top_treatment_per_sample.tsv",
    "teacher_labeled_prediction_comparison.tsv",
    "prediction_sample_manifest.tsv",
    "prediction_treatment_manifest.tsv",
    "all_sample_prediction_summary.txt",
    "run_config.json",
]
for f in required_06:
    add_check("06", f"{f} exists", "PASS" if exists_file(d06 / f) else "FAIL", str(d06 / f), "present and nonempty", "")

all_pred = safe_read(d06 / "all_sample_treatment_predictions.tsv", "06", "all_sample_treatment_predictions")
pred_matrix = safe_read(d06 / "prediction_matrix_sample_by_treatment.tsv", "06", "prediction_matrix_sample_by_treatment")
teacher_comp = safe_read(d06 / "teacher_labeled_prediction_comparison.tsv", "06", "teacher_labeled_prediction_comparison")
top_treat = safe_read(d06 / "top_treatment_per_sample.tsv", "06", "top_treatment_per_sample")

if not all_pred.empty:
    expect_equal("06", "all sample treatment prediction rows", len(all_pred), 4080)
    expect_equal("06", "prediction sample count", unique_count(all_pred, "sample_id"), 102)
    expect_equal("06", "prediction treatment count", unique_count(all_pred, "drug_key"), 40)
    pred_col = "predicted_fused_prob_responder" if "predicted_fused_prob_responder" in all_pred.columns else None
    if pred_col:
        preds = numeric_col(all_pred, pred_col)
        expect_equal("06", "nonmissing predictions", int(preds.notna().sum()), 4080)
        expect_between("06", "prediction minimum", float(preds.min()), 0.0, 1.0)
        expect_between("06", "prediction maximum", float(preds.max()), 0.0, 1.0)
        add_check("06", "prediction mean", "INFO", round(float(preds.mean()), 4), "review", "")

if not pred_matrix.empty:
    expect_equal("06", "prediction matrix rows", len(pred_matrix), 102)
    expect_at_least("06", "prediction matrix columns", pred_matrix.shape[1], 41)

if not teacher_comp.empty:
    expect_equal("06", "teacher labeled comparison rows", len(teacher_comp), 3909)

if not top_treat.empty:
    expect_equal("06", "top treatment per sample rows", len(top_treat), 102)

# Step 07
d07 = current_root / "07_prediction_qc"
required_07 = [
    "qc_summary.tsv",
    "qc_summary.txt",
    "qc_file_contract_report.tsv",
    "qc_dataset_summary.tsv",
    "qc_split_summary.tsv",
    "qc_model_metrics.tsv",
    "qc_teacher_overlap_metrics.tsv",
    "qc_prediction_distribution.tsv",
    "qc_by_sample.tsv",
    "qc_by_treatment.tsv",
    "qc_top_treatment_per_sample.tsv",
    "qc_feature_explanation_summary.tsv",
    "qc_spatial_vs_drug_summary.tsv",
    "qc_per_treatment_model_status.tsv",
    "presentation_figure_manifest.tsv",
    "run_config.json",
]
for f in required_07:
    add_check("07", f"{f} exists", "PASS" if exists_file(d07 / f) else "FAIL", str(d07 / f), "present and nonempty", "")

qc = safe_read(d07 / "qc_summary.tsv", "07", "qc_summary")
contract = safe_read(d07 / "qc_file_contract_report.tsv", "07", "qc_file_contract_report")
dataset_qc = safe_read(d07 / "qc_dataset_summary.tsv", "07", "qc_dataset_summary")
spatial_vs_drug_qc = safe_read(d07 / "qc_spatial_vs_drug_summary.tsv", "07", "qc_spatial_vs_drug_summary", required=False)

if not qc.empty:
    row = qc.iloc[0]
    expect_equal("07", "required files missing", int(row.get("required_files_missing", -1)), 0)
    expect_equal("07", "QC labeled rows", int(row.get("n_labeled_rows", -1)), 3909)
    expect_equal("07", "QC labeled samples", int(row.get("n_labeled_samples", -1)), 102)
    expect_equal("07", "QC labeled treatments", int(row.get("n_labeled_treatments", -1)), 40)
    expect_equal("07", "QC X features", int(row.get("n_x_features", -1)), 671)
    expect_equal("07", "QC prediction rows", int(row.get("n_prediction_rows", -1)), 4080)
    expect_equal("07", "QC per treatment failed", int(row.get("per_treatment_n_failed", -1)), 0)

    test_r2 = float(row.get("test_r2", np.nan))
    test_pearson = float(row.get("test_pearson", np.nan))
    teacher_pearson = float(row.get("teacher_overlap_pearson", np.nan))
    if not math.isnan(test_r2):
        expect_at_least("07", "QC test r2", round(test_r2, 4), 0.50)
    if not math.isnan(test_pearson):
        expect_at_least("07", "QC test pearson", round(test_pearson, 4), 0.80)
    if not math.isnan(teacher_pearson):
        expect_at_least("07", "QC teacher overlap pearson", round(teacher_pearson, 4), 0.90)

if not contract.empty and "status" in contract.columns:
    missing_required = int(contract["status"].astype(str).eq("missing_required").sum())
    expect_equal("07", "file contract missing_required", missing_required, 0)

fig07 = list((d07 / "figures").glob("*.png")) if (d07 / "figures").exists() else []
expect_at_least("07", "step 07 QC figures", len(fig07), 14)

# Compare with past runs
run_summaries = [summarize_run(current_root, current_run_name)]
for name in past_run_names:
    root = output_base / name
    if root.exists():
        run_summaries.append(summarize_run(root, name))
    else:
        run_summaries.append({"run_name": name, "path": str(root), "exists": False})

compare_df = pd.DataFrame(run_summaries)

if len(compare_df) > 1:
    current = compare_df[compare_df["run_name"].eq(current_run_name)].iloc[0].to_dict()
    for _, past in compare_df[~compare_df["run_name"].eq(current_run_name)].iterrows():
        past = past.to_dict()
        if not past.get("exists", False):
            add_flag("INFO", "comparison", f"Past run not found: {past.get('run_name')}", past.get("path", ""))
            continue

        past_name = past.get("run_name")

        for key in ["n_labeled_rows", "n_labeled_samples", "n_labeled_treatments", "n_x_features", "n_prediction_rows", "test_r2", "test_mae", "teacher_overlap_pearson"]:
            cur_val = current.get(key, np.nan)
            past_val = past.get(key, np.nan)
            if pd.notna(cur_val) and pd.notna(past_val):
                try:
                    delta = float(cur_val) - float(past_val)
                    add_check("COMPARE", f"{current_run_name} vs {past_name} {key}", "INFO", round(float(cur_val), 6), f"past={round(float(past_val), 6)} delta={round(delta, 6)}", "")
                except Exception:
                    add_check("COMPARE", f"{current_run_name} vs {past_name} {key}", "INFO", str(cur_val), f"past={past_val}", "")

        if past_name == "output_run_102":
            cur_features = current.get("n_x_features", np.nan)
            past_features = past.get("n_x_features", np.nan)
            if pd.notna(cur_features) and pd.notna(past_features) and float(cur_features) > float(past_features) * 2:
                add_flag("INFO", "comparison", "Current run has a much richer feature matrix than old output_run_102", f"current={cur_features}; old={past_features}")

            cur_r2 = current.get("test_r2", np.nan)
            past_r2 = past.get("test_r2", np.nan)
            if pd.notna(cur_r2) and pd.notna(past_r2) and float(cur_r2) < float(past_r2):
                add_flag("WARN", "comparison", "Current test R2 is lower than old output_run_102", f"current={cur_r2}; old={past_r2}; not apples to apples because feature/teacher handoff changed")

# Overall status
status_counts = pd.Series([c["status"] for c in checks]).value_counts().to_dict()
n_fail = int(status_counts.get("FAIL", 0))
n_warn = int(status_counts.get("WARN", 0))

summary_lines.append("Overall status")
summary_lines.append(f"  FAIL checks: {n_fail}")
summary_lines.append(f"  WARN checks: {n_warn}")
summary_lines.append(f"  PASS checks: {int(status_counts.get('PASS', 0))}")
summary_lines.append(f"  INFO checks: {int(status_counts.get('INFO', 0))}")
summary_lines.append("")

if n_fail == 0:
    summary_lines.append("VERDICT: PASS")
    summary_lines.append("The spatial_prediction_model output folder is complete enough to use downstream.")
else:
    summary_lines.append("VERDICT: FAIL")
    summary_lines.append("One or more required output checks failed. Review the TSV check file before using downstream.")

summary_lines.append("")
summary_lines.append("Interpretation notes")
summary_lines.append("  The expected full cohort scale is 102 samples, 40 treatments, 3909 labeled rows, and 4080 all sample treatment predictions.")
summary_lines.append("  Per treatment models are expected to skip treatments with low target variance. Skipped does not mean failed.")
summary_lines.append("  A high drug identity explanation fraction is a biological interpretation warning, not a file integrity failure.")
summary_lines.append("")

if flags:
    summary_lines.append("Flags")
    for f in flags:
        summary_lines.append(f"  {f['level']}: {f['category']}: {f['message']} {f['detail']}")
else:
    summary_lines.append("Flags")
    summary_lines.append("  No reasonableness flags recorded.")

pd.DataFrame(checks).to_csv(step_checks_tsv, sep="\t", index=False)
pd.DataFrame(flags).to_csv(flags_tsv, sep="\t", index=False)
compare_df.to_csv(compare_tsv, sep="\t", index=False)
report_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n".join(summary_lines))
print("")
print("Wrote:")
print(report_txt)
print(step_checks_tsv)
print(compare_tsv)
print(flags_tsv)

if n_fail > 0:
    sys.exit(2)
"@

Set-Content -Path $VerifierPy -Value $Code -Encoding UTF8

Write-Host ""
Write-Host "Running verifier:"
Write-Host $VerifierPy
Write-Host ""

& $Python $VerifierPy

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Verifier outputs:"
Write-Host $ReportTxt
Write-Host $StepChecksTsv
Write-Host $CompareTsv
Write-Host $FlagsTsv

if ($ExitCode -ne 0) {
    throw "Verification failed. Review $StepChecksTsv"
}




