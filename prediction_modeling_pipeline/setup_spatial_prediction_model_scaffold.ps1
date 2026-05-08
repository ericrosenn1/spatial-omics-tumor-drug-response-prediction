<#
.SYNOPSIS
    Create the spatial_prediction_model pipeline scaffold.

.DESCRIPTION
    Creates folders, starter files, a YAML configuration, and runner scripts for:

        C:\Users\Owner\OneDrive\spring 2026 NYU\Adv. Omics Fenyo\project\prediction_modeling_pipeline\spatial_prediction_model

    The scaffold is designed to consume teacher_builder step 05 handoff files:
        model_input_numeric.csv
        visium_fused_teacher_table.tsv
        feature_manifest.csv
        prediction_ready_training_table.tsv

    Initial mode is the 10-sample test run.
    Later full 102-sample execution should be controlled through the YAML file,
    not by editing Python scripts.

.USAGE
    powershell -ExecutionPolicy Bypass -File .\setup_spatial_prediction_model_scaffold.ps1
    powershell -ExecutionPolicy Bypass -File .\setup_spatial_prediction_model_scaffold.ps1 -Force
#>

param(
    [string]$Root = "C:\Users\Owner\OneDrive\spring 2026 NYU\Adv. Omics Fenyo\project\prediction_modeling_pipeline\spatial_prediction_model",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function New-DirectoryIfMissing {
    param([Parameter(Mandatory=$true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "[DIR ] created: $Path"
    }
    else {
        Write-Host "[DIR ] exists:  $Path"
    }
}

function Write-FileIfMissingOrForced {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Content,
        [switch]$ForceWrite
    )

    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    if ((Test-Path -LiteralPath $Path) -and (-not $ForceWrite)) {
        Write-Host "[FILE] exists, skipped: $Path"
        return
    }

    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.Encoding]::UTF8)
    Write-Host "[FILE] wrote: $Path"
}

function New-PythonStepStub {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$ScriptName,
        [Parameter(Mandatory=$true)][string]$StepTitle,
        [Parameter(Mandatory=$true)][string]$OutputSubdir,
        [Parameter(Mandatory=$true)][string]$Purpose,
        [Parameter(Mandatory=$true)][string[]]$InputKeys,
        [Parameter(Mandatory=$true)][string[]]$OutputFiles
    )

    $inputLiteral = ($InputKeys | ForEach-Object { "    `"$_`"," }) -join "`n"
    $outputLiteral = ($OutputFiles | ForEach-Object { "    `"$_`"," }) -join "`n"

    $content = @"
`"`"`"
Script:
    $ScriptName

Purpose:
    $Purpose

Pipeline contract:
    YAML-driven step in spatial_prediction_model.
    No hard-coded project-specific edits in this file.
    The config controls 10-sample test mode and later 102-sample full mode.

Expected input config keys:
$($InputKeys | ForEach-Object { "    - $_" } | Out-String)
Expected outputs under output_root/${OutputSubdir}:
$($OutputFiles | ForEach-Object { "    - $_" } | Out-String)
Implementation status:
    Scaffold only.
    Replace the TODO block with real logic when this step is built.
`"`"`"

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

import yaml


SCRIPT_NAME = "$ScriptName"
STEP_TITLE = "$StepTitle"
OUTPUT_SUBDIR = "$OutputSubdir"
EXPECTED_INPUT_KEYS = [
$inputLiteral
]
EXPECTED_OUTPUT_FILES = [
$outputLiteral
]


def parse_args() -> argparse.Namespace:
    """parse command line args"""
    parser = argparse.ArgumentParser(description=STEP_TITLE)
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(project_dir: Path, value: str | None) -> Path | None:
    """resolve absolute or project-relative path"""
    if value in [None, ""]:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_dir / path


def ensure_dir(path: Path) -> None:
    """make output folder"""
    path.mkdir(parents=True, exist_ok=True)


def write_step_contract(out_dir: Path, cfg: dict[str, Any]) -> None:
    """write scaffold input-output contract"""
    rows = {
        "script_name": SCRIPT_NAME,
        "step_title": STEP_TITLE,
        "output_subdir": OUTPUT_SUBDIR,
        "status": "scaffold_only",
        "expected_input_keys": EXPECTED_INPUT_KEYS,
        "expected_output_files": EXPECTED_OUTPUT_FILES,
        "run_scope": cfg.get("run_scope"),
        "test_mode": cfg.get("test_mode"),
        "test_n_samples": cfg.get("test_n_samples"),
    }

    with open(out_dir / "step_contract.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with open(out_dir / "README_step_contract.txt", "w", encoding="utf-8") as handle:
        handle.write(f"{SCRIPT_NAME}\n")
        handle.write(f"{STEP_TITLE}\n\n")
        handle.write("This is a scaffold file. Build the implementation before using it for modeling.\n\n")
        handle.write("Expected input config keys:\n")
        for key in EXPECTED_INPUT_KEYS:
            handle.write(f"  - {key}\n")
        handle.write("\nExpected output files:\n")
        for file_name in EXPECTED_OUTPUT_FILES:
            handle.write(f"  - {file_name}\n")


def main() -> None:
    """run scaffold step"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    project_dir = Path(cfg["project_dir"])
    output_root = Path(cfg["output_root"])
    out_dir = output_root / OUTPUT_SUBDIR
    ensure_dir(out_dir)

    # TODO: replace scaffold contract with real implementation.
    # Note: all paths should be read from cfg and resolved through resolve_path().
    write_step_contract(out_dir, cfg)

    print(f"[{SCRIPT_NAME}] scaffold contract written: {out_dir}")
    print("Implementation TODO: replace scaffold logic with actual pipeline code.")


if __name__ == "__main__":
    main()
"@

    Write-FileIfMissingOrForced -Path $Path -Content $content -ForceWrite:$Force
}

# ------------------------------------------------------------
# Root and folders
# ------------------------------------------------------------

$Root = [System.IO.Path]::GetFullPath($Root)
Write-Host ""
Write-Host "Spatial prediction model scaffold root:"
Write-Host "  $Root"
Write-Host ""

$dirs = @(
    $Root,
    (Join-Path $Root "configs"),
    (Join-Path $Root "scripts"),
    (Join-Path $Root "docs"),
    (Join-Path $Root "logs"),
    (Join-Path $Root "outputs"),
    (Join-Path $Root "outputs\output_run_10"),
    (Join-Path $Root "outputs\output_run_10\01_input_validation"),
    (Join-Path $Root "outputs\output_run_10\02_modeling_dataset"),
    (Join-Path $Root "outputs\output_run_10\03_global_model"),
    (Join-Path $Root "outputs\output_run_10\04_per_treatment_models"),
    (Join-Path $Root "outputs\output_run_10\04_per_treatment_models\models"),
    (Join-Path $Root "outputs\output_run_10\04_per_treatment_models\predictions"),
    (Join-Path $Root "outputs\output_run_10\04_per_treatment_models\feature_importance"),
    (Join-Path $Root "outputs\output_run_10\05_model_explanation"),
    (Join-Path $Root "outputs\output_run_10\06_all_sample_predictions"),
    (Join-Path $Root "outputs\output_run_10\07_prediction_qc"),
    (Join-Path $Root "outputs\output_run_10\pipeline_run_logs"),
    (Join-Path $Root "outputs\output_run_102"),
    (Join-Path $Root "outputs\output_run_102\01_input_validation"),
    (Join-Path $Root "outputs\output_run_102\02_modeling_dataset"),
    (Join-Path $Root "outputs\output_run_102\03_global_model"),
    (Join-Path $Root "outputs\output_run_102\04_per_treatment_models"),
    (Join-Path $Root "outputs\output_run_102\04_per_treatment_models\models"),
    (Join-Path $Root "outputs\output_run_102\04_per_treatment_models\predictions"),
    (Join-Path $Root "outputs\output_run_102\04_per_treatment_models\feature_importance"),
    (Join-Path $Root "outputs\output_run_102\05_model_explanation"),
    (Join-Path $Root "outputs\output_run_102\06_all_sample_predictions"),
    (Join-Path $Root "outputs\output_run_102\07_prediction_qc"),
    (Join-Path $Root "outputs\output_run_102\pipeline_run_logs")
)

foreach ($dir in $dirs) {
    New-DirectoryIfMissing -Path $dir
}

# ------------------------------------------------------------
# YAML config
# ------------------------------------------------------------

$yaml = @'
# =====================================================================
# spatial_prediction_model.yaml
# =====================================================================
# Purpose:
#   YAML control file for the spatial_prediction_model pipeline.
#   Initial mode: 10-sample test run using teacher_builder handoff files.
#   Full mode later: change run_name/output_root/test flags here only.
#
# Primary data flow:
#   teacher_builder/outputs/05_prediction_ready_teacher
#       -> spatial_prediction_model/scripts/01-07
#       -> spatial_prediction_model/outputs/output_run_10 or output_run_102
# =====================================================================

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
project_dir: "C:/Users/Owner/OneDrive/spring 2026 NYU/Adv. Omics Fenyo/project"
pipeline_dir: "C:/Users/Owner/OneDrive/spring 2026 NYU/Adv. Omics Fenyo/project/prediction_modeling_pipeline"
spatial_prediction_model_dir: "C:/Users/Owner/OneDrive/spring 2026 NYU/Adv. Omics Fenyo/project/prediction_modeling_pipeline/spatial_prediction_model"

pipeline_name: "spatial_prediction_model"
run_name: "output_run_10"
output_root: "C:/Users/Owner/OneDrive/spring 2026 NYU/Adv. Omics Fenyo/project/prediction_modeling_pipeline/spatial_prediction_model/outputs/output_run_10"

# ---------------------------------------------------------------------
# Run scope
# ---------------------------------------------------------------------
# Current first pass: train/check on 10 labeled teacher samples.
# Later full run:
#   run_name: "output_run_102"
#   output_root: ".../outputs/output_run_102"
#   run_scope: "full_102"
#   test_mode: false
#   limit_training_to_test_samples: false
#   prediction_sample_mode: "all_spatial_samples"
#   run_per_treatment_models: true
run_scope: "test_10"
test_mode: true
test_n_samples: 10
sample_selection_mode: "first_n_labeled_samples"
limit_training_to_test_samples: true
prediction_sample_mode: "test_labeled_samples"
max_prediction_samples: 10

# ---------------------------------------------------------------------
# Teacher-builder handoff inputs
# ---------------------------------------------------------------------
teacher_builder_output_dir: "prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher"
model_input_numeric: "prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/model_input_numeric.csv"
teacher_table: "prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/visium_fused_teacher_table.tsv"
feature_manifest: "prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/feature_manifest.csv"
training_table: "prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/prediction_ready_training_table.tsv"

# ---------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------
sample_col: "sample_id"
slide_col: "slide_id"
drug_col: "drug"
drug_key_col: "drug_key"
target_col: "fused_prob_responder"
confidence_col: "fused_confidence"

# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------
# Script 02 should build model_feature_manifest.csv from feature_manifest.csv
# and remove leakage columns listed below.
feature_selection_source: "feature_manifest"
feature_name_col: "feature_name"
feature_group_col: "feature_group"
include_only_numeric_features: true
include_drug_dummies: true
drug_dummy_prefix: "drug__"

# Do not let model train on teacher construction fields.
leakage_excluded_columns:
  - "fused_prob_responder"
  - "fused_confidence"
  - "fused_ci_low"
  - "fused_ci_high"
  - "expression_prob_responder"
  - "histology_prob_responder"
  - "expression_confidence"
  - "histology_confidence"
  - "expression_sample_confidence"
  - "histology_sample_confidence"
  - "expression_ci_low"
  - "expression_ci_high"
  - "histology_ci_low"
  - "histology_ci_high"
  - "modality_used"
  - "expression_available"
  - "histology_available"
  - "n_tiles_used"
  - "hires_image_path"
  - "source_image_path"

metadata_passthrough_columns:
  - "sample_id"
  - "slide_id"
  - "dataset_id"
  - "cancer_type"
  - "drug"
  - "drug_key"
  - "modality_used"

# ---------------------------------------------------------------------
# Modeling task
# ---------------------------------------------------------------------
task: "regression"
binary_threshold: 0.5
model_type: "random_forest"
random_state: 42
n_jobs: -1

# Group split avoids putting rows from the same sample into both train and test.
split_strategy: "group_holdout"
split_group_col: "sample_id"
test_size: 0.20
allow_small_test_mode_metrics: true

random_forest:
  n_estimators: 500
  max_depth: null
  min_samples_leaf: 1
  min_samples_split: 2
  max_features: "sqrt"
  bootstrap: true

xgboost:
  enabled: false
  n_estimators: 500
  max_depth: 4
  learning_rate: 0.03
  subsample: 0.85
  colsample_bytree: 0.85
  objective: "reg:squarederror"

# ---------------------------------------------------------------------
# Optional per-treatment models
# ---------------------------------------------------------------------
run_per_treatment_models: false
min_samples_per_treatment: 30
min_target_std: 0.02

# ---------------------------------------------------------------------
# Explanation / SHAP
# ---------------------------------------------------------------------
run_shap: true
shap_background_n: 100
shap_sample_n: 400
write_full_shap_values: true
write_filtered_spatial_shap: true
filter_out_drug_dummy_for_spatial_shap: true

# ---------------------------------------------------------------------
# All sample-treatment prediction
# ---------------------------------------------------------------------
run_all_sample_predictions: true
prediction_treatment_source: "training_table_unique_drugs"
write_teacher_overlap_columns: true
clip_predictions_to_01: true

# ---------------------------------------------------------------------
# Step toggles
# ---------------------------------------------------------------------
run_steps:
  01_validate_prediction_inputs: true
  02_build_spatial_modeling_dataset: true
  03_train_global_spatial_response_model: true
  04_train_per_treatment_models: false
  05_explain_spatial_response_model: true
  06_predict_all_sample_treatment_pairs: true
  07_qc_spatial_prediction_outputs: true

# ---------------------------------------------------------------------
# Output subdirectories
# ---------------------------------------------------------------------
output_subdirs:
  input_validation: "01_input_validation"
  modeling_dataset: "02_modeling_dataset"
  global_model: "03_global_model"
  per_treatment_models: "04_per_treatment_models"
  model_explanation: "05_model_explanation"
  all_sample_predictions: "06_all_sample_predictions"
  prediction_qc: "07_prediction_qc"
  pipeline_run_logs: "pipeline_run_logs"

# ---------------------------------------------------------------------
# Expected primary outputs by step
# ---------------------------------------------------------------------
expected_outputs:
  01_validate_prediction_inputs:
    - "01_input_validation/input_validation_summary.txt"
    - "01_input_validation/input_table_shapes.tsv"
    - "01_input_validation/input_column_report.tsv"
  02_build_spatial_modeling_dataset:
    - "02_modeling_dataset/modeling_table.tsv"
    - "02_modeling_dataset/X_features.csv"
    - "02_modeling_dataset/y_target.csv"
    - "02_modeling_dataset/model_feature_manifest.csv"
    - "02_modeling_dataset/leakage_excluded_columns.tsv"
    - "02_modeling_dataset/sample_split.tsv"
  03_train_global_spatial_response_model:
    - "03_global_model/model.joblib"
    - "03_global_model/predictions_train.tsv"
    - "03_global_model/predictions_test.tsv"
    - "03_global_model/metrics.tsv"
    - "03_global_model/feature_importance.tsv"
    - "03_global_model/run_config.json"
  04_train_per_treatment_models:
    - "04_per_treatment_models/per_treatment_model_summary.tsv"
  05_explain_spatial_response_model:
    - "05_model_explanation/shap_summary.tsv"
    - "05_model_explanation/filtered_spatial_shap_summary.tsv"
  06_predict_all_sample_treatment_pairs:
    - "06_all_sample_predictions/all_sample_treatment_predictions.tsv"
  07_qc_spatial_prediction_outputs:
    - "07_prediction_qc/qc_summary.tsv"
    - "07_prediction_qc/qc_summary.txt"
'@

Write-FileIfMissingOrForced `
    -Path (Join-Path $Root "configs\spatial_prediction_model.yaml") `
    -Content $yaml `
    -ForceWrite:$Force

# ------------------------------------------------------------
# README
# ------------------------------------------------------------

$readme = @'
# spatial_prediction_model

This folder is the modeling stage after `teacher_builder`.

Input handoff comes from:

```text
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/
    model_input_numeric.csv
    visium_fused_teacher_table.tsv
    feature_manifest.csv
    prediction_ready_training_table.tsv
```

Initial run mode is controlled by:

```text
configs/spatial_prediction_model.yaml
```

The scaffold starts in 10-sample mode:

```yaml
run_name: "output_run_10"
test_mode: true
test_n_samples: 10
limit_training_to_test_samples: true
prediction_sample_mode: "test_labeled_samples"
```

For the later full run, edit YAML only:

```yaml
run_name: "output_run_102"
output_root: ".../spatial_prediction_model/outputs/output_run_102"
run_scope: "full_102"
test_mode: false
limit_training_to_test_samples: false
prediction_sample_mode: "all_spatial_samples"
run_per_treatment_models: true
```

Planned steps:

```text
01_validate_prediction_inputs.py
02_build_spatial_modeling_dataset.py
03_train_global_spatial_response_model.py
04_train_per_treatment_models.py
05_explain_spatial_response_model.py
06_predict_all_sample_treatment_pairs.py
07_qc_spatial_prediction_outputs.py
```

The Python files are scaffold stubs. They write step contracts now and should be filled in one by one.
'@

Write-FileIfMissingOrForced `
    -Path (Join-Path $Root "README.md") `
    -Content $readme `
    -ForceWrite:$Force

# ------------------------------------------------------------
# Python step stubs
# ------------------------------------------------------------

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\01_validate_prediction_inputs.py") `
    -ScriptName "01_validate_prediction_inputs.py" `
    -StepTitle "Validate teacher-builder prediction inputs" `
    -OutputSubdir "01_input_validation" `
    -Purpose "Check teacher-builder handoff files, required columns, shapes, duplicate sample-drug rows, and target availability." `
    -InputKeys @("model_input_numeric", "teacher_table", "feature_manifest", "training_table", "sample_col", "drug_key_col", "target_col") `
    -OutputFiles @("input_validation_summary.txt", "input_table_shapes.tsv", "input_column_report.tsv")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\02_build_spatial_modeling_dataset.py") `
    -ScriptName "02_build_spatial_modeling_dataset.py" `
    -StepTitle "Build spatial modeling dataset" `
    -OutputSubdir "02_modeling_dataset" `
    -Purpose "Create leakage-safe ML table from prediction_ready_training_table.tsv and feature_manifest.csv." `
    -InputKeys @("training_table", "feature_manifest", "sample_col", "drug_col", "drug_key_col", "target_col", "leakage_excluded_columns", "include_drug_dummies") `
    -OutputFiles @("modeling_table.tsv", "X_features.csv", "y_target.csv", "model_feature_manifest.csv", "leakage_excluded_columns.tsv", "sample_split.tsv")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\03_train_global_spatial_response_model.py") `
    -ScriptName "03_train_global_spatial_response_model.py" `
    -StepTitle "Train global spatial response model" `
    -OutputSubdir "03_global_model" `
    -Purpose "Train pooled sample-treatment model using spatial features plus optional drug identity features." `
    -InputKeys @("output_root", "model_type", "task", "random_forest", "xgboost", "split_strategy", "split_group_col", "target_col") `
    -OutputFiles @("model.joblib", "predictions_train.tsv", "predictions_test.tsv", "metrics.tsv", "feature_importance.tsv", "run_config.json")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\04_train_per_treatment_models.py") `
    -ScriptName "04_train_per_treatment_models.py" `
    -StepTitle "Train per-treatment spatial response models" `
    -OutputSubdir "04_per_treatment_models" `
    -Purpose "Optional drug-specific models when enough samples and target variation are available." `
    -InputKeys @("run_per_treatment_models", "min_samples_per_treatment", "min_target_std", "model_type", "task") `
    -OutputFiles @("per_treatment_model_summary.tsv", "models/<drug_key>.joblib", "predictions/<drug_key>_predictions.tsv", "feature_importance/<drug_key>_feature_importance.tsv")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\05_explain_spatial_response_model.py") `
    -ScriptName "05_explain_spatial_response_model.py" `
    -StepTitle "Explain spatial response model" `
    -OutputSubdir "05_model_explanation" `
    -Purpose "Compute feature importance and SHAP views for full predictive and spatial-only interpretation." `
    -InputKeys @("run_shap", "shap_background_n", "shap_sample_n", "write_filtered_spatial_shap", "filter_out_drug_dummy_for_spatial_shap") `
    -OutputFiles @("shap_values_global.tsv", "shap_summary.tsv", "filtered_spatial_shap_summary.tsv", "fig_shap_top_features.png", "fig_filtered_spatial_shap_top_features.png")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\06_predict_all_sample_treatment_pairs.py") `
    -ScriptName "06_predict_all_sample_treatment_pairs.py" `
    -StepTitle "Predict all sample-treatment pairs" `
    -OutputSubdir "06_all_sample_predictions" `
    -Purpose "Apply trained global model to selected samples and treatments, including unlabeled samples when configured." `
    -InputKeys @("model_input_numeric", "prediction_treatment_source", "prediction_sample_mode", "max_prediction_samples", "write_teacher_overlap_columns") `
    -OutputFiles @("all_sample_treatment_predictions.tsv")

New-PythonStepStub `
    -Path (Join-Path $Root "scripts\07_qc_spatial_prediction_outputs.py") `
    -ScriptName "07_qc_spatial_prediction_outputs.py" `
    -StepTitle "QC spatial prediction outputs" `
    -OutputSubdir "07_prediction_qc" `
    -Purpose "Generate model QC tables and figures for predictions, metrics, and explanations." `
    -InputKeys @("output_root", "target_col", "drug_key_col", "sample_col", "expected_outputs") `
    -OutputFiles @("qc_summary.tsv", "qc_summary.txt", "qc_by_treatment.tsv", "qc_by_sample.tsv", "fig_01_observed_vs_predicted.png", "fig_02_prediction_distribution.png")

# ------------------------------------------------------------
# Python orchestrator
# ------------------------------------------------------------

$runnerPy = @'
"""
Script:
    run_spatial_prediction_model.py

Purpose:
    Run spatial_prediction_model steps in order.

Notes:
    YAML-driven runner.
    Logs every step under output_root/pipeline_run_logs/run_<timestamp>.
    Step scripts are scaffold stubs until implemented.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import subprocess
import sys
from typing import Iterable

import yaml


STEPS = [
    ("01", "01_validate_prediction_inputs", "scripts/01_validate_prediction_inputs.py"),
    ("02", "02_build_spatial_modeling_dataset", "scripts/02_build_spatial_modeling_dataset.py"),
    ("03", "03_train_global_spatial_response_model", "scripts/03_train_global_spatial_response_model.py"),
    ("04", "04_train_per_treatment_models", "scripts/04_train_per_treatment_models.py"),
    ("05", "05_explain_spatial_response_model", "scripts/05_explain_spatial_response_model.py"),
    ("06", "06_predict_all_sample_treatment_pairs", "scripts/06_predict_all_sample_treatment_pairs.py"),
    ("07", "07_qc_spatial_prediction_outputs", "scripts/07_qc_spatial_prediction_outputs.py"),
]


def parse_args() -> argparse.Namespace:
    """parse CLI args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spatial_prediction_model.yaml")
    parser.add_argument(
        "--steps",
        default="01,02,03,04,05,06,07",
        help="Comma-separated step numbers, e.g. 01,02,03",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """load YAML config"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_steps(step_text: str) -> set[str]:
    """normalize selected step numbers"""
    return {part.strip().zfill(2) for part in step_text.split(",") if part.strip()}


def should_run_step(cfg: dict, step_name: str) -> bool:
    """check run_steps config flag"""
    toggles = cfg.get("run_steps", {}) or {}
    return bool(toggles.get(step_name, True))


def run_step(root: Path, config_path: Path, log_dir: Path, step_num: str, step_name: str, script_rel: str) -> None:
    """run one pipeline step and save log"""
    script_path = root / script_rel
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    log_path = log_dir / f"step_{step_num}_{step_name}.log"
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]

    print(f"\n[{step_num}] {step_name}")
    print(" ".join(cmd))
    print(f"log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {step_num} {step_name}; see {log_path}")


def main() -> None:
    """run selected steps"""
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_config(config_path)
    output_root = Path(cfg["output_root"])
    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_root / "pipeline_run_logs" / f"run_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = selected_steps(args.steps)

    print("spatial_prediction_model runner")
    print(f"root: {root}")
    print(f"config: {config_path}")
    print(f"output_root: {output_root}")
    print(f"steps: {sorted(wanted)}")

    for step_num, step_name, script_rel in STEPS:
        if step_num not in wanted:
            print(f"skip {step_num}: not selected")
            continue

        if not should_run_step(cfg, step_name):
            print(f"skip {step_num}: disabled in YAML run_steps.{step_name}")
            continue

        run_step(root, config_path, log_dir, step_num, step_name, script_rel)

    print("\nDONE")
    print(f"logs: {log_dir}")


if __name__ == "__main__":
    main()
'@

Write-FileIfMissingOrForced `
    -Path (Join-Path $Root "run_spatial_prediction_model.py") `
    -Content $runnerPy `
    -ForceWrite:$Force

# ------------------------------------------------------------
# PowerShell runner
# ------------------------------------------------------------

$runnerPs1 = @'
<#
.SYNOPSIS
    Run spatial_prediction_model pipeline steps.

.EXAMPLES
    .\run_spatial_prediction_model.ps1
    .\run_spatial_prediction_model.ps1 -Steps "01,02,03"
    .\run_spatial_prediction_model.ps1 -Config "configs\spatial_prediction_model.yaml" -Steps "01,02,03,05,06,07"
#>

param(
    [string]$Config = "configs\spatial_prediction_model.yaml",
    [string]$Steps = "01,02,03,04,05,06,07"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

Write-Host "spatial_prediction_model"
Write-Host "Root:   $Here"
Write-Host "Config: $Config"
Write-Host "Steps:  $Steps"
Write-Host ""

python ".\run_spatial_prediction_model.py" --config $Config --steps $Steps
'@

Write-FileIfMissingOrForced `
    -Path (Join-Path $Root "run_spatial_prediction_model.ps1") `
    -Content $runnerPs1 `
    -ForceWrite:$Force

# ------------------------------------------------------------
# Marker files
# ------------------------------------------------------------

$gitkeep = "# keep folder`n"
foreach ($keepDir in @("docs", "logs")) {
    Write-FileIfMissingOrForced `
        -Path (Join-Path $Root "$keepDir\.gitkeep") `
        -Content $gitkeep `
        -ForceWrite:$Force
}

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

Write-Host ""
Write-Host "Scaffold complete."
Write-Host ""
Write-Host "Created/checked:"
Write-Host "  $Root\configs\spatial_prediction_model.yaml"
Write-Host "  $Root\scripts\01_validate_prediction_inputs.py"
Write-Host "  $Root\scripts\02_build_spatial_modeling_dataset.py"
Write-Host "  $Root\scripts\03_train_global_spatial_response_model.py"
Write-Host "  $Root\scripts\04_train_per_treatment_models.py"
Write-Host "  $Root\scripts\05_explain_spatial_response_model.py"
Write-Host "  $Root\scripts\06_predict_all_sample_treatment_pairs.py"
Write-Host "  $Root\scripts\07_qc_spatial_prediction_outputs.py"
Write-Host "  $Root\run_spatial_prediction_model.py"
Write-Host "  $Root\run_spatial_prediction_model.ps1"
Write-Host ""
Write-Host "Next command after scaffold:"
Write-Host "  cd `"$Root`""
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_spatial_prediction_model.ps1 -Steps `"01,02,03`""
Write-Host ""
Write-Host "Note: Python step files are scaffold contracts only until we implement each script."

