# ============================================================
# Runner: run_teacher_builder_governed.ps1
# ============================================================
#
# Purpose:
#   PowerShell runner for the governed teacher_builder workflow.
#
# Project context:
#   This script runs Steps 01 through 06 using a shared governed YAML config.
#   It is the command-line entry point for teacher input validation, expression
#   teacher scoring, histology teacher scoring, governed fusion, prediction-ready
#   teacher handoff construction, and final teacher QC.
#
# Scientific role:
#   The runner keeps teacher_builder reproducible by executing every governed
#   step in order with one config and one Python environment. Model thresholds,
#   treatment-prior shrinkage settings, artifact paths, and output roots live in
#   the YAML config and Python scripts, not in this runner.
#
# Documentation polish marker:
#   TEACHER_BUILDER_GOVERNED_RUNNER_PS1_DOC_POLISH_V1
#
# Important:
#   This documentation pass is intentionally non-behavioral. Comments may be
#   added, but PowerShell commands, parameters, step numbers, script paths, and
#   execution logic must remain unchanged.
# ============================================================

# ------------------------------------------------------------
# Runner parameters
# ------------------------------------------------------------
# StartAt/StopAt allow partial reruns; Config selects sample5 or full102; Python selects the interpreter.
param(
    [int]$StartAt = 1,
    [int]$StopAt = 6,
    [string]$Config = "configs\visium_teacher_builder_governed_sample5.yaml",
    [string]$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"
)


# Stop immediately if a step fails so partial teacher outputs are not mistaken for a complete run.
$ErrorActionPreference = "Stop"


# ------------------------------------------------------------
# YAML value reader
# ------------------------------------------------------------
# This helper reads scalar config values, such as output_root, through the same Python environment used by the workflow.
function Get-ConfigValue {
    param(
        [string]$ConfigPath,
        [string]$Key
    )

    $Code = @"
from pathlib import Path
import yaml

cfg_path = Path(r'''$ConfigPath''')
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

value = cfg
for part in '$Key'.split('.'):
    value = value[part]

print(value)
"@

    return (& $Python -c $Code).Trim()
}


# ------------------------------------------------------------
# Step execution helper
# ------------------------------------------------------------
# Each numbered step receives the same YAML config and is skipped when outside the requested range.
function Run-Step {
    param(
        [int]$StepNumber,
        [string]$Name,
        [string]$Script
    )


    # Respect the inclusive StartAt/StopAt range for resumable workflow execution.
    if ($StepNumber -lt $StartAt -or $StepNumber -gt $StopAt) {
        return
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Step $StepNumber`: $Name"
    Write-Host "============================================================"


    # All Python scripts are launched with the governed YAML config.
    & $Python $Script --config $Config


    # Nonzero exit codes stop the run at the failing step.
    if ($LASTEXITCODE -ne 0) {
        throw "Step $StepNumber failed: $Name"
    }
}

Write-Host ""

# ------------------------------------------------------------
# Run banner and output-root discovery
# ------------------------------------------------------------
# Print the execution context before any step runs.
Write-Host "Governed teacher_builder run"
Write-Host "Python: $Python"
Write-Host "Config: $Config"
Write-Host "Steps: $StartAt to $StopAt"


# Resolve output_root from the YAML config so final outputs and QC summaries are easy to locate.
$OutputRoot = Get-ConfigValue -ConfigPath $Config -Key "output_root"
Write-Host "Output root: $OutputRoot"


# ------------------------------------------------------------
# Governed teacher_builder step sequence
# ------------------------------------------------------------
# Step 01 validates inputs, sample availability, priors, and teacher registry metadata.
Run-Step 1 "validate_teacher_inputs" ".\scripts\01_validate_teacher_inputs.py"
# Step 02 builds expression pseudobulk teacher scores from approved expression-response models.
Run-Step 2 "build_expression_teacher" ".\scripts\02_build_expression_teacher.py"
# Step 03 delegates histology teacher scoring to the governed histology wrapper.
Run-Step 3 "build_histology_teacher" ".\scripts\03_build_histology_teacher.py"
# Step 04 fuses expression and histology teacher scores with prior-anchored shrinkage.
Run-Step 4 "fuse_teacher_tables" ".\scripts\04_fuse_teacher_tables.py"
# Step 05 joins fused teacher labels to numeric spatial features for downstream modeling.
Run-Step 5 "build_prediction_ready_teacher" ".\scripts\05_build_prediction_ready_teacher.py"
# Step 06 writes final QC summaries, checks, diagnostic figures, and QC decision.
Run-Step 6 "qc_teacher_outputs" ".\scripts\06_qc_teacher_outputs.py"

Write-Host ""
Write-Host "DONE"
Write-Host "Output root:"
Write-Host $OutputRoot


# ------------------------------------------------------------
# Final QC summary echo
# ------------------------------------------------------------
# If Step 06 produced a human-readable QC summary, print it at the end of the run.
$QcSummary = Join-Path $OutputRoot "06_teacher_qc\qc_summary.txt"
if (Test-Path $QcSummary) {
    Write-Host ""
    Write-Host "QC summary"
    Write-Host "============================================================"
    Get-Content $QcSummary
}

