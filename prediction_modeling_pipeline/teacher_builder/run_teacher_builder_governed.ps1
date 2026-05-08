param(
    [int]$StartAt = 1,
    [int]$StopAt = 6,
    [string]$Config = "configs\visium_teacher_builder_governed_smoke_test.local.yaml",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

# ============================================================
# Runner: run_teacher_builder_governed.ps1
# ============================================================
#
# Purpose:
#   Run the governed teacher_builder workflow.
#
# Project context:
#   This script runs Steps 01 through 06 using one governed YAML config.
#   It is the command-line entry point for teacher input validation, expression
#   teacher scoring, histology teacher scoring, governed fusion,
#   prediction-ready teacher handoff construction, and final teacher QC.
#
# Notes for GitHub users:
#   - The default Python executable is "python", so activate your environment
#     before running or pass -Python explicitly.
#   - Paths to data, model artifacts, and output folders should be controlled
#     in the YAML config, not hardcoded in this runner.
#   - Use the sample config for smoke testing and the full config for the
#     full governed run after local paths are updated.
# ============================================================

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

function Run-Step {
    param(
        [int]$StepNumber,
        [string]$Name,
        [string]$Script
    )

    if ($StepNumber -lt $StartAt -or $StepNumber -gt $StopAt) {
        return
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Step $StepNumber`: $Name"
    Write-Host "============================================================"

    & $Python $Script --config $Config

    if ($LASTEXITCODE -ne 0) {
        throw "Step $StepNumber failed: $Name"
    }
}

Write-Host ""
Write-Host "Governed teacher_builder run"
Write-Host "Python: $Python"
Write-Host "Config: $Config"
Write-Host "Steps: $StartAt to $StopAt"

if (!(Test-Path $Config)) {
    throw "Config file not found: $Config"
}

$OutputRoot = Get-ConfigValue -ConfigPath $Config -Key "output_root"
Write-Host "Output root: $OutputRoot"

Run-Step 1 "validate_teacher_inputs" ".\scripts\01_validate_teacher_inputs.py"
Run-Step 2 "build_expression_teacher" ".\scripts\02_build_expression_teacher.py"
Run-Step 3 "build_histology_teacher" ".\scripts\03_build_histology_teacher.py"
Run-Step 4 "fuse_teacher_tables" ".\scripts\04_fuse_teacher_tables.py"
Run-Step 5 "build_prediction_ready_teacher" ".\scripts\05_build_prediction_ready_teacher.py"
Run-Step 6 "qc_teacher_outputs" ".\scripts\06_qc_teacher_outputs.py"

Write-Host ""
Write-Host "DONE"
Write-Host "Output root:"
Write-Host $OutputRoot

$QcSummary = Join-Path $OutputRoot "06_teacher_qc\qc_summary.txt"
if (Test-Path $QcSummary) {
    Write-Host ""
    Write-Host "QC summary"
    Write-Host "============================================================"
    Get-Content $QcSummary
}
