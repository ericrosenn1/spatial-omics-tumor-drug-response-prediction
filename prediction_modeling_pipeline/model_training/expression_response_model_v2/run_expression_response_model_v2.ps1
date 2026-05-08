# ============================================================
# Runner: run_expression_response_model_v2.ps1
# ============================================================
#
# Purpose:
#   PowerShell runner for the numbered expression_response_model_v2 workflow.
#
# Project context:
#   This script runs Step 00 through Step 05 using a shared YAML config. It is
#   the command-line entry point for treatment ontology construction, input
#   validation, canonical training-table creation, deployable model training,
#   deployable model audit, and optional Visium expression-teacher scoring.
#
# Scientific role:
#   The runner keeps the expression-response model workflow reproducible by
#   executing the numbered scripts in order with one config and one Python
#   environment. It does not define model thresholds itself; those live in the
#   YAML config and Python step scripts.
#
# Documentation polish marker:
#   EXPRESSION_MODEL_V2_RUNNER_PS1_DOC_POLISH_V1
#
# Important:
#   This documentation pass is intentionally non-behavioral. Comments may be
#   added, but PowerShell commands, parameters, step numbers, script paths, and
#   execution logic must remain unchanged.
# ============================================================
# ------------------------------------------------------------
# Runner parameters
# ------------------------------------------------------------
# Config selects the YAML file; StartAt/StopAt allow partial reruns.
param(
    [string]$Config = ".\configs\expression_response_model_v2.yaml",
    [int]$StartAt = 0,
    [int]$StopAt = 4,
    [string]$Python = ""
)


# Stop immediately if a step fails so partial runs are obvious.
$ErrorActionPreference = "Stop"


# Resolve the runner location and execute relative to the model folder.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here


# Prefer the project virtual environment; fall back to system python if needed.
if ($Python -eq "") {
    $ProjectRoot = ""
    $Candidate = ""
    if ($Candidate -ne "" -and (Test-Path $Candidate)) {
        $Python = $Candidate
    } else {
        $Python = "python"
    }
}


# Fail early if the requested config cannot be found.
if (!(Test-Path $Config)) {
    throw "Config not found: $Config"
}


# Numbered pipeline contract.
# Each entry maps a step number to its human-readable name and Python script.
$Steps = @(
    @{N=0; Name="build treatment ontology"; Script="scripts\00_build_treatment_ontology.py"},
    @{N=1; Name="validate inputs"; Script="scripts\01_validate_inputs.py"},
    @{N=2; Name="build canonical training table"; Script="scripts\02_build_canonical_training_table.py"},
    @{N=3; Name="train deployable calibrated models"; Script="scripts\03_train_deployable_models.py"},
    @{N=4; Name="audit deployable models"; Script="scripts\04_audit_deployable_models.py"},
    @{N=5; Name="score Visium pseudobulk with deployable models"; Script="scripts\05_score_visium_expression_teacher.py"}
)

Write-Host ""
Write-Host "Expression response model v2 runner"
Write-Host "Config: $Config"
Write-Host "Python: $Python"
Write-Host "Steps: $StartAt to $StopAt"
Write-Host ""


# Execute only the requested inclusive step range.
foreach ($Step in $Steps) {
    if ($Step.N -lt $StartAt -or $Step.N -gt $StopAt) {
        continue
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Step $($Step.N): $($Step.Name)"
    Write-Host "============================================================"


# Resolve each step script relative to this runner.
    $ScriptPath = Join-Path $Here $Step.Script
    if (!(Test-Path $ScriptPath)) {
        throw "Missing step script: $ScriptPath"
    }


# Every Python step receives the same YAML config.
    & $Python $ScriptPath --config $Config


# Nonzero exit codes stop the pipeline and preserve the failing step number.
    if ($LASTEXITCODE -ne 0) {
        throw "Step $($Step.N) failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "DONE expression_response_model_v2"


