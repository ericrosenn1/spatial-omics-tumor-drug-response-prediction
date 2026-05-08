<#
histology_response_model_v2 runner documentation

Purpose:
    Sequentially run the canonical numbered histology response model v2 scripts.

Usage examples:
    .\run_histology_response_model_v2.ps1 -StartAt 0 -StopAt 9
    .\run_histology_response_model_v2.ps1 -StartAt 7 -StopAt 9

Notes:
    - This runner is intentionally thin: each numbered Python script owns its
      step-specific logic.
    - StartAt and StopAt allow safe resumption from completed intermediate outputs.
    - Documentation comments should not change command order, default values,
      script paths, or argument behavior.
#>

# Command-line parameters define config path, step range, and optional Python executable.
param(
    [string]$Config = "configs\histology_response_model_v2.yaml",
    [int]$StartAt = 0,
    [int]$StopAt = 9,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

if ($Python -eq "") {
    $ProjectRoot = ""
    $Candidate = ""
    if ($Candidate -ne "" -and (Test-Path $Candidate)) {
        $Python = $Candidate
    } else {
        $Python = "python"
    }
}

# Canonical production workflow: steps 00 through 09 run in dependency order.
$Steps = @(
    @{N=0; Name="build_treatment_ontology"; Script="scripts\00_build_treatment_ontology.py"},
    @{N=1; Name="validate_inputs"; Script="scripts\01_validate_inputs.py"},
    @{N=2; Name="build_case_label_table"; Script="scripts\02_build_case_label_table.py"},
    @{N=3; Name="build_slide_manifest"; Script="scripts\03_build_slide_manifest.py"},
    @{N=4; Name="tile_slides"; Script="scripts\04_tile_slides.py"},
    @{N=5; Name="build_tile_training_table"; Script="scripts\05_build_tile_training_table.py"},
    @{N=6; Name="build_patient_split"; Script="scripts\06_build_patient_split.py"},
    @{N=7; Name="train_models"; Script="scripts\07_train_baselines_and_conditioned_model.py"},
    @{N=8; Name="run_controls"; Script="scripts\08_run_control_inference.py"},
    @{N=9; Name="audit_model"; Script="scripts\09_audit_histology_model.py"}
)

Write-Host "Python: $Python"
Write-Host "Config: $Config"
Write-Host "Running steps $StartAt to $StopAt"

# Execute only the requested inclusive step range so interrupted runs can resume.
foreach ($Step in $Steps) {
    if ($Step.N -lt $StartAt -or $Step.N -gt $StopAt) {
        continue
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Step $($Step.N): $($Step.Name)"
    Write-Host "============================================================"

    & $Python $Step.Script --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Step $($Step.N) failed: $($Step.Name)"
    }
}

Write-Host ""
Write-Host "DONE"


