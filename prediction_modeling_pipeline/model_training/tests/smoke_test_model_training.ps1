param(
    [string]$Root = "" ,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

if ($Root -eq "") {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if ($Python -eq "") {
    $Python = "python"
}

$PyFiles = @(
    "expression_response_model_v2\scripts\expression_model_v2_lib.py",
    "expression_response_model_v2\scripts\00_build_treatment_ontology.py",
    "expression_response_model_v2\scripts\01_validate_inputs.py",
    "expression_response_model_v2\scripts\02_build_canonical_training_table.py",
    "expression_response_model_v2\scripts\03_train_deployable_models.py",
    "expression_response_model_v2\scripts\04_audit_deployable_models.py",
    "expression_response_model_v2\scripts\05_score_visium_expression_teacher.py",
    "histology_response_model_v2\scripts\histology_model_v2_lib.py",
    "histology_response_model_v2\scripts\00_build_treatment_ontology.py",
    "histology_response_model_v2\scripts\01_validate_inputs.py",
    "histology_response_model_v2\scripts\02_build_case_label_table.py",
    "histology_response_model_v2\scripts\03_build_slide_manifest.py",
    "histology_response_model_v2\scripts\04_tile_slides.py",
    "histology_response_model_v2\scripts\05_build_tile_training_table.py",
    "histology_response_model_v2\scripts\06_build_patient_split.py",
    "histology_response_model_v2\scripts\07_train_baselines_and_conditioned_model.py",
    "histology_response_model_v2\scripts\08_run_control_inference.py",
    "histology_response_model_v2\scripts\09_audit_histology_model.py"
)

Write-Host ""
Write-Host "model_training smoke test"
Write-Host "Root: $Root"
Write-Host "Python: $Python"

foreach ($Rel in $PyFiles) {
    $Path = Join-Path $Root $Rel
    if (!(Test-Path $Path)) { throw "Missing active Python file: $Path" }
}

Push-Location $Root
try {
    & $Python -c "import pathlib, sys; [compile(pathlib.Path(p).read_text(encoding='utf-8-sig'), p, 'exec') for p in sys.argv[1:]]; print('Python syntax: PASS')" @PyFiles
    if ($LASTEXITCODE -ne 0) { throw "Python syntax check failed" }
    & $Python -c "import yaml, pathlib; [yaml.safe_load(pathlib.Path(p).read_text(encoding='utf-8-sig')) for p in ['expression_response_model_v2/configs/expression_response_model_v2.example.yaml','histology_response_model_v2/configs/histology_response_model_v2.example.yaml']]; print('Example YAML parse: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "YAML parse failed" }
} finally {
    Pop-Location
}

Write-Host "Source/configuration checks: PASS (no training or numerical validation performed)"



