param(
    [string]$Root = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\model_training",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

if ($Python -eq "") {
    $Candidate = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"
    if (Test-Path $Candidate) { $Python = $Candidate } else { $Python = "python" }
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
    & $Python -m py_compile @PyFiles
    if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }
    & $Python -c "import yaml, pathlib; [yaml.safe_load(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['expression_response_model_v2/configs/expression_response_model_v2.yaml','histology_response_model_v2/configs/histology_response_model_v2.yaml']]; print('YAML parse: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "YAML parse failed" }
    & $Python -c "import pandas as pd, pathlib; paths=['expression_response_model_v2/outputs/deployable_CH1/model_index.tsv','expression_response_model_v2/outputs/deployable_CH1/model_index_approved.tsv','histology_response_model_v2/outputs/histology_v2/07_models/model_comparison.tsv','histology_response_model_v2/outputs/histology_v2/09_audit/histology_model_index.tsv']; [pd.read_csv(p, sep='\t', nrows=5) for p in paths if pathlib.Path(p).exists()]; print('Key TSV read check: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "Key TSV read failed" }
} finally {
    Pop-Location
}

Get-ChildItem -LiteralPath (Join-Path $Root "expression_response_model_v2\scripts"), (Join-Path $Root "histology_response_model_v2\scripts") -Recurse -Force |
    Where-Object { $_.Name -eq "__pycache__" -or $_.Extension -in @(".pyc", ".pyo") } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }

Write-Host "py_compile: PASS"
Write-Host "Smoke test: PASS"
