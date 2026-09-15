Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$HandoffRoot = "<PATH_TO_TEACHER_BUILDER_HANDOFF>"
python "$RepoRoot\scripts\00_run_spatial_prediction_model_v2.py" --mode full --handoff-root "$HandoffRoot" --max-workers 2 --full-step09-n-shuffles 1000 --full-step09-n-repeats 5 --open-output
if ($LASTEXITCODE -ne 0) { throw "Spatial modeling failed: $LASTEXITCODE" }
