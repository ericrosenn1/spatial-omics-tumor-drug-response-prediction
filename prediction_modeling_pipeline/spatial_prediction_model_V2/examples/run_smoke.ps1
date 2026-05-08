Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$HandoffRoot = "<PATH_TO_TEACHER_BUILDER_HANDOFF>"
python "$RepoRoot\scripts\00_run_spatial_prediction_model_v2.py" --mode smoke --handoff-root "$HandoffRoot" --max-workers 0 --open-output
