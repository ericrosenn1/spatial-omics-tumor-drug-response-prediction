param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$Root = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder"
$Outputs = Join-Path $Root "outputs"
$FinalRun = Join-Path $Outputs "teacher_builder_governed_full102_20260505_072436"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OldRuns = Join-Path $Outputs "_old_runs_deprecated_$Stamp"
$TestRuns = Join-Path $OldRuns "test_runs"
$DeprecatedRuns = Join-Path $OldRuns "deprecated_runs"
$PreflightRuns = Join-Path $OldRuns "preflight_and_logs"

$FinalStepFolders = @(
    "01_input_validation",
    "02_expression_teacher",
    "03_histology_teacher",
    "04_fused_teacher",
    "05_prediction_ready_teacher",
    "06_teacher_qc",
    "07_treatment_caution_list"
)

function Show-Action {
    param(
        [string]$Action,
        [string]$Source,
        [string]$Destination,
        [string]$Reason
    )

    [pscustomobject]@{
        Action = if ($Apply) { $Action } else { "DRY_RUN_$Action" }
        Source = $Source
        Destination = $Destination
        Reason = $Reason
    }
}

function Ensure-Dir {
    param([string]$Path)

    if ($Apply) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Move-Safe {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Reason
    )

    if (!(Test-Path $Source)) {
        return
    }

    Show-Action "MOVE" $Source $Destination $Reason

    if ($Apply) {
        $Parent = Split-Path $Destination -Parent
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        Move-Item $Source $Destination -Force
    }
}

function Copy-Safe {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Reason
    )

    if (!(Test-Path $Source)) {
        return
    }

    Show-Action "COPY" $Source $Destination $Reason

    if ($Apply) {
        $Parent = Split-Path $Destination -Parent
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        Copy-Item $Source $Destination -Recurse -Force
    }
}

Write-Host ""
Write-Host "Promote final governed teacher_builder outputs"
Write-Host "Root: $Root"
Write-Host "Outputs: $Outputs"
Write-Host "Final run: $FinalRun"
Write-Host "Old runs folder: $OldRuns"
Write-Host "Mode: $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host ""

if (!(Test-Path $FinalRun)) {
    throw "Final run folder does not exist: $FinalRun"
}

$Required = @(
    "05_prediction_ready_teacher\prediction_ready_training_table.tsv",
    "05_prediction_ready_teacher\visium_fused_teacher_table.tsv",
    "06_teacher_qc\qc_summary.txt",
    "06_teacher_qc\teacher_qc_decision.txt",
    "07_treatment_caution_list\saturated_treatment_caution_list.tsv"
)

foreach ($Rel in $Required) {
    $Path = Join-Path $FinalRun $Rel
    if (!(Test-Path $Path)) {
        throw "Required final output is missing: $Path"
    }
}

Ensure-Dir $OldRuns
Ensure-Dir $TestRuns
Ensure-Dir $DeprecatedRuns
Ensure-Dir $PreflightRuns

# Archive any existing canonical step folders before promoting the final run.
foreach ($Folder in $FinalStepFolders) {
    $Existing = Join-Path $Outputs $Folder
    if (Test-Path $Existing) {
        $Dest = Join-Path $DeprecatedRuns $Folder
        Move-Safe $Existing $Dest "archive existing root-level output before promotion"
    }
}

# Promote final run step folders to the main outputs folder.
foreach ($Folder in $FinalStepFolders) {
    $Src = Join-Path $FinalRun $Folder
    $Dst = Join-Path $Outputs $Folder

    if (!(Test-Path $Src)) {
        throw "Final run is missing expected folder: $Src"
    }

    Move-Safe $Src $Dst "promote governed full102 final output to canonical outputs root"
}

# Move old or test run folders away from the main outputs view.
$MoveToTest = @(
    "teacher_builder_governed_sample5_20260505_072436"
)

foreach ($Name in $MoveToTest) {
    $Src = Join-Path $Outputs $Name
    $Dst = Join-Path $TestRuns $Name
    Move-Safe $Src $Dst "test or smoke-test output"
}

$MoveToDeprecated = @(
    "output_run_102_reactome75_20260503"
)

foreach ($Name in $MoveToDeprecated) {
    $Src = Join-Path $Outputs $Name
    $Dst = Join-Path $DeprecatedRuns $Name
    Move-Safe $Src $Dst "old teacher_builder run superseded by governed full102"
}

$MoveToPreflight = @(
    "_preflight_reactome75_102",
    "pipeline_run_logs",
    "_archive_spatial_pipeline_expression_teacher_2026_04_28"
)

foreach ($Name in $MoveToPreflight) {
    $Src = Join-Path $Outputs $Name
    $Dst = Join-Path $PreflightRuns $Name
    Move-Safe $Src $Dst "preflight, logs, or old archive"
}

# Keep _histology_v2_compat in outputs because current histology wrapper/configs may still depend on it.

# Move now-empty final run shell into old runs if it still exists.
if (Test-Path $FinalRun) {
    $Remaining = Get-ChildItem $FinalRun -Force -ErrorAction SilentlyContinue
    $Dest = Join-Path $OldRuns "promoted_final_run_folder_shell_teacher_builder_governed_full102_20260505_072436"
    Move-Safe $FinalRun $Dest "final run folder after step folders were promoted"
}

$ManifestPath = Join-Path $Outputs "FINAL_OUTPUT_README_20260506.txt"

$Manifest = @(
    "Teacher builder canonical final outputs"
    ""
    "Canonical output root:"
    $Outputs
    ""
    "Promoted final run:"
    "teacher_builder_governed_full102_20260505_072436"
    ""
    "Canonical folders now expected at outputs root:"
    ($FinalStepFolders -join "`n")
    ""
    "Important downstream files:"
    "outputs\05_prediction_ready_teacher\prediction_ready_training_table.tsv"
    "outputs\05_prediction_ready_teacher\visium_fused_teacher_table.tsv"
    "outputs\05_prediction_ready_teacher\model_input_numeric.csv"
    "outputs\05_prediction_ready_teacher\feature_manifest.csv"
    "outputs\06_teacher_qc\qc_summary.txt"
    "outputs\06_teacher_qc\teacher_qc_decision.txt"
    "outputs\07_treatment_caution_list\saturated_treatment_caution_list.tsv"
    ""
    "Old runs and test runs archived under:"
    $OldRuns
    ""
    "Note:"
    "_histology_v2_compat was intentionally left in outputs because histology scoring compatibility files may still be needed for reruns."
)

Show-Action "WRITE" $ManifestPath $ManifestPath "write canonical output readme"

if ($Apply) {
    $Manifest | Set-Content $ManifestPath -Encoding UTF8
}

Write-Host ""
if ($Apply) {
    Write-Host "Promotion applied."
    Write-Host "Canonical final outputs are now under:"
    Write-Host $Outputs
    Write-Host ""
    Write-Host "README:"
    Write-Host $ManifestPath
} else {
    Write-Host "Dry run complete. Nothing moved."
    Write-Host "To apply:"
    Write-Host ".\promote_final_teacher_builder_outputs_20260506.ps1 -Apply"
}
