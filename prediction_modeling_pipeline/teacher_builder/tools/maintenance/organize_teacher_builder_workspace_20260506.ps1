param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$Root = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$ArchiveRoot = Join-Path $Root "_archive_workspace_$Stamp"
$PatchDir = Join-Path $ArchiveRoot "patches_and_scripts"
$DeprecatedRunnerDir = Join-Path $ArchiveRoot "deprecated_runners"
$DeprecatedOutputDir = Join-Path $ArchiveRoot "deprecated_outputs"
$TestRunDir = Join-Path $ArchiveRoot "test_and_incomplete_runs"
$FileTreeDir = Join-Path $ArchiveRoot "filetrees_and_bundles"

$Dirs = @(
    $ArchiveRoot,
    $PatchDir,
    $DeprecatedRunnerDir,
    $DeprecatedOutputDir,
    $TestRunDir,
    $FileTreeDir
)

foreach ($d in $Dirs) {
    if ($Apply) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
}

function Move-Safe {
    param(
        [string]$Path,
        [string]$DestinationDir,
        [string]$Reason
    )

    if (!(Test-Path $Path)) {
        return
    }

    $Item = Get-Item $Path
    $Dest = Join-Path $DestinationDir $Item.Name

    [pscustomobject]@{
        Action = if ($Apply) { "MOVE" } else { "DRY_RUN_MOVE" }
        Source = $Item.FullName
        Destination = $Dest
        Reason = $Reason
    }

    if ($Apply) {
        Move-Item $Item.FullName $Dest -Force
    }
}

Write-Host ""
Write-Host "Teacher builder workspace organizer"
Write-Host "Root: $Root"
Write-Host "Archive root: $ArchiveRoot"
Write-Host "Mode: $(if ($Apply) {'APPLY'} else {'DRY RUN'})"
Write-Host ""

Set-Location $Root

# Root-level patch and utility files.
Move-Safe ".\patch_teacher_builder_governed_v2.ps1" $PatchDir "one-time governed patch script"
Move-Safe ".\audit_teacher_builder_outputs_01_06.ps1" $PatchDir "one-time audit utility"
Move-Safe ".\teacher_builder_filetree_sizes_dates_20260506_024309.txt" $FileTreeDir "file tree snapshot"
Move-Safe ".\outputs\outputs_all_code+data_combined_2026_05_06_part1.txt" $FileTreeDir "combined output bundle"

# Deprecated or superseded runners.
Move-Safe ".\run_teacher_builder_10.ps1" $DeprecatedRunnerDir "older small-run runner"
Move-Safe ".\run_teacher_builder_reactome75_102_checked.ps1" $DeprecatedRunnerDir "old reactome75 runner superseded by governed configs"
Move-Safe ".\run_teacher_builder_reactome75_102_checked.ps1.before_output_run_10_false_positive_patch" $DeprecatedRunnerDir "runner backup"
Move-Safe ".\run_teacher_builder_reactome75_102_checked.ps1.before_PC_path_patch" $DeprecatedRunnerDir "runner backup"
Move-Safe ".\run_teacher_builder_reactome75_102_checked.ps1.before_python_c_patch" $DeprecatedRunnerDir "runner backup"
Move-Safe ".\run_teacher_builder_reactome75_102_checked.ps1.before_stale_check_exact_patch" $DeprecatedRunnerDir "runner backup"

# Deprecated root-level old outputs from before named runs.
foreach ($Folder in @(
    ".\outputs\01_input_validation",
    ".\outputs\02_expression_teacher",
    ".\outputs\03_histology_teacher",
    ".\outputs\04_fused_teacher",
    ".\outputs\05_prediction_ready_teacher",
    ".\outputs\06_teacher_qc",
    ".\outputs\output_run_3",
    ".\outputs\output_run_102visium_04_30_26"
)) {
    Move-Safe $Folder $DeprecatedOutputDir "old or test teacher_builder output"
}

# Keep output_run_102_reactome75_20260503 for comparison for now.
# Keep teacher_builder_governed_sample5_20260505_072436 as the passing smoke test for now.
# Keep _histology_v2_compat because active histology scorer uses it.
# Keep scripts\_backup_governed_20260505_072355 because active step 03 calls it.

foreach ($Folder in Get-ChildItem ".\outputs" -Directory -ErrorAction SilentlyContinue) {
    if ($Folder.Name -like "teacher_builder_governed_sample5_*_incomplete_before_rerun_*") {
        Move-Safe $Folder.FullName $TestRunDir "incomplete governed sample5 attempt"
    }
}

# Script backup clutter, but do not touch the active dependency backup.
foreach ($File in Get-ChildItem ".\scripts" -File -ErrorAction SilentlyContinue) {
    if ($File.Name -like "*.before_*") {
        Move-Safe $File.FullName $PatchDir "script patch backup"
    }
}

# Move the recursive wrapper backup, but keep the active real-original backup.
Move-Safe ".\scripts\_backup_governed_20260505_072436" $PatchDir "recursive wrapper backup, not active"

Write-Host ""
if ($Apply) {
    Write-Host "Applied organization."
    Write-Host "Archive created:"
    Write-Host $ArchiveRoot
} else {
    Write-Host "Dry run complete. Nothing was moved."
    Write-Host "Review the plan above. To apply:"
    Write-Host ".\organize_teacher_builder_workspace_20260506.ps1 -Apply"
}
