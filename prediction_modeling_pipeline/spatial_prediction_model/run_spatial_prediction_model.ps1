<#
.SYNOPSIS
    Run spatial_prediction_model pipeline steps.

.EXAMPLES
    .\run_spatial_prediction_model.ps1
    .\run_spatial_prediction_model.ps1 -Steps "01,02,03"
    .\run_spatial_prediction_model.ps1 -Config "configs\spatial_prediction_model.yaml" -Steps "01,02,03,05,06,07"
#>

param(
    [string]$Config = "configs\spatial_prediction_model.yaml",
    [string]$Steps = "01,02,03,04,05,06,07"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

Write-Host "spatial_prediction_model"
Write-Host "Root:   $Here"
Write-Host "Config: $Config"
Write-Host "Steps:  $Steps"
Write-Host ""

python ".\run_spatial_prediction_model.py" --config $Config --steps $Steps