# Detached run script for Davidson dataset
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# Ensure outputs directory exists
if (-not (Test-Path outputs)) { New-Item -ItemType Directory -Path outputs | Out-Null }

Write-Output "Starting prepare_davidson.py at $(Get-Date) on $env:COMPUTERNAME"
python prepare_davidson.py

Write-Output "Starting main.py full run at $(Get-Date)"
python main.py --mode full --data_path dataset_davidson/ 2>&1 | Tee-Object -FilePath outputs/davidson_run.log

Write-Output "Completed at $(Get-Date)"
