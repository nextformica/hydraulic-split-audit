<#
.SYNOPSIS
  Rebuild the frozen hydraulic data products and estimator checks in a clean output directory.

.DESCRIPTION
  This is a reproducibility aid. It never edits the LaTeX/Markdown manuscript. Existing
  frozen files are preserved; outputs are written below results/repro_run_<timestamp>.
  Run from any directory: the project root is resolved from this script's parent directory.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$DataZip,
  [string]$Python = "python",
  [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$ZipPath = (Resolve-Path $DataZip).Path
Set-Location $Root
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Out = Join-Path $Root ("results\repro_run_" + $stamp)
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Invoke-Step([string]$Label, [string[]]$CommandArgs) {
  Write-Host ("[repro] " + $Label)
  & $Python @CommandArgs
  if ($LASTEXITCODE -ne 0) { throw ("step failed: " + $Label + " (exit " + $LASTEXITCODE + ")") }
}

Invoke-Step "archive audit" @("src\audit_hydraulic_uci.py", "--zip", $ZipPath, "--json", (Join-Path $Out "hydraulic_uci_full_audit.json"), "--csv", (Join-Path $Out "hydraulic_uci_signal_audit.csv"))
Invoke-Step "registry" @("src\build_hydraulic_registry.py", "--zip", $ZipPath, "--cycles", (Join-Path $Out "hydraulic_uci_cycles.csv"), "--blocks", (Join-Path $Out "hydraulic_uci_blocks.csv"), "--summary", (Join-Path $Out "hydraulic_uci_registry_summary.json"))
Invoke-Step "feature extraction" @("src\extract_hydraulic_features.py", "--zip", $ZipPath, "--registry", (Join-Path $Out "hydraulic_uci_cycles.csv"), "--output", (Join-Path $Out "hydraulic_uci_cycle_features.csv"))

if (-not $SkipModels) {
  Invoke-Step "classification" @("src\evaluate_hydraulic_models.py", "--features", (Join-Path $Out "hydraulic_uci_cycle_features.csv"), "--task", "accumulator", "--sensor-set", "pressure_power", "--grouping", "state4", "--output", (Join-Path $Out "hydraulic_accumulator_primary_state4.json"))
  Invoke-Step "ordered regression" @("src\evaluate_hydraulic_ordinal.py", "--features", (Join-Path $Out "hydraulic_uci_cycle_features.csv"), "--sensor-set", "pressure_power", "--splitter", "stratified", "--output", (Join-Path $Out "hydraulic_accumulator_ordinal_primary.json"))
  Invoke-Step "paired estimator audit" @("src\compare_hydraulic_estimators_fair.py", "--features", (Join-Path $Out "hydraulic_uci_cycle_features.csv"), "--output", (Join-Path $Out "repro_20261016_estimator_comparison_fair.json"), "--selected-oof", (Join-Path $Out "repro_20261016_estimator_comparison_fair_oof.csv"), "--fixed-oof", (Join-Path $Out "repro_20261016_estimator_comparison_fair_fixed_oof.csv"))
}

Write-Host ("[repro] complete: " + $Out)
Write-Host ("[repro] external data SHA-256: " + (Get-FileHash $ZipPath -Algorithm SHA256).Hash)
