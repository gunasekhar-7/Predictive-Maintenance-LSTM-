<# 
.SYNOPSIS
    Downloads and extracts the NASA C-MAPSS Turbofan Engine Degradation dataset.
.DESCRIPTION
    Fetches the official PCoE zip from phm-datasets.s3.amazonaws.com, extracts it,
    and stages the FD001 (and FD002-004) files into data/raw/ for the pipeline.
    Requires internet access. No API key needed.
.NOTES
    Run from the project root: .\scripts\download_cmapss.ps1
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RawDir      = Join-Path (Join-Path $ProjectRoot "data") "raw"
$ZipUrl      = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
$ZipPath     = Join-Path $env:TEMP "cmapss_turbofan.zip"
$ExtractDir  = Join-Path $env:TEMP "cmapss_extract"

Write-Host "=== NASA C-MAPSS Turbofan Dataset Downloader ===" -ForegroundColor Cyan
Write-Host "Project root : $ProjectRoot"
Write-Host "Raw data dir : $RawDir"
Write-Host "Source URL   : $ZipUrl"
Write-Host ""

# Ensure raw dir exists
if (-not (Test-Path -LiteralPath $RawDir)) {
    New-Item -ItemType Directory -Path $RawDir -Force | Out-Null
    Write-Host "Created $RawDir"
}

# Download
Write-Host "Downloading dataset (~40 MB)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath
Write-Host "Downloaded to $ZipPath ($(Get-Item $ZipPath).Length bytes)" -ForegroundColor Green

# Extract
if (Test-Path -LiteralPath $ExtractDir) { Remove-Item -LiteralPath $ExtractDir -Recurse -Force }
New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null
Write-Host "Extracting archive..." -ForegroundColor Yellow
Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractDir -Force

# Find extracted files (zip may contain a top-level folder)
$candidates = Get-ChildItem -LiteralPath $ExtractDir -Recurse -Include @("train_FD*.txt","test_FD*.txt","RUL_FD*.txt")
if ($candidates.Count -eq 0) {
    Write-Error "No CMAPSS txt files found in archive. Inspect $ExtractDir"
}

# Copy into data/raw
$copied = 0
foreach ($f in $candidates) {
    $dest = Join-Path $RawDir $f.Name
    Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
    Write-Host "  staged $($f.Name) -> $dest"
    $copied++
}

# Cleanup temp
Remove-Item -LiteralPath $ZipPath -Force
Remove-Item -LiteralPath $ExtractDir -Recurse -Force

Write-Host ""
Write-Host "Staged $copied file(s) in $RawDir" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  python -m src.data.loader --config configs/config.yaml"
Write-Host "  python -m src.models.baseline_cli --config configs/config.yaml"
Write-Host "  python -m src.training.trainer_cli --config configs/config.yaml"
Write-Host "  python -m src.evaluation.eval_cli --config configs/config.yaml"
