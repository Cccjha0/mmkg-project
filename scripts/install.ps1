param(
    [switch]$SkipPython,
    [switch]$SkipFrontend,
    [switch]$SkipKgData
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $ProjectRoot "frontend"
$KgDir = Join-Path $ProjectRoot "kg"
$ProcessedDir = Join-Path $ProjectRoot "data\datasets\openbg_img\processed"
$RawDir = Join-Path $ProjectRoot "data\datasets\openbg_img\raw"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Write-Host "MMKG setup starting..." -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

Require-Command python
Require-Command npm

if (-not $SkipPython) {
    Write-Host "`n[1/3] Installing Python dependencies..." -ForegroundColor Cyan
    Push-Location $ProjectRoot
    python -m pip install -r requirements.txt
    Pop-Location
}

if (-not $SkipFrontend) {
    Write-Host "`n[2/3] Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location $FrontendDir
    npm install
    Pop-Location
}

if (-not $SkipKgData) {
    Write-Host "`n[3/3] Checking KG processed data..." -ForegroundColor Cyan
    $DataCsv = Join-Path $ProcessedDir "data.csv"
    $MetadataJson = Join-Path $ProcessedDir "metadata.json"
    $TrainTsv = Join-Path $RawDir "OpenBG-IMG_train.tsv"

    if (-not (Test-Path -LiteralPath $TrainTsv)) {
        Write-Warning "Raw OpenBG-IMG files were not found. Skipping KG data generation."
        Write-Warning "Expected: $TrainTsv"
    }
    else {
        New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

        Push-Location $KgDir
        if (-not (Test-Path -LiteralPath $DataCsv)) {
            Write-Host "Generating data.csv..."
            python convert_openbg.py
        }
        else {
            Write-Host "data.csv already exists."
        }

        if (-not (Test-Path -LiteralPath $MetadataJson)) {
            Write-Host "Generating metadata.json..."
            python generate_metadata.py
        }
        else {
            Write-Host "metadata.json already exists."
        }
        Pop-Location
    }
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Run the app with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1"
