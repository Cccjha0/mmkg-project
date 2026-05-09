param(
    [switch]$SkipFastApi,
    [switch]$SkipKgService,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$ProcessedDir = Join-Path $ProjectRoot "data\datasets\openbg_img\processed"
$ProductionModelsDir = Join-Path $ProjectRoot "ml\artifacts\production_models"

function Start-ServiceWindow($Title, $WorkingDirectory, $Command) {
    $EscapedTitle = $Title.Replace("'", "''")
    $EscapedDir = $WorkingDirectory.Replace("'", "''")
    $EscapedCommand = $Command.Replace("'", "''")
    $Script = "`$host.UI.RawUI.WindowTitle = '$EscapedTitle'; Set-Location -LiteralPath '$EscapedDir'; $EscapedCommand"

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $Script
    )
}

Write-Host "Starting MMKG dev services..." -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

$DataCsv = Join-Path $ProcessedDir "data.csv"
$MetadataJson = Join-Path $ProcessedDir "metadata.json"
if (-not (Test-Path -LiteralPath $DataCsv) -or -not (Test-Path -LiteralPath $MetadataJson)) {
    Write-Warning "KG processed data is missing. The KG page may be blank."
    Write-Warning "Run: powershell -ExecutionPolicy Bypass -File scripts\install.ps1"
}

if (-not (Test-Path -LiteralPath $ProductionModelsDir)) {
    Write-Warning "Production models directory is missing. Attribute Completion may run in metadata-only mode."
    Write-Warning "Expected: $ProductionModelsDir"
}

if (-not $SkipFastApi) {
    Start-ServiceWindow `
        -Title "MMKG FastAPI :8000" `
        -WorkingDirectory $BackendDir `
        -Command "uvicorn app.main:app --reload --port 8000"
}

if (-not $SkipKgService) {
    Start-ServiceWindow `
        -Title "MMKG KG Flask :5000" `
        -WorkingDirectory $BackendDir `
        -Command "python flask_app.py"
}

if (-not $SkipFrontend) {
    Start-ServiceWindow `
        -Title "MMKG Frontend :3000" `
        -WorkingDirectory $FrontendDir `
        -Command "npm run dev"
}

Write-Host "`nStarted requested services in separate PowerShell windows." -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000"
Write-Host "FastAPI:  http://127.0.0.1:8000"
Write-Host "KG Flask: http://127.0.0.1:5000"
