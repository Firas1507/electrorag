param(
    [ValidateSet("mock", "live")]
    [string]$Mode = "live"
)

# Run from this folder so the GUI config and package are found.
Set-Location -LiteralPath $PSScriptRoot

$projectRoot = Split-Path -Parent $PSScriptRoot
$activate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path -LiteralPath $activate) { & $activate }

$env:ELECTRORAG_GUI_MODE = $Mode

Write-Host "Starting ElectroRAG GUI in '$Mode' mode at http://127.0.0.1:8501" -ForegroundColor Cyan
python -m streamlit run gui_app.py
