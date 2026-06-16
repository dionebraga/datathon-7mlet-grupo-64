# =============================================================================
# start.ps1 — sobe a stack completa do Adaptive Offers em janelas separadas:
#   - API REST + Swagger  -> http://localhost:8000/docs
#   - MLflow (experimentos)-> http://localhost:5000
#   - Dashboard BI         -> http://localhost:8501
#
# Uso (PowerShell, na pasta do projeto):
#   .\start.ps1
# Se bloquear por política de execução, rode uma vez:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Para encerrar tudo:  .\stop.ps1   (ou feche as janelas / CTRL+C em cada uma)
# =============================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# Ativa o .venv automaticamente se existir (senão usa o Python/entrypoints globais).
$activate = Join-Path $root ".venv\Scripts\Activate.ps1"
$hasVenv = Test-Path $activate
if (-not $hasVenv) {
    Write-Host "[aviso] .venv nao encontrado — usando o Python global." -ForegroundColor Yellow
}

function Start-Service-Window([string]$title, [string]$cmd) {
    $prefix = if ($hasVenv) { "& '$activate'; " } else { "" }
    $full = "$prefix Set-Location '$root'; `$Host.UI.RawUI.WindowTitle = '$title'; " +
            "Write-Host '>> $title' -ForegroundColor Cyan; $cmd"
    # -EncodedCommand evita qualquer problema de aspas/escapes no caminho.
    $enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($full))
    Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $enc | Out-Null
}

Write-Host ""
Write-Host "  Adaptive Offers Platform — subindo a stack..." -ForegroundColor Magenta
Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray

Start-Service-Window "Adaptive Offers - API (8000)"       "adaptive-offers serve"
Write-Host "  [1/3] API REST       -> http://localhost:8000/docs" -ForegroundColor Green

Start-Service-Window "Adaptive Offers - MLflow (5000)"     "`$env:MLFLOW_ALLOW_FILE_STORE='true'; mlflow ui --port 5000"
Write-Host "  [2/3] MLflow         -> http://localhost:5000" -ForegroundColor Green

Start-Service-Window "Adaptive Offers - Dashboard (8501)"  "streamlit run dashboard\app.py"
Write-Host "  [3/3] Dashboard BI   -> http://localhost:8501" -ForegroundColor Green

Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  Abrindo o dashboard no navegador em alguns segundos..." -ForegroundColor DarkGray

# Dá tempo do Streamlit/MLflow iniciarem antes de abrir o navegador.
Start-Sleep -Seconds 8
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "  Pronto! Cada servico esta em sua propria janela." -ForegroundColor Magenta
Write-Host "  Para encerrar tudo:  .\stop.ps1" -ForegroundColor Yellow
Write-Host ""
