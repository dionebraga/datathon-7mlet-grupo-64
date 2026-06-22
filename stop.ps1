# =============================================================================
# stop.ps1 — encerra a stack do Adaptive Offers (API 8000, MLflow 5001, BI 8503).
# Uso:  .\stop.ps1
# =============================================================================
$ports = 8000, 5000, 5001, 5050, 8503
foreach ($p in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $conns.OwningProcess | Select-Object -Unique | ForEach-Object {
            try { Stop-Process -Id $_ -Force -ErrorAction Stop; Write-Host "  porta $p (pid $_) encerrada" -ForegroundColor Green }
            catch { }
        }
    } else {
        Write-Host "  porta ${p}: nada rodando" -ForegroundColor DarkGray
    }
}
Write-Host "Stack encerrada." -ForegroundColor Magenta
