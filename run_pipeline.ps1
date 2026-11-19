# PowerShell Single Execution Script for Stage 4 Capstone Pipeline
Write-Host "Starting Stage 4 Capstone Pipeline..." -ForegroundColor Cyan
$pythonPath = "C:\Users\USER\anaconda3\envs\tf-env\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = "python"
}

& $pythonPath scripts/run_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Pipeline completed successfully!" -ForegroundColor Green

