# Starts API and dashboard in separate PowerShell processes from the repository root.
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; & '$python' -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; & '$python' -m streamlit run dashboard/app.py"
Write-Output 'API: http://127.0.0.1:8000  Dashboard: http://127.0.0.1:8501'
