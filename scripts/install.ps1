# Instalación en Windows (desarrollo, sin Docker):  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Gpu]
param([switch]$Gpu, [switch]$Pyannote)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location "$root\backend"
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
$idx = if ($Gpu) { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
& $py -m pip install torch --index-url $idx
& $py -m pip install -r requirements-dev.txt -r requirements-ml.txt
if ($Pyannote) { & $py -m pip install "pyannote.audio>=3.3,<4" }
Write-Host "FFmpeg: se usa el del PATH o el de imageio-ffmpeg (ya instalado)."
if (-not (Test-Path "$root\.env")) {
  Copy-Item "$root\.env.example" "$root\.env"
  (Get-Content "$root\.env") -replace "^DATABASE_URL=.*", "DATABASE_URL=sqlite:///./data/app.db" `
     -replace "^TASK_MODE=.*", "TASK_MODE=inline" -replace "^DATA_DIR=.*", "DATA_DIR=./data" -replace "^UPLOAD_DIR=.*", "UPLOAD_DIR=./data/uploads" `
     -replace "^REPORT_DIR=.*", "REPORT_DIR=./data/reports" -replace "^MODELS_DIR=.*", "MODELS_DIR=./data/models" -replace "^TEMP_DIR=.*", "TEMP_DIR=./data/tmp" `
     -replace "^CACHE_DIR=.*", "CACHE_DIR=./data/cache" -replace "^MODEL_CACHE_DIR=.*", "MODEL_CACHE_DIR=./data/model_cache" `
     -replace "^BACKEND_URL=.*", "BACKEND_URL=http://localhost:8000" | Set-Content "$root\.env"
  Write-Host ".env creado para desarrollo local (SQLite + tareas en hilo). Edite SECRET_KEY antes de producción."
}
Set-Location "$root\frontend"
npm install --no-audit --no-fund
Write-Host "`nListo. Ejecute:`n  Backend : cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`n  Frontend: cd frontend; npm run dev`n  Modelo  : cd backend; .\.venv\Scripts\python.exe scripts\test_emotion_model.py"
