#!/usr/bin/env bash
# Instalación en Linux/macOS (desarrollo, sin Docker):  bash scripts/install.sh [--gpu] [--pyannote]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU=0; PYA=0
for a in "$@"; do [ "$a" = "--gpu" ] && GPU=1; [ "$a" = "--pyannote" ] && PYA=1; done
command -v ffmpeg >/dev/null || echo "AVISO: ffmpeg no está en el PATH (sudo apt install ffmpeg libsndfile1); se usará imageio-ffmpeg como respaldo."
cd "$ROOT/backend"
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
if [ $GPU = 1 ]; then pip install torch --index-url https://download.pytorch.org/whl/cu124; else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
pip install -r requirements-dev.txt -r requirements-ml.txt
[ $PYA = 1 ] && pip install "pyannote.audio>=3.3,<4"
if [ ! -f "$ROOT/.env" ]; then
  sed -e 's|^DATABASE_URL=.*|DATABASE_URL=sqlite:///./data/app.db|' -e 's|^TASK_MODE=.*|TASK_MODE=inline|' \
      -e 's|^\(DATA_DIR\|UPLOAD_DIR\|REPORT_DIR\|MODELS_DIR\|TEMP_DIR\|CACHE_DIR\|MODEL_CACHE_DIR\)=/data|\1=./data|' \
      -e 's|^BACKEND_URL=.*|BACKEND_URL=http://localhost:8000|' "$ROOT/.env.example" > "$ROOT/.env"
  echo ".env creado para desarrollo local (SQLite + tareas en hilo)."
fi
cd "$ROOT/frontend" && npm install --no-audit --no-fund
echo -e "\nListo.\n  Backend : cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000\n  Frontend: cd frontend && npm run dev"
