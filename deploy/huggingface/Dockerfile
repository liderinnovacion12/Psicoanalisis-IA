# Hugging Face Space (SDK Docker): interfaz Next.js + backend FastAPI + modelos, todo en UN contenedor (puerto 7860).
# El código se clona de GitHub al construir, así que este Space solo necesita este Dockerfile y el README.md de la tarjeta.
# Reconstruir (Settings → Factory rebuild) trae siempre la última versión de `main`.

# ---------- 1) compilar el frontend ----------
FROM node:22-slim AS web
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG REPO=https://github.com/liderinnovacion12/Psicoanalisis-IA.git
ARG REF=main
WORKDIR /src
RUN git clone --depth 1 --branch ${REF} ${REPO} .
WORKDIR /src/frontend
# Modo real (no demo). El navegador habla con el mismo origen; Next reenvía /api/v1/* al backend local.
ENV NEXT_PUBLIC_DEMO=0 BACKEND_URL=http://127.0.0.1:8000 NEXT_TELEMETRY_DISABLED=1
RUN npm ci --no-audit --no-fund && npm run build

# ---------- 2) imagen final: Python + modelos + Node (solo el binario) ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=web /usr/local/bin/node /usr/local/bin/node
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --from=web /src/backend ./backend
COPY --from=web /src/frontend/.next/standalone ./frontend
COPY --from=web /src/frontend/.next/static ./frontend/.next/static
COPY --from=web /src/frontend/public ./frontend/public

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r backend/requirements.txt -r backend/requirements-ml.txt

# Descarga los modelos UNA vez, al construir (así no se bajan en cada reinicio del Space).
ENV MODEL_CACHE_DIR=/app/model_cache
RUN python -c "from huggingface_hub import snapshot_download as d; d('r-f/wav2vec-english-speech-emotion-recognition', cache_dir='/app/model_cache'); d('Systran/faster-whisper-small', cache_dir='/app/model_cache/whisper')"

COPY --from=web /src/deploy/huggingface/start.sh /app/start.sh
RUN chmod +x /app/start.sh && chown -R user:user /app

USER user
ENV DATA_DIR=/home/user/data UPLOAD_DIR=/home/user/data/uploads REPORT_DIR=/home/user/data/reports \
    MODELS_DIR=/home/user/data/models TEMP_DIR=/home/user/data/tmp CACHE_DIR=/home/user/data/cache \
    DATABASE_URL=sqlite:////home/user/data/app.db TASK_MODE=inline INLINE_WORKERS=1 DEVICE=cpu \
    ALLOW_SIGNUP=false COOKIE_SECURE=true LOG_JSON=true
EXPOSE 7860
CMD ["/app/start.sh"]
