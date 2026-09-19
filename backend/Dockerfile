# Imagen única para API y workers (cambia el comando). CPU por defecto; GPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
FROM python:3.11-slim AS base
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --upgrade pip \
    && pip install torch --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt -r requirements-ml.txt
# Diarización con pyannote (requiere HF_TOKEN en ejecución). Quite esta línea si no la va a usar.
RUN pip install "pyannote.audio>=3.3,<4" || echo "pyannote.audio no se pudo instalar: se usará la diarización de respaldo"
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /data /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
