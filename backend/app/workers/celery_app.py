"""Celery: una cola por etapa (escalado independiente) + cola de entrenamiento + tareas periódicas."""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings
from app.core.logging import setup_logging

s = get_settings()
setup_logging(s.log_level, s.log_json, s.data_dir / "logs")

celery_app = Celery("call_analyzer", broker=s.redis_url, backend=s.redis_url, include=["app.workers.tasks"])

STAGE_QUEUE = {
    "audio_processing": "audio", "diarization": "diarization", "transcription": "transcription",
    "emotion_analysis": "emotion", "satisfaction": "satisfaction", "report": "satisfaction",
}

celery_app.conf.update(
    task_acks_late=True,                 # si un worker cae, la tarea se reentrega; las etapas son idempotentes
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,        # tareas largas: no acaparar
    task_track_started=True,
    task_time_limit=60 * 60 * 12,
    broker_transport_options={"visibility_timeout": 60 * 60 * 14},
    task_default_queue="audio",
    timezone="UTC",
    beat_schedule={
        "purge-expired-calls": {"task": "maintenance.purge_expired", "schedule": crontab(hour=3, minute=15)},
    },
)
