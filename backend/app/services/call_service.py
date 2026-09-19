"""Lógica de negocio de llamadas: subida (streaming), borrado granular, retención."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, InvalidAudio
from app.core.logging import get_logger
from app.ml.audio import ffmpeg
from app.models import (AudioSegment, Call, CallStatus, CriticalEvent, EmotionPrediction, SatisfactionFeedback,
                        SatisfactionScore, TrainingSample, Transcription, User)
from app.services.audit import audit
from app.services.cache import cache_purge
from app.services.config_service import get_all
from app.services.storage import call_key, get_storage
from app.workers.pipeline import work_dir

log = get_logger(__name__)
CHUNK = 1024 * 1024


def create_call_from_upload(db: Session, user: User, filename: str, stream: BinaryIO, *, allow_training: bool,
                            display_name: str | None, max_size: int | None = None) -> Call:
    """Guarda el archivo en streaming (sin cargarlo en memoria), calcula SHA-256, valida con FFmpeg y crea la llamada."""
    s = get_settings()
    cfg = get_all(db, user.org_id)["audio"]
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in cfg["accepted_extensions"]:
        raise InvalidAudio("Formato no soportado. Use MP3, WAV, M4A, AAC, FLAC u OGG.", detail=f"ext={ext}")
    limit = max_size or min(s.max_file_size, cfg["max_file_size_mb"] * 1024 * 1024)
    tmp_dir = s.temp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix="." + ext, dir=tmp_dir)
    tmp = Path(tmp_name)
    h = hashlib.sha256()
    size = 0
    try:
        with open(fd, "wb") as out:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise AppError(f"El archivo supera el tamaño máximo permitido ({limit // (1024 * 1024)} MB).",
                                   status_code=413, code="file_too_large")
                h.update(chunk)
                out.write(chunk)
        if size == 0:
            raise InvalidAudio(detail="archivo vacío")
        meta = ffmpeg.probe(tmp)                                  # valida que sea audio decodificable
        if meta["duration"] > cfg["max_duration_hours"] * 3600:
            raise InvalidAudio("La grabación supera la duración máxima permitida.")
        call = Call(org_id=user.org_id, uploaded_by=user.id, filename=Path(filename).name[:490],
                    display_name=display_name, file_hash=h.hexdigest(), size_bytes=size, duration=meta["duration"],
                    sample_rate=meta["sample_rate"], channels=meta["channels"], bitrate=meta["bitrate"],
                    status=CallStatus.UPLOADED.value, allow_training=bool(allow_training), checkpoints={}, warnings=[])
        db.add(call)
        db.flush()
        call.storage_key = call_key(user.org_id, call.id, f"original.{ext}")
        get_storage().put_file(call.storage_key, tmp)
        audit(db, org_id=user.org_id, user_id=user.id, action="call.upload", entity="call", entity_id=call.id,
              details={"filename": call.filename, "size": size, "allow_training": bool(allow_training)}, commit=False)
        db.commit()
        return call
    except Exception:
        db.rollback()
        raise
    finally:
        tmp.unlink(missing_ok=True)


def delete_call_data(db: Session, call: Call, what: str, user: User | None = None) -> None:
    """what: all | transcription | results | audio. Elimina también las muestras de dataset derivadas (derecho al olvido)."""
    storage = get_storage()
    if what in ("transcription", "all"):
        db.execute(delete(Transcription).where(Transcription.call_id == call.id))
        call.checkpoints = {k: v for k, v in (call.checkpoints or {}).items() if k != "transcription"}
    if what in ("results", "all"):
        for m in (EmotionPrediction, SatisfactionScore, CriticalEvent, SatisfactionFeedback):
            db.execute(delete(m).where(m.call_id == call.id))
        call.summary = call.interaction = call.analysis_quality = None
        call.satisfaction_p1 = call.satisfaction_p2 = None
        call.checkpoints = {k: v for k, v in (call.checkpoints or {}).items()
                            if k not in ("emotion_analysis", "satisfaction", "report")}
    if what in ("audio", "all"):
        for key in (call.storage_key, call.playback_key):
            if key:
                storage.delete(key)
        shutil.rmtree(work_dir(call.id), ignore_errors=True)
        cache_purge(call.org_id, call.file_hash)
        call.storage_key = call.playback_key = None
    if what == "all":
        samples = db.scalars(select(TrainingSample).where(TrainingSample.call_id == call.id)).all()
        for sm in samples:
            if sm.audio_key:
                storage.delete(sm.audio_key)
            db.delete(sm)
        db.delete(call)
    if user:
        audit(db, org_id=call.org_id, user_id=user.id, action=f"call.delete.{what}", entity="call",
              entity_id=call.id, commit=False)
    db.commit()


def retention_days(db: Session, org_id: str) -> int:
    return int(get_all(db, org_id)["app"]["privacy"]["retention_days"] or 0)
