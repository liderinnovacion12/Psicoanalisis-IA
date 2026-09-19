"""Endpoints de llamadas: subida, listado, detalle, estado, hablantes, transcripción, emociones,
satisfacción, eventos, audio con Range, borrado y reanudación."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import AnalystUser, AnyUser, client_ip, get_call_or_404
from app.core.errors import Conflict, NotFound
from app.database.base import get_db
from app.models import (Call, CallStatus, CriticalEvent, EmotionPrediction, SatisfactionFeedback, SatisfactionScore,
                        Speaker, Transcription, User)
from app.schemas.common import CallPatch, FeedbackIn, Msg, SpeakerPatch
from app.services import serializers as ser
from app.services.audit import audit
from app.services.call_service import create_call_from_upload, delete_call_data
from app.services.config_service import get_all
from app.services.storage import get_storage
from app.workers.dispatch import enqueue_call
from app.workers.pipeline import invalidate_from

router = APIRouter(prefix="/calls", tags=["calls"])

ROLES = {"client", "agent", "user", "advisor", "other"}


# ---- subida ---------------------------------------------------------------------------------------------------
@router.post("/upload", status_code=201)
def upload_call(request: Request, file: UploadFile = File(...), allow_training: bool = Form(False),
                display_name: str | None = Form(None), user: User = AnalystUser, db: Session = Depends(get_db)):
    """Recibe la grabación en streaming (sin cargarla en memoria), la valida y la encola."""
    call = create_call_from_upload(db, user, file.filename or "grabacion", file.file,
                                   allow_training=allow_training, display_name=display_name)
    call.status = CallStatus.QUEUED.value
    call.processed_by = user.id
    db.commit()
    call.task_id = enqueue_call(call.id, call.checkpoints)
    db.commit()
    return {"id": call.id, "filename": call.filename, "duration": call.duration, "status": call.status,
            "sample_rate": call.sample_rate, "channels": call.channels, "size_bytes": call.size_bytes}


# ---- listado --------------------------------------------------------------------------------------------------
@router.get("")
def list_calls(q: str | None = None, status: str | None = None, date_from: datetime | None = None,
               date_to: datetime | None = None, min_duration: float | None = None, max_duration: float | None = None,
               min_satisfaction: float | None = Query(None, ge=0, le=100), max_satisfaction: float | None = Query(None, ge=0, le=100),
               speaker: str = Query("any", pattern="^(any|SPEAKER_00|SPEAKER_01)$"), model: str | None = None,
               sort: str = "created_at", order: str = Query("desc", pattern="^(asc|desc)$"),
               page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200),
               user: User = AnyUser, db: Session = Depends(get_db)):
    stmt = select(Call).where(Call.org_id == user.org_id)
    if q:
        like = f"%{q.strip().lower()}%"
        stmt = stmt.where(or_(func.lower(Call.id).like(like), func.lower(Call.filename).like(like),
                              func.lower(func.coalesce(Call.display_name, "")).like(like),
                              cast(Call.created_at, String).like(f"%{q.strip()}%")))
    if status:
        stmt = stmt.where(Call.status == status)
    if date_from:
        stmt = stmt.where(Call.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Call.created_at < date_to + timedelta(days=1))
    if min_duration is not None:
        stmt = stmt.where(Call.duration >= min_duration)
    if max_duration is not None:
        stmt = stmt.where(Call.duration <= max_duration)
    if model:
        stmt = stmt.where(Call.model_version.like(f"%{model}%"))
    cols = {"any": [Call.satisfaction_p1, Call.satisfaction_p2], "SPEAKER_00": [Call.satisfaction_p1],
            "SPEAKER_01": [Call.satisfaction_p2]}[speaker]
    if min_satisfaction is not None:
        stmt = stmt.where(or_(*[c >= min_satisfaction for c in cols]))
    if max_satisfaction is not None:
        stmt = stmt.where(or_(*[c <= max_satisfaction for c in cols]))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    sort_col = {"created_at": Call.created_at, "duration": Call.duration, "filename": Call.filename, "status": Call.status,
                "satisfaction_p1": Call.satisfaction_p1, "satisfaction_p2": Call.satisfaction_p2}.get(sort, Call.created_at)
    stmt = stmt.order_by(sort_col.desc() if order == "desc" else sort_col.asc()).offset((page - 1) * page_size).limit(page_size)
    rows = db.scalars(stmt).all()
    return {"items": [ser.call_item(c) for c in rows], "total": total, "page": page, "page_size": page_size}


# ---- detalle --------------------------------------------------------------------------------------------------
@router.get("/{call_id}")
def get_call(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    roles = get_all(db, user.org_id)["app"]["general"]["roles_labels"]
    return ser.call_detail(call, roles)


@router.patch("/{call_id}")
def patch_call(call_id: str, body: CallPatch, request: Request, user: User = AnalystUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    if body.display_name is not None:
        call.display_name = body.display_name[:300] or None
    if body.allow_training is not None and body.allow_training != call.allow_training:
        call.allow_training = body.allow_training
        audit(db, org_id=user.org_id, user_id=user.id, action="call.allow_training", entity="call", entity_id=call.id,
              details={"value": body.allow_training}, ip=client_ip(request), commit=False)
    db.commit()
    return ser.call_item(call)


@router.delete("/{call_id}", response_model=Msg)
def delete_call(call_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    delete_call_data(db, call, "all", user)
    return Msg(message="La llamada y todos sus datos fueron eliminados.")


@router.delete("/{call_id}/data", response_model=Msg)
def delete_call_partial(call_id: str, what: str = Query(..., pattern="^(transcription|results|audio)$"),
                        user: User = AnalystUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    delete_call_data(db, call, what, user)
    return Msg(message="Datos eliminados.")


@router.get("/{call_id}/status")
def call_status(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    return ser.status_payload(get_call_or_404(db, user, call_id))


@router.post("/{call_id}/resume", response_model=Msg)
def resume_call(call_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    """Reanuda desde el último checkpoint (no reinicia etapas completadas)."""
    call = get_call_or_404(db, user, call_id)
    if call.status not in (CallStatus.ERROR.value, CallStatus.QUEUED.value, CallStatus.UPLOADED.value):
        raise Conflict("La llamada ya se está procesando o ya terminó.")
    call.status = CallStatus.QUEUED.value
    call.error_code = None
    db.commit()
    call.task_id = enqueue_call(call.id, call.checkpoints)
    db.commit()
    return Msg(message="Procesamiento reanudado desde el último checkpoint.")


@router.post("/{call_id}/reanalyze", response_model=Msg)
def reanalyze(call_id: str, from_stage: str = Query("emotion_analysis"), user: User = AnalystUser,
              db: Session = Depends(get_db)):
    """Recalcula desde una etapa (p. ej. tras activar otro modelo o cambiar pesos). El cache evita recomputar lo idéntico."""
    call = get_call_or_404(db, user, call_id)
    if call.status not in (CallStatus.COMPLETED.value, CallStatus.ERROR.value):
        raise Conflict("La llamada aún se está procesando.")
    from app.workers.pipeline import STAGES
    if from_stage not in STAGES:
        raise NotFound("Etapa desconocida.")
    if from_stage == "emotion_analysis" or from_stage == "audio_processing":
        call.model_id = None                          # toma el modelo actualmente en producción
    if from_stage in ("emotion_analysis", "satisfaction"):
        call.config_snapshot = None
    invalidate_from(db, call, from_stage)
    call.status = CallStatus.QUEUED.value
    db.commit()
    if from_stage == "emotion_analysis":
        # el modelo se fija en la etapa de audio; para re-analizar con el modelo activo se resuelve aquí
        from app.ml.emotion.factory import get_production_model_row
        row = get_production_model_row(db, call.org_id, "emotion")
        call.model_id, call.model_version = row.id, f"{row.name}:{row.version}"
        db.commit()
    call.task_id = enqueue_call(call.id, call.checkpoints)
    db.commit()
    audit(db, org_id=user.org_id, user_id=user.id, action="call.reanalyze", entity="call", entity_id=call.id,
          details={"from": from_stage})
    return Msg(message="Re-análisis encolado.")


# ---- hablantes / datos ----------------------------------------------------------------------------------------
@router.get("/{call_id}/speakers")
def call_speakers(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    roles = get_all(db, user.org_id)["app"]["general"]["roles_labels"]
    return [ser.speaker_out(s, roles) for s in call.speakers]


@router.patch("/{call_id}/speakers/{speaker_id}")
def patch_speaker(call_id: str, speaker_id: str, body: SpeakerPatch, user: User = AnalystUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    sp = next((s for s in call.speakers if s.id == speaker_id), None)
    if sp is None:
        raise NotFound("El participante no existe.")
    if body.role is not None:
        if body.role not in ROLES:
            raise Conflict("Rol de participante inválido.")
        sp.role = body.role
    if body.display_name is not None:
        sp.display_name = body.display_name[:200] or None
    db.commit()
    roles = get_all(db, user.org_id)["app"]["general"]["roles_labels"]
    return ser.speaker_out(sp, roles)


def _label_map(call: Call) -> dict[str, str]:
    return {s.id: s.label for s in call.speakers}


@router.get("/{call_id}/segments")
def call_segments(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    from app.models import AudioSegment
    call = get_call_or_404(db, user, call_id)
    lm = _label_map(call)
    rows = db.scalars(select(AudioSegment).where(AudioSegment.call_id == call.id).order_by(AudioSegment.start)).all()
    return [{"id": r.id, "speaker": lm[r.speaker_id], "start": r.start, "end": r.end, "overlap": r.overlap,
             "source": r.source} for r in rows]


@router.get("/{call_id}/transcription")
def call_transcription(call_id: str, redact: bool = False, words: bool = False, user: User = AnyUser,
                       db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    lm = _label_map(call)
    rows = db.scalars(select(Transcription).where(Transcription.call_id == call.id).order_by(Transcription.start)).all()
    return {"language": call.language, "language_confidence": call.language_confidence, "redacted": redact,
            "items": [ser.transcription_out(t, redact, lm[t.speaker_id], words) for t in rows]}


@router.get("/{call_id}/emotions")
def call_emotions(call_id: str, speaker: str | None = None, start: float | None = None, end: float | None = None,
                  user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    lm = _label_map(call)
    stmt = select(EmotionPrediction).where(EmotionPrediction.call_id == call.id)
    if speaker:
        sid = next((s.id for s in call.speakers if s.label == speaker), None)
        stmt = stmt.where(EmotionPrediction.speaker_id == sid)
    if start is not None:
        stmt = stmt.where(EmotionPrediction.end >= start)
    if end is not None:
        stmt = stmt.where(EmotionPrediction.start <= end)
    rows = db.scalars(stmt.order_by(EmotionPrediction.start)).all()
    return {"model": call.model_version, "items": [ser.emotion_out(p, lm[p.speaker_id]) for p in rows]}


@router.get("/{call_id}/satisfaction")
def call_satisfaction(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    lm = _label_map(call)
    rows = db.scalars(select(SatisfactionScore).where(SatisfactionScore.call_id == call.id)).all()
    out = {"speakers": {}, "interaction": None,
           "notice": "El modelo estima un nivel de satisfacción a partir de señales emocionales y contextuales; "
                     "no es una medición directa ni una afirmación definitiva."}
    for r in rows:
        d = {"score": r.score, "confidence": r.confidence, "trend": r.trend, "initial_score": r.initial_score,
             "final_score": r.final_score, "engine_version": r.engine_version, **(r.details or {})}
        if r.scope == "interaction":
            out["interaction"] = d
        else:
            out["speakers"][lm[r.speaker_id]] = d
    return out


@router.get("/{call_id}/events")
def call_events(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    lm = _label_map(call)
    rows = db.scalars(select(CriticalEvent).where(CriticalEvent.call_id == call.id).order_by(CriticalEvent.timestamp)).all()
    return [ser.event_out(e, lm.get(e.speaker_id)) for e in rows]


@router.get("/{call_id}/interaction")
def call_interaction(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    return get_call_or_404(db, user, call_id).interaction or {}


# ---- satisfacción real (calibración) ----------------------------------------------------------------------------
@router.post("/{call_id}/feedback", status_code=201)
def add_feedback(call_id: str, body: FeedbackIn, user: User = AnalystUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    sp = next((s for s in call.speakers if s.label == body.speaker), None) if body.speaker else None
    pred = None
    if call.summary:
        pred = ((call.summary.get("speakers", {}).get(body.speaker, {}).get("score")) if body.speaker
                else (call.summary.get("interaction", {}) or {}).get("final"))
    fb = SatisfactionFeedback(org_id=user.org_id, call_id=call.id, speaker_id=sp.id if sp else None,
                              predicted_satisfaction=pred, real_satisfaction=body.real_satisfaction, csat=body.csat,
                              nps=body.nps, survey_result=body.survey_result, created_by=user.id)
    db.add(fb)
    db.commit()
    return {"id": fb.id, "predicted_satisfaction": pred, "real_satisfaction": body.real_satisfaction}


@router.get("/{call_id}/feedback")
def list_feedback(call_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    rows = db.scalars(select(SatisfactionFeedback).where(SatisfactionFeedback.call_id == call.id)).all()
    lm = _label_map(call)
    return [{"id": r.id, "speaker": lm.get(r.speaker_id), "predicted_satisfaction": r.predicted_satisfaction,
             "real_satisfaction": r.real_satisfaction, "csat": r.csat, "nps": r.nps, "survey_result": r.survey_result,
             "created_at": r.created_at.isoformat()} for r in rows]


# ---- audio con soporte Range ---------------------------------------------------------------------------------------
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


@router.get("/{call_id}/audio")
def call_audio(call_id: str, request: Request, original: bool = False, user: User = AnyUser,
               db: Session = Depends(get_db)):
    call = get_call_or_404(db, user, call_id)
    if original and user.role == "VIEWER":
        from app.core.errors import Forbidden
        raise Forbidden()
    key = call.storage_key if original else (call.playback_key or call.storage_key)
    storage = get_storage()
    if not key or not storage.exists(key):
        raise NotFound("El audio de esta llamada ya no está disponible.")
    size = storage.size(key)
    ctype = "audio/mpeg" if key.endswith(".mp3") else "application/octet-stream"
    start, end, status = 0, size - 1, 200
    m = _RANGE.match(request.headers.get("range", ""))
    if m:
        a, b = m.groups()
        if a == "" and b:
            start = max(0, size - int(b))
        else:
            start = int(a or 0)
            end = min(int(b), size - 1) if b else size - 1
        if start > end or start >= size:
            from fastapi import Response
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "private, max-age=3600"}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    if original:
        headers["Content-Disposition"] = f'attachment; filename="{call.filename}"'
    return StreamingResponse(storage.open_range(key, start, end), status_code=status, media_type=ctype, headers=headers)
