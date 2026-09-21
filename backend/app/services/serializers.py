"""Conversión de entidades a estructuras JSON de la API. Nunca se exponen errores técnicos."""
from __future__ import annotations

from app.models import Call, CriticalEvent, EmotionPrediction, Speaker, Transcription
from app.services import narrative
from app.workers.pipeline import STAGE_LABELS, STAGES

GENERIC_ERROR = "No fue posible completar el análisis de la llamada. Puede reintentar el procesamiento."
AUDIO_ERROR = "No fue posible procesar el audio. Verifique que el archivo sea válido."


def user_error_message(call: Call) -> str | None:
    if call.status != "ERROR":
        return None
    code = call.error_code or ""
    parts = code.split(":", 2)
    if len(parts) == 3:                      # ProcessingError con mensaje apto para el usuario
        return parts[2]
    if code.startswith("audio_processing"):
        return AUDIO_ERROR
    return GENERIC_ERROR


def _iso(d):
    return d.isoformat() if d else None


def stages_payload(call: Call) -> list[dict]:
    cps = call.checkpoints or {}
    out = []
    for st in STAGES:
        cp = cps.get(st, {})
        state = cp.get("status", "PENDING")
        out.append({"key": st, "label": STAGE_LABELS[st], "state": state, "progress": cp.get("progress", 100 if state == "COMPLETED" else 0),
                    "message": cp.get("message"), "seconds": cp.get("seconds"), "started_at": cp.get("started_at")})
    return out


def status_payload(call: Call) -> dict:
    return {"id": call.id, "status": call.status, "progress": call.progress, "stages": stages_payload(call),
            "error": user_error_message(call), "resumable": call.status == "ERROR"}


def speaker_out(s: Speaker, roles_labels: dict | None = None) -> dict:
    return {"id": s.id, "label": s.label, "role": s.role, "display_name": s.display_name, "channel": s.channel,
            "talk_time": s.talk_time, "name": s.display_name or narrative.person_name(s.label, s.role, roles_labels)}


def call_item(c: Call) -> dict:
    sm = (c.summary or {}).get("speakers", {})
    return {
        "id": c.id, "filename": c.filename, "display_name": c.display_name, "created_at": _iso(c.created_at),
        "duration": c.duration, "status": c.status, "progress": c.progress, "language": c.language,
        "satisfaction_p1": c.satisfaction_p1, "satisfaction_p2": c.satisfaction_p2,
        "model_version": c.model_version, "audio_quality": (c.audio_quality or {}).get("score"),
        "analysis_quality": (c.analysis_quality or {}).get("score"), "allow_training": c.allow_training,
        "frustration_p1": sm.get("SPEAKER_00", {}).get("metrics", {}).get("frustration"),
        "frustration_p2": sm.get("SPEAKER_01", {}).get("metrics", {}).get("frustration"),
        "error": user_error_message(c),
    }


def call_detail(c: Call, roles_labels: dict | None = None) -> dict:
    d = call_item(c)
    d.update({
        "uploaded_by": c.uploaded_by, "size_bytes": c.size_bytes, "sample_rate": c.sample_rate, "channels": c.channels,
        "bitrate": c.bitrate, "language_confidence": c.language_confidence, "diarization_mode": c.diarization_mode,
        "audio_quality_detail": c.audio_quality, "analysis_quality_detail": c.analysis_quality,
        "warnings": c.warnings or [], "summary": c.summary, "interaction": c.interaction,
        "speakers": [speaker_out(s, roles_labels) for s in c.speakers], "stages": stages_payload(c),
        "model_id": c.model_id, "completed_at": _iso(c.completed_at), "retention_until": _iso(c.retention_until),
        "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk not in ("files",)} for k, v in (c.checkpoints or {}).items()},
    })
    return d


def transcription_out(t: Transcription, redact: bool, label: str, include_words: bool = False) -> dict:
    d = {"id": t.id, "speaker": label, "start": t.start, "end": t.end,
         "text": (t.text_redacted if redact and t.text_redacted is not None else t.text),
         "confidence": t.confidence, "language": t.language, "sentiment": t.sentiment,
         "pii_count": len(t.pii or [])}
    if include_words and not redact:
        d["words"] = t.words
    return d


def emotion_out(p: EmotionPrediction, label: str) -> dict:
    return {"id": p.id, "speaker": label, "segment_id": p.segment_id, "start": p.start, "end": p.end,
            "duration": p.duration, "emotion": p.emotion, "confidence": p.confidence,
            "probabilities": p.probabilities, "model": p.model_name, "model_version": p.model_version,
            "prosody": p.prosody}


def event_out(e: CriticalEvent, label: str | None) -> dict:
    return {"id": e.id, "speaker": label, "timestamp": e.timestamp, "end_timestamp": e.end_timestamp,
            "event_type": e.event_type, "label": narrative.EVENT_ES.get(e.event_type, e.event_type),
            "severity": e.severity, "confidence": e.confidence, "emotion": e.emotion, "satisfaction": e.satisfaction,
            "description": e.description, "evidence": e.evidence}
