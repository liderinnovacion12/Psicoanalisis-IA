"""Salud, información del sistema (GPU/CPU, modelos) y dashboard agregado."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import AnyUser
from app.core.config import get_settings
from app.core.device import detect_device
from app.database.base import get_db
from app.models import Call, MLModel, User
from app.services import serializers as ser

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/system/info")
def system_info(user: User = AnyUser, db: Session = Depends(get_db)):
    s = get_settings()
    dev = detect_device(s.device)
    from app.ml.audio.ffmpeg import ffmpeg_path
    from app.ml.diarization.pyannote import pyannote_available
    try:
        ff = ffmpeg_path()
    except Exception:
        ff = None
    prod = db.scalar(select(MLModel).where(MLModel.kind == "emotion", MLModel.status == "PRODUCTION"))
    return {"device": dev.to_dict(), "task_mode": s.task_mode, "storage_backend": s.storage_backend,
            "ffmpeg": bool(ff), "pyannote_available": pyannote_available(),
            "hf_token_configured": bool(s.hf_token),
            "active_emotion_model": None if not prod else {"id": prod.id, "name": prod.name, "version": prod.version,
                                                             "language": prod.language}}


@router.get("/dashboard/summary")
def dashboard(days: int = 90, user: User = AnyUser, db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    base = select(Call).where(Call.org_id == user.org_id)
    all_calls = db.scalars(base.where(Call.created_at >= since).order_by(Call.created_at)).all()
    done = [c for c in all_calls if c.status == "COMPLETED" and c.summary]
    total = db.scalar(select(func.count()).select_from(Call).where(Call.org_id == user.org_id)) or 0

    def avg(vals):
        v = [x for x in vals if x is not None]
        return round(float(np.mean(v)), 1) if v else None

    sats = [s["score"] for c in done for s in c.summary["speakers"].values()]
    frusts = [s["metrics"]["frustration"] * 100 for c in done for s in c.summary["speakers"].values()]
    emo_acc: dict[str, list[float]] = {}
    for c in done:
        for s in c.summary["speakers"].values():
            for k, v in s["metrics"]["mean_probabilities"].items():
                emo_acc.setdefault(k, []).append(v)
    dom: dict[str, int] = {}
    for c in done:
        for s in c.summary["speakers"].values():
            d = s["metrics"]["dominant_emotion"]
            dom[d] = dom.get(d, 0) + 1
    # evolución diaria y distribución de satisfacción
    daily: dict[str, list[float]] = {}
    for c in done:
        day = c.created_at.date().isoformat()
        daily.setdefault(day, []).extend(s["score"] for s in c.summary["speakers"].values())
    bins = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for v in sats:
        k = "0-20" if v <= 20 else "21-40" if v <= 40 else "41-60" if v <= 60 else "61-80" if v <= 80 else "81-100"
        bins[k] += 1
    recent = db.scalars(base.order_by(Call.created_at.desc()).limit(8)).all()
    return {
        "totals": {"total_calls": total, "analyzed_calls": len(done),
                   "processing": sum(1 for c in all_calls if c.status not in ("COMPLETED", "ERROR")),
                   "errors": sum(1 for c in all_calls if c.status == "ERROR"),
                   "avg_satisfaction": avg(sats), "avg_frustration": avg(frusts),
                   "avg_duration": avg([c.duration for c in done]),
                   "avg_quality": avg([(c.analysis_quality or {}).get("score") for c in done])},
        "avg_emotions": {k: round(float(np.mean(v)), 4) for k, v in emo_acc.items()},
        "dominant_emotions": dom,
        "satisfaction_distribution": bins,
        "satisfaction_evolution": [{"date": d, "avg": round(float(np.mean(v)), 1), "n": len(v)} for d, v in sorted(daily.items())],
        "recent": [ser.call_item(c) for c in recent],
        "days": days,
    }
