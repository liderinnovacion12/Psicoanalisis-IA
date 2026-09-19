"""Configuración por organización (audio, emociones, satisfacción, app) + calibración con satisfacción real."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, AnalystUser, AnyUser, client_ip
from app.core.config import CONFIG_SECTIONS, get_config_store
from app.core.errors import Conflict
from app.database.base import get_db
from app.ml.satisfaction.model import fit_linear_calibration
from app.models import SatisfactionFeedback, User
from app.schemas.common import ConfigIn
from app.services import config_service as cs
from app.services.audit import audit

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def all_settings(user: User = AnyUser, db: Session = Depends(get_db)):
    return {"config": cs.get_all(db, user.org_id),
            "defaults": {s: get_config_store().defaults(s) for s in CONFIG_SECTIONS}}


@router.put("/{section}")
def update_section(section: str, body: ConfigIn, request: Request, user: User = AdminUser, db: Session = Depends(get_db)):
    """Guarda solo las diferencias respecto a los valores por defecto (override por organización)."""
    merged = cs.set_section(db, user.org_id, section, body.value, user.id)
    audit(db, org_id=user.org_id, user_id=user.id, action="settings.update", entity="settings", entity_id=section,
          details={"keys": list(body.value.keys())}, ip=client_ip(request))
    return merged


@router.delete("/{section}")
def reset_section(section: str, user: User = AdminUser, db: Session = Depends(get_db)):
    d = cs.reset_section(db, user.org_id, section)
    audit(db, org_id=user.org_id, user_id=user.id, action="settings.reset", entity="settings", entity_id=section)
    return d


@router.get("/calibration/status")
def calibration_status(user: User = AnyUser, db: Session = Depends(get_db)):
    """Compara la satisfacción predicha con la real (CSAT/encuesta) registrada."""
    rows = db.scalars(select(SatisfactionFeedback).where(SatisfactionFeedback.org_id == user.org_id,
                                                         SatisfactionFeedback.real_satisfaction.is_not(None),
                                                         SatisfactionFeedback.predicted_satisfaction.is_not(None))).all()
    pred, real = [r.predicted_satisfaction for r in rows], [r.real_satisfaction for r in rows]
    cfg = cs.get_section(db, user.org_id, "satisfaction")["calibration"]
    fit = fit_linear_calibration(pred, real) if len(pred) >= 3 else None
    return {"n": len(pred), "min_samples": cfg["min_samples"], "enabled": cfg["enabled"], "current": cfg["coefficients"],
            "proposed": fit, "pairs": [{"predicted": p, "real": r} for p, r in zip(pred, real)][:500]}


@router.post("/calibration/apply")
def calibration_apply(user: User = AdminUser, db: Session = Depends(get_db)):
    st = calibration_status(user, db)
    if st["n"] < st["min_samples"] or not st["proposed"]:
        raise Conflict(f"Se necesitan al menos {st['min_samples']} pares (predicha, real) para calibrar de forma fiable.")
    p = st["proposed"]
    cs.set_section(db, user.org_id, "satisfaction", {"calibration": {"enabled": True, "coefficients": {"a": p["a"], "b": p["b"]}}}, user.id)
    audit(db, org_id=user.org_id, user_id=user.id, action="calibration.apply", details=p)
    return {"applied": p}
