"""Configuración efectiva por organización = YAML por defecto + overrides guardados en `settings`."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import CONFIG_SECTIONS, deep_merge, get_config_store
from app.core.errors import AppError
from app.models import Setting


def get_overrides(db: Session, org_id: str, section: str) -> dict:
    row = db.scalar(select(Setting).where(Setting.org_id == org_id, Setting.key == section))
    return row.value if row else {}


def get_section(db: Session, org_id: str, section: str) -> dict:
    if section not in CONFIG_SECTIONS:
        raise AppError("Sección de configuración desconocida.", status_code=404)
    return get_config_store().resolve(section, get_overrides(db, org_id, section))


def get_all(db: Session, org_id: str) -> dict:
    return {s: get_section(db, org_id, s) for s in CONFIG_SECTIONS}


def set_section(db: Session, org_id: str, section: str, value: dict, user_id: str | None) -> dict:
    """Guarda solo las diferencias respecto a los valores por defecto (overrides mínimos)."""
    if section not in CONFIG_SECTIONS:
        raise AppError("Sección de configuración desconocida.", status_code=404)
    defaults = get_config_store().defaults(section)
    merged = deep_merge(defaults, value)
    _validate(section, merged)
    diff = _diff(defaults, merged)
    row = db.scalar(select(Setting).where(Setting.org_id == org_id, Setting.key == section))
    if row:
        row.value, row.updated_by = diff, user_id
    else:
        db.add(Setting(org_id=org_id, key=section, value=diff, updated_by=user_id))
    db.commit()
    return merged


def reset_section(db: Session, org_id: str, section: str) -> dict:
    row = db.scalar(select(Setting).where(Setting.org_id == org_id, Setting.key == section))
    if row:
        db.delete(row)
        db.commit()
    return get_config_store().defaults(section)


def _diff(base: dict, new: dict) -> dict:
    out = {}
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            d = _diff(base[k], v)
            if d:
                out[k] = d
        elif base.get(k) != v:
            out[k] = v
    return out


def _validate(section: str, cfg: dict) -> None:
    try:
        if section == "audio":
            seg = cfg["segmentation"]
            assert 0.5 <= seg["window_size"] <= 30, "window_size"
            assert 0.1 <= seg["hop_size"] <= seg["window_size"], "hop_size"
            assert cfg["normalization"]["sample_rate"] in (8000, 16000, 22050, 24000, 44100, 48000)
        elif section == "satisfaction":
            assert cfg["mode"] in ("rules", "model", "hybrid")
            rng = cfg["interpretation"]
            assert all(r["min"] <= r["max"] for r in rng)
            assert cfg["score"]["scale"] > 0
        elif section == "emotion":
            assert cfg["device"] in ("auto", "cuda", "cpu")
    except (AssertionError, KeyError, TypeError) as e:
        raise AppError("Configuración inválida. Revise los valores ingresados.", detail=str(e), status_code=422)
