"""Resolución del modelo emocional a partir del Model Registry. Cambiar de modelo = activar otro registro."""
from __future__ import annotations

import threading
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_config_store, get_settings
from app.core.logging import get_logger
from app.ml.emotion.base import EmotionModel
from app.ml.emotion.wav2vec import BASELINE_HF_ID, FineTunedEmotionModel, Wav2VecEmotionModel
from app.models import MLModel, ModelStatus

log = get_logger(__name__)

# loader -> constructor(row: MLModel, emotion_cfg: dict) -> EmotionModel
_LOADERS: dict[str, Callable[[MLModel, dict], EmotionModel]] = {}
_CACHE: dict[str, EmotionModel] = {}
_LOCK = threading.Lock()


def register_loader(name: str, fn: Callable[[MLModel, dict], EmotionModel]) -> None:
    """Punto de extensión: registre aquí un nuevo tipo de modelo sin tocar el resto de la app."""
    _LOADERS[name] = fn


def _opts(cfg: dict) -> dict:
    return {"max_seconds": cfg.get("max_window_seconds", 30), "fp16": cfg.get("fp16_on_cuda", True),
            "temperature": cfg.get("temperature", 1.0)}


register_loader("wav2vec", lambda m, cfg: Wav2VecEmotionModel(
    hf_id=m.hf_id or BASELINE_HF_ID, name=m.name, version=m.version, language=m.language, **_opts(cfg)))
register_loader("finetuned", lambda m, cfg: FineTunedEmotionModel(
    path=m.path or "", name=m.name, version=m.version, language=m.language, **_opts(cfg)))


def build_model(row: MLModel, cfg: dict | None = None) -> EmotionModel:
    cfg = cfg or get_config_store().defaults("emotion")
    key = f"{row.id}:{row.path or row.hf_id}"
    with _LOCK:
        if key not in _CACHE:
            if row.loader not in _LOADERS:
                raise ValueError(f"loader de modelo desconocido: {row.loader}")
            _CACHE[key] = _LOADERS[row.loader](row, cfg)
    return _CACHE[key].ensure_loaded()


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def ensure_builtin_model(db: Session) -> MLModel:
    """Registra el baseline (Hugging Face) si aún no existe; queda en PRODUCTION por defecto."""
    row = db.scalar(select(MLModel).where(MLModel.is_builtin.is_(True), MLModel.kind == "emotion"))
    if row:
        return row
    cfg = get_config_store().defaults("emotion")["default_model"]
    has_prod = db.scalar(select(MLModel).where(MLModel.kind == "emotion",
                                               MLModel.status == ModelStatus.PRODUCTION.value)) is not None
    row = MLModel(org_id=None, name=cfg["name"], version="baseline", kind="emotion", loader=cfg["loader"],
                  base_model=None, hf_id=get_settings().model_name or cfg["hf_id"], language=cfg.get("language"),
                  status=ModelStatus.PRODUCTION.value,       # el baseline global siempre es el respaldo por defecto
                  is_builtin=True, parameters={"note": "Baseline. La accuracy publicada NO refleja llamadas reales."},
                  labels=get_config_store().defaults("emotion")["canonical_emotions"])
    db.add(row)
    db.commit()
    return row


def get_production_model_row(db: Session, org_id: str | None, kind: str = "emotion") -> MLModel:
    """Modelo activo: PRODUCTION de la organización; si no hay, PRODUCTION global (builtin)."""
    q = select(MLModel).where(MLModel.kind == kind, MLModel.status == ModelStatus.PRODUCTION.value)
    rows = db.scalars(q).all()
    own = [r for r in rows if r.org_id == org_id and org_id]
    if own:
        return sorted(own, key=lambda r: r.created_at)[-1]
    glob = [r for r in rows if r.org_id is None]
    if glob:
        return glob[0]
    if kind == "emotion":
        return ensure_builtin_model(db)
    raise LookupError(f"no hay modelo {kind} en producción")
