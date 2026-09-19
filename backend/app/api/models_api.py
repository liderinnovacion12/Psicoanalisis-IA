"""Model Registry: listado, detalle, activación, archivado, comparación y evaluación."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, AnalystUser, AnyUser
from app.core.errors import Conflict, NotFound
from app.database.base import get_db
from app.ml.emotion.factory import clear_cache
from app.ml.training.versioning import activate_model, compare_models
from app.models import MLModel, ModelStatus, TrainingDataset, User
from app.schemas.common import Msg
from app.services.audit import audit
from app.workers.dispatch import enqueue_evaluation

router = APIRouter(prefix="/models", tags=["models"])


def _out(m: MLModel, detail: bool = False, active_id: str | None = None) -> dict:
    d = {"is_active": m.id == active_id, "id": m.id, "name": m.name, "version": m.version, "kind": m.kind, "status": m.status, "loader": m.loader,
         "base_model": m.base_model, "hf_id": m.hf_id, "language": m.language, "labels": m.labels,
         "dataset_id": m.dataset_id, "parameters": m.parameters, "is_builtin": m.is_builtin, "created_by": m.created_by,
         "created_at": m.created_at.isoformat(), "summary_metrics": m.summary_metrics, "training_run_id": m.training_run_id}
    if detail:
        d["metrics"] = [{"id": x.id, "split": x.split, "n_samples": x.n_samples, "accuracy": x.accuracy,
                         "precision": x.precision, "recall": x.recall, "f1": x.f1, "macro_f1": x.macro_f1,
                         "weighted_f1": x.weighted_f1, "per_class": x.per_class, "confusion_matrix": x.confusion_matrix,
                         "labels": x.labels, "extra": x.extra, "created_at": x.created_at.isoformat()}
                        for x in sorted(m.metrics, key=lambda x: x.created_at)]
    return d


def _get(db: Session, user: User, model_id: str) -> MLModel:
    m = db.get(MLModel, model_id)
    if m is None or m.org_id not in (None, user.org_id):
        raise NotFound("El modelo no existe.")
    return m


@router.get("")
def list_models(kind: str | None = None, user: User = AnyUser, db: Session = Depends(get_db)):
    stmt = select(MLModel).where(or_(MLModel.org_id == user.org_id, MLModel.org_id.is_(None)))
    if kind:
        stmt = stmt.where(MLModel.kind == kind)
    from app.ml.emotion.factory import get_production_model_row
    active = {}
    for k in ("emotion", "satisfaction"):
        try:
            active[k] = get_production_model_row(db, user.org_id, k).id
        except LookupError:
            active[k] = None
    return [_out(m, False, active.get(m.kind)) for m in db.scalars(stmt.order_by(MLModel.created_at.desc())).all()]


@router.get("/compare")
def compare(ids: list[str] = Query(...), dataset_id: str | None = None, user: User = AnyUser, db: Session = Depends(get_db)):
    return compare_models(db, ids, user.org_id, dataset_id)


@router.get("/{model_id}")
def get_model(model_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    m = _get(db, user, model_id)
    from app.ml.emotion.factory import get_production_model_row
    try:
        active = get_production_model_row(db, user.org_id, m.kind).id
    except LookupError:
        active = None
    return _out(m, True, active)


@router.post("/{model_id}/activate", response_model=Msg)
def activate(model_id: str, user: User = AdminUser, db: Session = Depends(get_db)):
    """Las nuevas llamadas usarán este modelo (cada análisis guarda `model_version`)."""
    m = _get(db, user, model_id)
    activate_model(db, m, user.org_id)
    audit(db, org_id=user.org_id, user_id=user.id, action="model.activate", entity="model", entity_id=m.id,
          details={"name": m.name, "version": m.version}, commit=False)
    db.commit()
    clear_cache()
    return Msg(message=f"Modelo {m.name} activado. Las nuevas llamadas lo utilizarán.")


@router.post("/{model_id}/archive", response_model=Msg)
def archive(model_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    m = _get(db, user, model_id)
    if m.is_builtin:
        raise Conflict("El modelo base integrado no se puede archivar.")
    if m.status == ModelStatus.PRODUCTION.value:
        raise Conflict("No se puede archivar el modelo en producción; active otro primero.")
    m.status = ModelStatus.ARCHIVED.value
    db.commit()
    return Msg(message="Modelo archivado.")


@router.post("/{model_id}/evaluate", response_model=Msg)
def evaluate(model_id: str, dataset_id: str, split: str = "test", user: User = AnalystUser, db: Session = Depends(get_db)):
    """Evalúa el modelo sobre el split de un dataset (permite comparar baseline vs. fine-tuned con los MISMOS datos)."""
    m = _get(db, user, model_id)
    ds = db.get(TrainingDataset, dataset_id)
    if ds is None or ds.org_id != user.org_id:
        raise NotFound("El dataset no existe.")
    if not any(s.split == split for s in ds.samples):
        raise Conflict("El dataset no tiene muestras en ese conjunto. Cree los splits primero.")
    enqueue_evaluation(m.id, ds.id, split)
    return Msg(message="Evaluación encolada. Las métricas aparecerán en el detalle del modelo.")
