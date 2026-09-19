"""Model Registry: registro, versionado, activación y comparación de modelos.

Estados: TRAINING → VALIDATION → PRODUCTION → ARCHIVED. Solo un modelo PRODUCTION por (organización, tipo);
activar uno archiva el anterior (se puede reactivar: rollback).
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import Conflict
from app.models import MLModel, ModelMetric, ModelStatus


def next_version(db: Session, org_id: str, kind: str) -> tuple[str, str]:
    """(version, nombre) p. ej. ('v3', 'emotion_model_v3')."""
    n = db.scalar(select(func.count()).select_from(MLModel).where(MLModel.org_id == org_id, MLModel.kind == kind,
                                                                   MLModel.is_builtin.is_(False))) or 0
    v = f"v{n + 1}"
    return v, f"{kind}_model_{v}"


def register_model(db: Session, *, org_id: str, kind: str, loader: str, path: str | None, base_model: str | None,
                   dataset_id: str | None, params: dict, labels: list[str] | None, language: str | None,
                   created_by: str | None, status: str = ModelStatus.TRAINING.value, run_id: str | None = None,
                   hf_id: str | None = None) -> MLModel:
    version, name = next_version(db, org_id, kind)
    m = MLModel(org_id=org_id, name=name, version=version, kind=kind, loader=loader, path=path, base_model=base_model,
                dataset_id=dataset_id, parameters=params, labels=labels, language=language, status=status,
                created_by=created_by, training_run_id=run_id, hf_id=hf_id)
    db.add(m)
    db.flush()
    return m


def add_metrics(db: Session, model: MLModel, split: str, metrics: dict, run_id: str | None = None,
                dataset_id: str | None = None) -> ModelMetric:
    extra = {k: v for k, v in metrics.items() if k not in ("n_samples", "accuracy", "precision", "recall", "f1",
                                                          "macro_f1", "weighted_f1", "per_class", "confusion_matrix", "labels")}
    if dataset_id:
        extra["dataset_id"] = dataset_id
    mm = ModelMetric(model_id=model.id, run_id=run_id, split=split, n_samples=metrics.get("n_samples", 0),
                     accuracy=metrics.get("accuracy"), precision=metrics.get("precision"), recall=metrics.get("recall"),
                     f1=metrics.get("f1"), macro_f1=metrics.get("macro_f1"), weighted_f1=metrics.get("weighted_f1"),
                     per_class=metrics.get("per_class"), confusion_matrix=metrics.get("confusion_matrix"),
                     labels=metrics.get("labels"), extra=extra or None)
    db.add(mm)
    model.summary_metrics = {"split": split, **{k: metrics.get(k) for k in
                                                ("accuracy", "macro_f1", "weighted_f1", "precision", "recall", "n_samples")},
                             **{k: v for k, v in extra.items() if isinstance(v, (int, float))}}
    return mm


def activate_model(db: Session, model: MLModel, org_id: str) -> MLModel:
    """Activa el modelo para la organización. Un modelo propio en PRODUCTION tiene prioridad sobre el baseline global;
    activar el baseline (global) archiva los modelos propios en producción (rollback)."""
    if model.org_id not in (None, org_id):
        raise Conflict("El modelo no pertenece a su organización.")
    if model.status == ModelStatus.TRAINING.value:
        raise Conflict("El modelo aún se está entrenando.")
    q = select(MLModel).where(MLModel.kind == model.kind, MLModel.status == ModelStatus.PRODUCTION.value,
                              MLModel.org_id == org_id)
    for cur in db.scalars(q).all():
        if cur.id != model.id:
            cur.status = ModelStatus.ARCHIVED.value
    if model.org_id is not None:
        model.status = ModelStatus.PRODUCTION.value
    db.flush()
    return model


def compare_models(db: Session, ids: list[str], org_id: str, dataset_id: str | None = None) -> list[dict]:
    """Métricas lado a lado. Si `dataset_id` se indica, usa las evaluaciones sobre ese dataset (comparación justa)."""
    out = []
    for mid in ids:
        m = db.get(MLModel, mid)
        if m is None or m.org_id not in (None, org_id):
            continue
        mets = sorted(m.metrics, key=lambda x: x.created_at)
        pick = None
        for x in reversed(mets):
            if dataset_id and (x.extra or {}).get("dataset_id") != dataset_id:
                continue
            if x.split == "test":
                pick = x
                break
        if pick is None and not dataset_id:
            pick = next((x for x in reversed(mets)), None)
        out.append({"id": m.id, "name": m.name, "version": m.version, "kind": m.kind, "status": m.status,
                    "base_model": m.base_model, "dataset_id": m.dataset_id,
                    "metrics": None if pick is None else {
                        "split": pick.split, "n_samples": pick.n_samples, "accuracy": pick.accuracy, "precision": pick.precision,
                        "recall": pick.recall, "f1": pick.f1, "macro_f1": pick.macro_f1, "weighted_f1": pick.weighted_f1,
                        "per_class": pick.per_class, "confusion_matrix": pick.confusion_matrix, "labels": pick.labels,
                        "extra": pick.extra}})
    return out
