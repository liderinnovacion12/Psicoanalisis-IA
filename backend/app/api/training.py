"""Entrenamiento: iniciar, listar, detalle/progreso, detener y continuar."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AnalystUser, AnyUser
from app.core.errors import Conflict, NotFound
from app.database.base import get_db
from app.models import RunStatus, TrainingRun, User
from app.schemas.common import Msg, TrainingStart
from app.services import training_service as tsvc
from app.services.audit import audit
from app.workers.dispatch import enqueue_training

router = APIRouter(prefix="/training", tags=["training"])


def _get(db: Session, user: User, run_id: str) -> TrainingRun:
    r = db.get(TrainingRun, run_id)
    if r is None or r.org_id != user.org_id:
        raise NotFound("El entrenamiento no existe.")
    return r


@router.post("/start", status_code=201)
def start(body: TrainingStart, user: User = AnalystUser, db: Session = Depends(get_db)):
    active = db.scalar(select(TrainingRun.id).where(TrainingRun.org_id == user.org_id,
                                                    TrainingRun.status.in_([RunStatus.RUNNING.value, RunStatus.PENDING.value])))
    if active:
        raise Conflict("Ya hay un entrenamiento en curso. Espere a que termine o deténgalo.")
    run = tsvc.create_run(db, user, dataset_id=body.dataset_id, kind=body.kind, base_model_id=body.base_model_id,
                          name=body.name, params=body.params, resplit=body.resplit)
    run.task_id = enqueue_training(run.id)
    db.commit()
    return tsvc.run_out(run)


@router.get("")
def list_runs(user: User = AnyUser, db: Session = Depends(get_db)):
    rows = db.scalars(select(TrainingRun).where(TrainingRun.org_id == user.org_id).order_by(TrainingRun.created_at.desc())).all()
    return [tsvc.run_out(r) for r in rows]


@router.get("/{run_id}")
def get_run(run_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    return tsvc.run_out(_get(db, user, run_id))


@router.post("/{run_id}/stop", response_model=Msg)
def stop(run_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    r = _get(db, user, run_id)
    if r.status not in (RunStatus.RUNNING.value, RunStatus.PENDING.value):
        raise Conflict("El entrenamiento no está en ejecución.")
    r.stop_requested = True
    r.status = RunStatus.STOPPING.value if r.status == RunStatus.RUNNING.value else RunStatus.STOPPED.value
    audit(db, org_id=user.org_id, user_id=user.id, action="training.stop", entity="training_run", entity_id=r.id, commit=False)
    db.commit()
    return Msg(message="Se solicitó detener el entrenamiento; se guardará un checkpoint.")


@router.post("/{run_id}/resume", response_model=Msg)
def resume(run_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    r = _get(db, user, run_id)
    if r.status not in (RunStatus.STOPPED.value, RunStatus.FAILED.value):
        raise Conflict("Solo se puede continuar un entrenamiento detenido o fallido.")
    if db.scalar(select(TrainingRun.id).where(TrainingRun.org_id == user.org_id, TrainingRun.id != r.id,
                                              TrainingRun.status.in_([RunStatus.RUNNING.value, RunStatus.PENDING.value]))):
        raise Conflict("Ya hay otro entrenamiento en curso.")
    r.status, r.stop_requested, r.error = RunStatus.PENDING.value, False, None
    db.commit()
    r.task_id = enqueue_training(r.id)
    db.commit()
    return Msg(message="Entrenamiento reanudado desde el último checkpoint.")
