"""Tareas Celery. Cada etapa del pipeline es una tarea y encola la siguiente en SU cola."""
from __future__ import annotations

from app.core.logging import get_logger
from app.workers.celery_app import STAGE_QUEUE, celery_app
from app.workers.pipeline import STAGES, run_stage

log = get_logger(__name__)


@celery_app.task(name="pipeline.run_stage", bind=True)
def run_stage_task(self, call_id: str, stage: str) -> bool:
    ok = run_stage(call_id, stage)
    if ok and stage != STAGES[-1]:
        nxt = STAGES[STAGES.index(stage) + 1]
        celery_app.send_task("pipeline.run_stage", args=[call_id, nxt], queue=STAGE_QUEUE[nxt])
    return ok


@celery_app.task(name="training.run", bind=True)
def training_task(self, run_id: str) -> None:
    from app.services.training_service import execute_run
    execute_run(run_id)


@celery_app.task(name="training.evaluate")
def evaluate_task(model_id: str, dataset_id: str, split: str = "test") -> None:
    from app.services.training_service import execute_evaluation
    execute_evaluation(model_id, dataset_id, split)


@celery_app.task(name="maintenance.purge_expired")
def purge_expired_task() -> int:
    from app.services.retention import purge_expired
    return purge_expired()
