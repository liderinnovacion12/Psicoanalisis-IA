"""Despacho de trabajo: Celery (producción) o hilos en el proceso API (desarrollo, sin Redis)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from app.core.config import get_settings
from app.core.logging import get_logger
from app.workers.pipeline import STAGES, run_pipeline, stage_done

log = get_logger(__name__)
_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=max(1, get_settings().inline_workers),
                                           thread_name_prefix="inline-worker")
        return _executor


def enqueue_call(call_id: str, checkpoints: dict | None = None) -> str | None:
    """Encola (o reanuda) el análisis. Devuelve el id de tarea si existe."""
    s = get_settings()
    if s.task_mode == "celery":
        from app.workers.celery_app import STAGE_QUEUE, celery_app
        first = next((st for st in STAGES if (checkpoints or {}).get(st, {}).get("status") != "COMPLETED"), STAGES[-1])
        r = celery_app.send_task("pipeline.run_stage", args=[call_id, first], queue=STAGE_QUEUE[first])
        return r.id
    _pool().submit(run_pipeline, call_id)
    return None


def enqueue_training(run_id: str) -> str | None:
    s = get_settings()
    if s.task_mode == "celery":
        from app.workers.celery_app import celery_app
        return celery_app.send_task("training.run", args=[run_id], queue="training").id
    from app.services.training_service import execute_run
    threading.Thread(target=execute_run, args=(run_id,), daemon=True, name=f"train-{run_id[:6]}").start()
    return None


def enqueue_evaluation(model_id: str, dataset_id: str, split: str = "test") -> None:
    s = get_settings()
    if s.task_mode == "celery":
        from app.workers.celery_app import celery_app
        celery_app.send_task("training.evaluate", args=[model_id, dataset_id, split], queue="training")
        return
    from app.services.training_service import execute_evaluation
    threading.Thread(target=execute_evaluation, args=(model_id, dataset_id, split), daemon=True).start()
