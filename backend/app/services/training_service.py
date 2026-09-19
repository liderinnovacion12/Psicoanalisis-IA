"""Ejecución de entrenamientos (emoción: fine-tuning; satisfacción: regresor), evaluación y registro de modelos."""
from __future__ import annotations

import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_config_store, get_settings
from app.core.device import detect_device
from app.core.errors import AppError, Conflict, StopRequested
from app.core.logging import get_logger
from app.database.base import SessionLocal
from app.ml.emotion.factory import build_model, get_production_model_row
from app.ml.training.dataset import Sample
from app.ml.training.evaluator import metrics_from_predictions, regression_metrics
from app.ml.training.trainer import Callbacks, EmotionTrainer, TrainParams
from app.ml.training.versioning import add_metrics, register_model
from app.models import (AudioSegment, Call, EmotionPrediction, MLModel, ModelStatus, RunStatus, SatisfactionFeedback,
                        Speaker, TrainingDataset, TrainingRun, TrainingSample, Transcription, User)
from app.services.audit import audit
from app.services.config_service import get_all
from app.services.dataset_service import recompute_stats, split_dataset
from app.services.storage import get_storage

log = get_logger(__name__)


# -------------------------------------------------------------------------------------------------------------
def create_run(db: Session, user: User, *, dataset_id: str, kind: str, base_model_id: str | None, name: str | None,
               params: dict, resplit: bool) -> TrainingRun:
    ds = db.get(TrainingDataset, dataset_id)
    if ds is None or ds.org_id != user.org_id:
        raise AppError("El dataset no existe.", status_code=404)
    if kind not in ("emotion", "satisfaction"):
        raise AppError("Tipo de entrenamiento inválido.", status_code=422)
    defaults = get_all(db, user.org_id)["app"]["training"]["defaults"]
    merged = {**defaults, **{k: v for k, v in (params or {}).items() if v is not None}}
    if kind == "emotion":
        if base_model_id is None:
            base_model_id = get_production_model_row(db, user.org_id, "emotion").id
        base = db.get(MLModel, base_model_id)
        if base is None or base.org_id not in (None, user.org_id) or base.kind != "emotion":
            raise AppError("El modelo base no existe.", status_code=404)
    run = TrainingRun(org_id=user.org_id, name=name or f"{ds.name}_v{ds.version}", kind=kind, dataset_id=ds.id,
                      base_model_id=base_model_id, params={**merged, "resplit": resplit}, status=RunStatus.PENDING.value,
                      total_epochs=int(merged.get("epochs", 0)) if kind == "emotion" else 1, created_by=user.id,
                      device=detect_device(get_settings().device).label)
    db.add(run)
    db.flush()
    audit(db, org_id=user.org_id, user_id=user.id, action="training.start", entity="training_run", entity_id=run.id,
          details={"dataset": ds.id, "kind": kind, "params": run.params}, commit=False)
    db.commit()
    return run


def run_out(r: TrainingRun) -> dict:
    return {"id": r.id, "name": r.name, "kind": r.kind, "dataset_id": r.dataset_id, "base_model_id": r.base_model_id,
            "model_id": r.model_id, "status": r.status, "progress": r.progress, "current_epoch": r.current_epoch,
            "total_epochs": r.total_epochs, "params": r.params, "metrics_history": r.metrics_history,
            "best_metric": r.best_metric, "error": ("El entrenamiento falló. Revise los datos e intente nuevamente."
                                                    if r.status == RunStatus.FAILED.value else None),
            "device": r.device, "created_at": r.created_at.isoformat(),
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "resumable": r.status in (RunStatus.STOPPED.value, RunStatus.FAILED.value)}


# -------------------------------------------------------------------------------------------------------------
def execute_run(run_id: str) -> None:
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = RunStatus.RUNNING.value
        run.stop_requested = False
        run.started_at = run.started_at or datetime.now(timezone.utc)
        db.commit()
        if run.kind == "emotion":
            _train_emotion(db, run)
        else:
            _train_satisfaction(db, run)
        run = db.get(TrainingRun, run_id)
        run.status, run.progress = RunStatus.COMPLETED.value, 100.0
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
        log.info("entrenamiento completado", extra={"run_id": run_id})
    except StopRequested:
        db.rollback()
        run = db.get(TrainingRun, run_id)
        run.status, run.finished_at = RunStatus.STOPPED.value, datetime.now(timezone.utc)
        db.commit()
        log.info("entrenamiento detenido por el usuario", extra={"run_id": run_id})
    except Exception as e:
        db.rollback()
        log.error("entrenamiento fallido", extra={"run_id": run_id, "trace": traceback.format_exc()})
        run = db.get(TrainingRun, run_id)
        if run:
            run.status, run.error, run.finished_at = RunStatus.FAILED.value, f"{type(e).__name__}: {e}", datetime.now(timezone.utc)
            if isinstance(e, AppError):
                run.error = e.user_message
            db.commit()
    finally:
        db.close()


def _stop_checker(run_id: str):
    state = {"t": 0.0, "v": False}

    def should_stop() -> bool:
        if state["v"]:
            return True
        if time.time() - state["t"] > 1.5:
            state["t"] = time.time()
            s = SessionLocal()
            try:
                r = s.get(TrainingRun, run_id)
                state["v"] = bool(r and r.stop_requested)
            finally:
                s.close()
        return state["v"]
    return should_stop


def _materialize(db: Session, samples: list[TrainingSample], dst: Path) -> dict[str, str]:
    """Copia (y descifra si aplica) los recortes a una carpeta temporal para el DataLoader."""
    dst.mkdir(parents=True, exist_ok=True)
    storage, out = get_storage(), {}
    for s in samples:
        p = dst / f"{s.id}.wav"
        if not p.exists():
            with storage.local_path(s.audio_key, ".wav") as src:
                shutil.copyfile(src, p)
        out[s.id] = str(p)
    return out


def _train_emotion(db: Session, run: TrainingRun) -> None:
    ds = db.get(TrainingDataset, run.dataset_id)
    settings = get_settings()
    params = TrainParams.from_dict(run.params)
    fr = {"train": 1 - run.params.get("validation_split", 0.15) - run.params.get("test_split", 0.15),
          "validation": run.params.get("validation_split", 0.15), "test": run.params.get("test_split", 0.15)}
    usable = [s for s in ds.samples if s.emotion and s.audio_key and (s.validation or {}).get("ok", True)]
    if len(usable) < 6:
        raise AppError("El dataset no tiene suficientes muestras etiquetadas y válidas para entrenar (mínimo 6).", status_code=422)
    if run.params.get("resplit") or not all(s.split for s in usable):
        info = split_dataset(db, ds, fr, params.seed)
        db.refresh(ds)
        usable = [s for s in ds.samples if s.emotion and s.audio_key and (s.validation or {}).get("ok", True)]
        run.params = {**run.params, "split_info": info}
        db.commit()
    by = {sp: [s for s in usable if s.split == sp] for sp in ("train", "validation", "test")}
    if not by["train"] or not by["validation"]:
        raise AppError("No fue posible crear conjuntos de entrenamiento y validación con estos datos.", status_code=422)
    present = {s.emotion for s in usable}
    labels = [l for l in (ds.labels or sorted(present)) if l in present]

    base = db.get(MLModel, run.base_model_id)
    base_source = base.path or base.hf_id
    tmp = settings.temp_dir / f"train_{run.id}"
    try:
        paths = _materialize(db, usable, tmp)
        mk = lambda ss: [Sample(s.id, paths[s.id], s.emotion) for s in ss]
        # modelo del registry (se crea una vez; al continuar se reutiliza)
        model = db.get(MLModel, run.model_id) if run.model_id else None
        if model is None:
            model = register_model(db, org_id=run.org_id, kind="emotion", loader="finetuned", path=None,
                                   base_model=f"{base.name}:{base.version}", dataset_id=ds.id, params=run.params, labels=labels,
                                   language=ds.language, created_by=run.created_by, run_id=run.id)
            run.model_id = model.id
        out_dir = settings.models_dir / run.org_id / model.name
        model.path = str(out_dir / "best")
        model.parameters = {**run.params, "dataset": f"{ds.name}_v{ds.version}", "n_train": len(by["train"]),
                            "n_validation": len(by["validation"]), "n_test": len(by["test"])}
        ds.model_used = f"{base.name}:{base.version}"
        run.total_epochs = params.epochs
        db.commit()

        step_t = {"t": 0.0}

        def on_step(info: dict) -> None:
            if time.time() - step_t["t"] < 2:
                return
            step_t["t"] = time.time()
            s = SessionLocal()
            try:
                r = s.get(TrainingRun, run.id)
                r.progress = round(100 * info["step"] / max(info["total_steps"], 1), 1)
                r.current_epoch = info["epoch"]
                s.commit()
            finally:
                s.close()

        def on_epoch(info: dict) -> None:
            s = SessionLocal()
            try:
                r = s.get(TrainingRun, run.id)
                r.metrics_history = list(r.metrics_history or []) + [{k: v for k, v in info.items()}]
                r.current_epoch, r.best_metric = info["epoch"], info["best_macro_f1"]
                r.progress = round(100 * info["epoch"] / info["epochs"], 1)
                s.commit()
            finally:
                s.close()

        cb = Callbacks(on_step=on_step, on_epoch_end=on_epoch, should_stop=_stop_checker(run.id))
        trainer = EmotionTrainer(base_source, labels, params, out_dir, cb, str(settings.model_cache_dir), settings.hf_offline)
        res = trainer.fit(mk(by["train"]), mk(by["validation"]), mk(by["test"]) if by["test"] else None,
                          resume=(out_dir / "checkpoint.pt").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)         # no dejar audio de entrenamiento en disco

    db.expire_all()
    model = db.get(MLModel, run.model_id)
    add_metrics(db, model, "validation", res["validation"], run.id, ds.id)
    if res.get("test"):
        add_metrics(db, model, "test", res["test"], run.id, ds.id)
    model.labels = labels
    model.status = ModelStatus.VALIDATION.value
    recompute_stats(db, ds)
    audit(db, org_id=run.org_id, user_id=run.created_by, action="training.complete", entity="model", entity_id=model.id,
          details={"macro_f1_val": res["validation"]["macro_f1"]}, commit=False)
    db.commit()


# ---- satisfacción ------------------------------------------------------------------------------------------------
def call_speaker_features(db: Session, call: Call, speaker_label: str, cfg: dict) -> np.ndarray | None:
    from app.ml.prosody.features import build_baseline, tension_from_prosody
    from app.ml.satisfaction.engine import SatisfactionEngine
    from app.ml.satisfaction.features import build_series, call_features
    sp = next((s for s in call.speakers if s.label == speaker_label), None)
    if sp is None:
        return None
    preds = db.scalars(select(EmotionPrediction).where(EmotionPrediction.call_id == call.id,
                                                       EmotionPrediction.speaker_id == sp.id)
                       .order_by(EmotionPrediction.start)).all()
    if not preds:
        return None
    base = build_baseline([p.prosody for p in preds if p.prosody])
    wins = [{"start": p.start, "end": p.end, "probabilities": p.probabilities, "confidence": p.confidence,
             "tension": tension_from_prosody(p.prosody or {}, base)} for p in preds]
    utts = [{"start": t.start, "end": t.end, "sentiment": t.sentiment} for t in
            db.scalars(select(Transcription).where(Transcription.call_id == call.id, Transcription.speaker_id == sp.id))]
    series = build_series(speaker_label, list(preds[0].probabilities.keys()), wins, utts, 1.0)
    if series is None or series.n < 3:
        return None
    eng = SatisfactionEngine(cfg["satisfaction"], cfg["emotion"])
    return call_features(series, eng.components(series), cfg["emotion"])


def dataset_from_feedback(db: Session, user: User, ds: TrainingDataset) -> int:
    """Crea muestras de satisfacción a partir de la satisfacción REAL registrada (CSAT/encuesta) en las llamadas."""
    n = 0
    have = {(s.call_id, s.speaker) for s in ds.samples}
    for fb in db.scalars(select(SatisfactionFeedback).where(SatisfactionFeedback.org_id == user.org_id,
                                                            SatisfactionFeedback.real_satisfaction.is_not(None))):
        sp = db.get(Speaker, fb.speaker_id) if fb.speaker_id else None
        key = (fb.call_id, sp.label if sp else None)
        call = db.get(Call, fb.call_id)
        if key in have or call is None or not call.allow_training:
            continue
        db.add(TrainingSample(org_id=user.org_id, dataset_id=ds.id, call_id=fb.call_id, speaker=key[1],
                              speaker_group=fb.call_id, satisfaction=fb.real_satisfaction, validation={"ok": True, "issues": []},
                              meta={"source": "feedback"}, created_by=user.id))
        n += 1
    recompute_stats(db, ds)
    db.commit()
    return n


def _train_satisfaction(db: Session, run: TrainingRun) -> None:
    from app.ml.satisfaction.model import train_regressor
    ds = db.get(TrainingDataset, run.dataset_id)
    cfg = get_all(db, run.org_id)
    X, y, groups = [], [], []
    for s in ds.samples:
        if s.satisfaction is None or not s.call_id or not s.speaker:
            continue
        call = db.get(Call, s.call_id)
        if call is None or call.org_id != run.org_id or call.status != "COMPLETED":
            continue
        f = call_speaker_features(db, call, s.speaker, cfg)
        if f is not None:
            X.append(f); y.append(s.satisfaction); groups.append(s.speaker_group or s.call_id)
    if len(y) < 10:
        raise AppError("Se necesitan al menos 10 muestras de satisfacción con llamadas analizadas para entrenar.", status_code=422)
    reg, metrics = train_regressor(np.vstack(X), np.array(y), groups, kind=run.params.get("regressor", "ridge"))
    model = register_model(db, org_id=run.org_id, kind="satisfaction", loader="sklearn", path=None, base_model=None,
                           dataset_id=ds.id, params=run.params, labels=None, language=ds.language,
                           created_by=run.created_by, run_id=run.id)
    out = get_settings().models_dir / run.org_id / model.name / "model.joblib"
    reg.save(out)
    model.path = str(out)
    model.status = ModelStatus.VALIDATION.value
    add_metrics(db, model, "cv", {**metrics, "n_samples": metrics["n"]}, run.id, ds.id)
    run.model_id = model.id
    run.metrics_history = [{"epoch": 1, **metrics}]
    run.current_epoch, run.best_metric = 1, -metrics["mae"]
    db.commit()


# ---- evaluación de cualquier modelo sobre un dataset -----------------------------------------------------------------
def execute_evaluation(model_id: str, dataset_id: str, split: str = "test") -> None:
    """Evalúa un modelo del registry sobre el split de un dataset y guarda las métricas (comparación justa)."""
    db = SessionLocal()
    try:
        model, ds = db.get(MLModel, model_id), db.get(TrainingDataset, dataset_id)
        if model is None or ds is None or model.kind != "emotion":
            return
        samples = [s for s in ds.samples if s.emotion and s.audio_key and s.split == split and (s.validation or {}).get("ok", True)]
        if not samples:
            return
        emo = build_model(model, get_config_store().defaults("emotion"))
        import soundfile as sf
        y_true, y_pred = [], []
        storage = get_storage()
        for i in range(0, len(samples), 16):
            batch = samples[i:i + 16]
            waves = []
            for s in batch:
                with storage.local_path(s.audio_key, ".wav") as p:
                    x, _ = sf.read(p, dtype="float32")
                waves.append(x if x.ndim == 1 else x.mean(axis=1))
            for s, o in zip(batch, emo.predict(waves)):
                y_true.append(s.emotion); y_pred.append(o.emotion)
        m = metrics_from_predictions(y_true, y_pred, ds.labels or sorted(set(y_true)))
        add_metrics(db, model, split, m, None, ds.id)
        db.commit()
        log.info("evaluación completada", extra={"model": model.id, "dataset": ds.id, "macro_f1": m["macro_f1"]})
    except Exception:
        db.rollback()
        log.error("evaluación fallida", extra={"model": model_id, "trace": traceback.format_exc()})
    finally:
        db.close()
