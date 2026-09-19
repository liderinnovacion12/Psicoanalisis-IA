"""Gestión de datasets: alta de muestras (desde llamadas, subida o CSV), validación, splits, estadísticas, versionado."""
from __future__ import annotations

import csv
import io
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, Conflict, NotFound
from app.core.logging import get_logger
from app.ml.audio import ffmpeg
from app.ml.training.dataset import class_distribution, imbalance_report, validate_audio
from app.ml.training.splits import group_stratified_split
from app.models import Call, MLModel, Setting, Speaker, TrainingDataset, TrainingSample, User
from app.services.audit import audit
from app.services.config_service import get_all
from app.services.storage import get_storage
from app.workers.pipeline import work_dir

log = get_logger(__name__)
CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def custom_labels(db: Session, org_id: str) -> list[str]:
    row = db.scalar(select(Setting).where(Setting.org_id == org_id, Setting.key == "custom_labels"))
    return list((row.value or {}).get("labels", [])) if row else []


def all_labels(db: Session, org_id: str) -> list[str]:
    return CANONICAL + [l for l in custom_labels(db, org_id) if l not in CANONICAL]


def add_custom_label(db: Session, org_id: str, label: str, user_id: str | None) -> list[str]:
    label = label.strip().lower().replace(" ", "_")
    if not label.isidentifier():
        raise AppError("La categoría solo puede contener letras, números y guiones bajos.", status_code=422)
    labels = custom_labels(db, org_id)
    if label not in CANONICAL and label not in labels:
        labels.append(label)
        row = db.scalar(select(Setting).where(Setting.org_id == org_id, Setting.key == "custom_labels"))
        if row:
            row.value = {"labels": labels}
        else:
            db.add(Setting(org_id=org_id, key="custom_labels", value={"labels": labels}, updated_by=user_id))
        db.commit()
    return all_labels(db, org_id)


def get_dataset(db: Session, user: User, dataset_id: str) -> TrainingDataset:
    ds = db.get(TrainingDataset, dataset_id)
    if ds is None or ds.org_id != user.org_id:
        raise NotFound("El dataset no existe.")
    return ds


def create_dataset(db: Session, user: User, name: str, description: str | None, language: str | None,
                   labels: list[str] | None) -> TrainingDataset:
    ds = TrainingDataset(org_id=user.org_id, name=name, version=1, description=description, language=language,
                         labels=labels or all_labels(db, user.org_id), created_by=user.id, stats={})
    db.add(ds)
    db.flush()
    audit(db, org_id=user.org_id, user_id=user.id, action="dataset.create", entity="dataset", entity_id=ds.id, commit=False)
    db.commit()
    return ds


def clone_version(db: Session, ds: TrainingDataset, user: User) -> TrainingDataset:
    """dataset_v1 → dataset_v2: copia los registros (el audio se comparte por referencia)."""
    top = db.scalar(select(TrainingDataset.version).where(TrainingDataset.org_id == ds.org_id, TrainingDataset.name == ds.name)
                    .order_by(TrainingDataset.version.desc()))
    new = TrainingDataset(org_id=ds.org_id, name=ds.name, version=(top or ds.version) + 1, parent_id=ds.id,
                          description=ds.description, labels=list(ds.labels or []), language=ds.language,
                          created_by=user.id, stats={})
    db.add(new)
    db.flush()
    for s in ds.samples:
        db.add(TrainingSample(org_id=s.org_id, dataset_id=new.id, call_id=s.call_id, audio_key=s.audio_key,
                              source_audio=s.source_audio, speaker=s.speaker, speaker_group=s.speaker_group,
                              emotion=s.emotion, satisfaction=s.satisfaction, start=s.start, end=s.end,
                              duration=s.duration, language=s.language, audio_hash=s.audio_hash, split=s.split,
                              validation=s.validation, meta=s.meta, created_by=user.id))
    db.flush()
    recompute_stats(db, new)
    audit(db, org_id=ds.org_id, user_id=user.id, action="dataset.version", entity="dataset", entity_id=new.id,
          details={"parent": ds.id}, commit=False)
    db.commit()
    return new


def _tmpfile(suffix: str) -> Path:
    """Archivo temporal cerrado (mkstemp devuelve un descriptor abierto que hay que cerrar)."""
    import os
    fd, name = tempfile.mkstemp(suffix=suffix, dir=get_settings().temp_dir)
    os.close(fd)
    return Path(name)


def _sample_key(org_id: str, dataset_id: str, sample_id: str) -> str:
    return f"orgs/{org_id}/datasets/{dataset_id}/{sample_id}.wav"


def _extract_from_call(call: Call, speaker_label: str | None, start: float, end: float, dst: Path, db: Session) -> None:
    """Recorta [start,end] del audio del hablante (canal propio si es estéreo) a WAV 16 kHz mono."""
    sp = next((s for s in call.speakers if s.label == speaker_label), None) if speaker_label else None
    cps = (call.checkpoints or {}).get("audio_processing", {})
    files = cps.get("files") or {}
    wd = work_dir(call.id)
    key = (f"ch{sp.channel}" if sp is not None and sp.channel is not None and f"ch{sp.channel}" in files else "mono")
    local = wd / files[key] if key in files and (wd / files[key]).exists() else None
    if local:
        with sf.SoundFile(str(local)) as f:
            f.seek(int(start * f.samplerate))
            x = f.read(int((end - start) * f.samplerate), dtype="float32", always_2d=True).mean(axis=1)
            sf.write(dst, x, f.samplerate, subtype="PCM_16")
        return
    if not call.storage_key:
        raise Conflict("El audio original de esta llamada ya no está disponible.")
    with get_storage().local_path(call.storage_key, suffix=Path(call.storage_key).suffix) as src:
        ffmpeg.cut_segment(src, dst, start, end, channel=sp.channel if sp is not None and sp.channel is not None
                           and call.diarization_mode == "channels" else None)


def add_sample(db: Session, user: User, ds: TrainingDataset, *, call_id: str | None = None, speaker: str | None = None,
               emotion: str | None = None, satisfaction: float | None = None, start: float = 0.0, end: float | None = None,
               language: str | None = None, speaker_group: str | None = None, meta: dict | None = None,
               upload: tuple[str, BinaryIO] | None = None, commit: bool = True) -> TrainingSample:
    if ds.frozen:
        raise Conflict("El dataset está congelado. Cree una nueva versión para modificarlo.")
    if emotion and ds.labels and emotion not in ds.labels:
        raise AppError(f"La categoría '{emotion}' no existe en el dataset. Agréguela primero como categoría personalizada.",
                       status_code=422)
    cfg = get_all(db, user.org_id)["app"]["training"]
    storage = get_storage()
    sample = TrainingSample(org_id=user.org_id, dataset_id=ds.id, call_id=call_id, speaker=speaker, emotion=emotion,
                            satisfaction=satisfaction, start=0.0, language=language, meta=meta or {}, created_by=user.id)
    db.add(sample)
    db.flush()
    tmp = _tmpfile(".wav")
    try:
        if call_id:
            call = db.get(Call, call_id)
            if call is None or call.org_id != user.org_id:
                raise NotFound("La llamada no existe.")
            if not call.allow_training:
                raise Conflict("Esta llamada no tiene habilitada la opción «Permitir utilizar esta llamada para mejorar el modelo».")
            if end is None:
                raise AppError("Indique el inicio y fin del segmento.", status_code=422)
            _extract_from_call(call, speaker, start, end, tmp, db)
            sample.speaker_group = speaker_group or call_id
            sample.language = language or call.language
            sample.start, sample.end = 0.0, None                  # el recorte ya está extraído
            sample.source_audio = f"call:{call_id}:{speaker}:{start:.2f}-{end:.2f}"
        elif upload:
            fname, stream = upload
            raw = _tmpfile(Path(fname).suffix or ".wav")
            try:
                with open(raw, "wb") as f:
                    shutil.copyfileobj(stream, f, 1024 * 1024)
                ffmpeg.probe(raw)
                if end is not None:
                    ffmpeg.cut_segment(raw, tmp, start, end)
                else:
                    ffmpeg.to_wav(raw, tmp, sample_rate=16000, channels=1)
            finally:
                raw.unlink(missing_ok=True)
            sample.speaker_group = speaker_group or speaker or sample.id
            sample.source_audio = f"upload:{fname}"
        else:
            raise AppError("Indique una llamada o un archivo de audio.", status_code=422)
        v = validate_audio(str(tmp), 0.0, None, cfg["min_sample_seconds"], cfg["max_sample_seconds"])
        sample.duration = v["duration"]
        sample.audio_hash = v["hash"]
        issues = list(v["issues"])
        if v["hash"] and db.scalar(select(TrainingSample.id).where(TrainingSample.dataset_id == ds.id,
                                                                     TrainingSample.audio_hash == v["hash"],
                                                                     TrainingSample.id != sample.id)):
            issues.append("duplicado")
        if not emotion and satisfaction is None:
            issues.append("etiqueta_faltante")
        sample.validation = {"ok": not issues, "issues": issues}
        sample.audio_key = _sample_key(user.org_id, ds.id, sample.id)
        storage.put_file(sample.audio_key, tmp)
    except Exception:
        db.rollback()
        raise
    finally:
        tmp.unlink(missing_ok=True)
    if commit:
        recompute_stats(db, ds)
        db.commit()
    return sample


def import_rows(db: Session, user: User, ds: TrainingDataset, rows: list[dict], base_dir: Path) -> dict:
    """Importa filas de un manifiesto CSV (columnas: audio, speaker, emotion, satisfaction, start, end, language,
    speaker_group). `audio` es una ruta relativa a base_dir."""
    ok, errors = 0, []
    for i, r in enumerate(rows, start=2):
        try:
            audio = (base_dir / r["audio"].strip()).resolve()
            if base_dir.resolve() not in audio.parents:
                raise AppError("Ruta de audio fuera de la carpeta permitida.", status_code=422)
            if not audio.exists():
                raise AppError("Archivo de audio no encontrado.", status_code=422)
            f = lambda k: (r.get(k) or "").strip() or None
            with open(audio, "rb") as fh:
                add_sample(db, user, ds, speaker=f("speaker"), emotion=(f("emotion") or "").lower() or None,
                           satisfaction=float(f("satisfaction")) if f("satisfaction") else None,
                           start=_ts(f("start")) or 0.0, end=_ts(f("end")), language=f("language"),
                           speaker_group=f("speaker_group"), upload=(audio.name, fh), commit=False)
            ok += 1
        except Exception as e:
            db.rollback()
            errors.append({"row": i, "error": getattr(e, "user_message", None) or "Fila inválida."})
            log.warning("import fila fallida", extra={"row": i, "error": str(e)})
    recompute_stats(db, ds)
    db.commit()
    return {"imported": ok, "errors": errors[:200], "error_count": len(errors)}


def _ts(v: str | None) -> float | None:
    """Acepta segundos ('83.5') o 'HH:MM:SS' / 'MM:SS'."""
    if not v:
        return None
    if ":" in v:
        parts = [float(p) for p in v.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(v)


def parse_csv(stream: BinaryIO) -> list[dict]:
    text = stream.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


# ---- validación / stats / split ------------------------------------------------------------------------------------
def validate_dataset(db: Session, ds: TrainingDataset) -> dict:
    cfg = get_all(db, ds.org_id)["app"]["training"]
    storage = get_storage()
    issues_count: Counter = Counter()
    seen: dict[str, str] = {}
    for s in ds.samples:
        issues: list[str] = []
        if s.audio_key and storage.exists(s.audio_key):
            with storage.local_path(s.audio_key, ".wav") as p:
                v = validate_audio(str(p), 0.0, None, cfg["min_sample_seconds"], cfg["max_sample_seconds"])
            issues += v["issues"]
            s.duration, s.audio_hash = v["duration"], v["hash"]
        elif s.audio_key or not s.call_id:
            issues.append("archivo_corrupto")
        if not s.emotion and s.satisfaction is None:
            issues.append("etiqueta_faltante")
        if s.emotion and ds.labels and s.emotion not in ds.labels:
            issues.append("inconsistencia_categoria")
        if s.satisfaction is not None and not (0 <= s.satisfaction <= 100):
            issues.append("inconsistencia_satisfaccion")
        if s.emotion == "happy" and s.satisfaction is not None and s.satisfaction < 15:
            issues.append("posible_inconsistencia_emocion_satisfaccion")   # aviso: no es necesariamente un error
        if s.emotion == "angry" and s.satisfaction is not None and s.satisfaction > 90:
            issues.append("posible_inconsistencia_emocion_satisfaccion")
        if s.audio_hash:
            if s.audio_hash in seen:
                issues.append("duplicado")
            seen[s.audio_hash] = s.id
        s.validation = {"ok": not [i for i in issues if not i.startswith("posible")], "issues": issues}
        issues_count.update(issues)
    db.flush()
    recompute_stats(db, ds)
    db.commit()
    return {"issues": dict(issues_count), "samples": len(ds.samples),
            "invalid": sum(1 for s in ds.samples if not (s.validation or {}).get("ok", True))}


def recompute_stats(db: Session, ds: TrainingDataset) -> dict:
    db.flush()
    samples = db.scalars(select(TrainingSample).where(TrainingSample.dataset_id == ds.id)).all()
    cfg = get_all(db, ds.org_id)["app"]["training"]
    emo = class_distribution([s.emotion for s in samples])
    spk = Counter(s.speaker or "n/d" for s in samples)
    sat_bins = Counter()
    for s in samples:
        if s.satisfaction is not None:
            sat_bins["0-20" if s.satisfaction <= 20 else "21-40" if s.satisfaction <= 40 else "41-60" if s.satisfaction <= 60
                     else "61-80" if s.satisfaction <= 80 else "81-100"] += 1
    durs = [s.duration for s in samples if s.duration]
    total = float(sum(durs))
    issues = Counter(i for s in samples for i in (s.validation or {}).get("issues", []))
    ds.n_samples, ds.total_duration = len(samples), round(total, 2)
    ds.stats = {
        "emotions": emo, "speakers": dict(spk), "satisfaction": dict(sat_bins),
        "avg_duration": round(total / len(durs), 3) if durs else 0.0, "total_duration": round(total, 2),
        "imbalance": imbalance_report(emo, cfg["imbalance_alert_ratio"]),
        "issues": dict(issues), "unlabeled": sum(1 for s in samples if not s.emotion),
        "splits": dict(Counter(s.split or "sin_asignar" for s in samples)),
        "groups": len({s.speaker_group or s.call_id or s.id for s in samples}),
    }
    return ds.stats


def split_dataset(db: Session, ds: TrainingDataset, fractions: dict[str, float], seed: int = 42) -> dict:
    samples = [s for s in ds.samples if s.emotion and (s.validation or {}).get("ok", True)]
    if len(samples) < 3:
        raise Conflict("Se necesitan más muestras etiquetadas válidas para crear los conjuntos.")
    assign, info = group_stratified_split(
        [{"id": s.id, "label": s.emotion, "group": s.speaker_group or s.call_id} for s in samples], fractions, seed)
    for s in ds.samples:
        s.split = assign.get(s.id)
    recompute_stats(db, ds)
    db.commit()
    return info


def delete_sample(db: Session, ds: TrainingDataset, sample_id: str) -> None:
    s = db.get(TrainingSample, sample_id)
    if s is None or s.dataset_id != ds.id:
        raise NotFound("La muestra no existe.")
    if ds.frozen:
        raise Conflict("El dataset está congelado.")
    key = s.audio_key
    db.delete(s)
    db.flush()
    if key and not db.scalar(select(TrainingSample.id).where(TrainingSample.audio_key == key)):
        get_storage().delete(key)             # solo si ningún otro dataset/versión lo referencia
    recompute_stats(db, ds)
    db.commit()


def delete_dataset(db: Session, ds: TrainingDataset) -> None:
    keys = {s.audio_key for s in ds.samples if s.audio_key}
    db.delete(ds)
    db.flush()
    for k in keys:
        if not db.scalar(select(TrainingSample.id).where(TrainingSample.audio_key == k)):
            get_storage().delete(k)
    db.commit()


def sample_out(s: TrainingSample) -> dict:
    return {"id": s.id, "call_id": s.call_id, "speaker": s.speaker, "speaker_group": s.speaker_group, "emotion": s.emotion,
            "satisfaction": s.satisfaction, "duration": s.duration, "language": s.language, "split": s.split,
            "validation": s.validation, "source": s.source_audio, "created_at": s.created_at.isoformat()}


def dataset_out(ds: TrainingDataset, detail: bool = False) -> dict:
    d = {"id": ds.id, "name": ds.name, "version": ds.version, "code": f"{ds.name}_v{ds.version}", "parent_id": ds.parent_id,
         "description": ds.description, "language": ds.language, "labels": ds.labels, "frozen": ds.frozen,
         "n_samples": ds.n_samples, "total_duration": ds.total_duration, "model_used": ds.model_used,
         "created_at": ds.created_at.isoformat(), "created_by": ds.created_by}
    if detail:
        d["stats"] = ds.stats
    else:
        d["stats"] = {"emotions": (ds.stats or {}).get("emotions", {}), "imbalance": (ds.stats or {}).get("imbalance")}
    return d
