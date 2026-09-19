"""Datasets, muestras etiquetadas, categorías personalizadas, validación, splits, versionado e importación."""
from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AnalystUser, AnyUser
from app.core.config import get_settings
from app.core.errors import AppError, Conflict
from app.database.base import get_db
from app.models import TrainingDataset, TrainingSample, User
from app.schemas.common import DatasetCreate, LabelIn, Msg, SampleIn, SplitIn
from app.services import dataset_service as dsvc
from app.services import training_service as tsvc
from app.services.audit import audit
from app.services.storage import get_storage

router = APIRouter(tags=["datasets"])


@router.get("/labels")
def list_labels(user: User = AnyUser, db: Session = Depends(get_db)):
    return {"labels": dsvc.all_labels(db, user.org_id), "custom": dsvc.custom_labels(db, user.org_id)}


@router.post("/labels", status_code=201)
def add_label(body: LabelIn, user: User = AnalystUser, db: Session = Depends(get_db)):
    return {"labels": dsvc.add_custom_label(db, user.org_id, body.label, user.id)}


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetCreate, user: User = AnalystUser, db: Session = Depends(get_db)):
    return dsvc.dataset_out(dsvc.create_dataset(db, user, body.name, body.description, body.language, body.labels), True)


@router.get("/datasets")
def list_datasets(user: User = AnyUser, db: Session = Depends(get_db)):
    rows = db.scalars(select(TrainingDataset).where(TrainingDataset.org_id == user.org_id)
                      .order_by(TrainingDataset.name, TrainingDataset.version.desc())).all()
    return [dsvc.dataset_out(d) for d in rows]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    return dsvc.dataset_out(dsvc.get_dataset(db, user, dataset_id), True)


@router.delete("/datasets/{dataset_id}", response_model=Msg)
def delete_dataset(dataset_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    dsvc.delete_dataset(db, ds)
    audit(db, org_id=user.org_id, user_id=user.id, action="dataset.delete", entity="dataset", entity_id=dataset_id)
    return Msg(message="Dataset eliminado.")


@router.post("/datasets/{dataset_id}/version", status_code=201)
def new_version(dataset_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    return dsvc.dataset_out(dsvc.clone_version(db, dsvc.get_dataset(db, user, dataset_id), user), True)


@router.post("/datasets/{dataset_id}/freeze", response_model=Msg)
def freeze(dataset_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    ds.frozen = True
    db.commit()
    return Msg(message="Dataset congelado: ya no admite cambios (cree una nueva versión para editarlo).")


@router.get("/datasets/{dataset_id}/samples")
def list_samples(dataset_id: str, split: str | None = None, emotion: str | None = None, issues: bool = False,
                 page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                 user: User = AnyUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    stmt = select(TrainingSample).where(TrainingSample.dataset_id == ds.id)
    if split:
        stmt = stmt.where(TrainingSample.split == split)
    if emotion:
        stmt = stmt.where(TrainingSample.emotion == emotion)
    rows = db.scalars(stmt.order_by(TrainingSample.created_at.desc())).all()
    if issues:
        rows = [r for r in rows if (r.validation or {}).get("issues")]
    total = len(rows)
    rows = rows[(page - 1) * page_size: page * page_size]
    return {"items": [dsvc.sample_out(s) for s in rows], "total": total, "page": page, "page_size": page_size}


@router.post("/datasets/{dataset_id}/samples", status_code=201)
def add_sample(dataset_id: str, body: SampleIn, user: User = AnalystUser, db: Session = Depends(get_db)):
    """Agrega una muestra etiquetada a partir de un segmento de una llamada analizada."""
    ds = dsvc.get_dataset(db, user, dataset_id)
    s = dsvc.add_sample(db, user, ds, call_id=body.call_id, speaker=body.speaker,
                        emotion=(body.emotion or "").lower() or None, satisfaction=body.satisfaction, start=body.start,
                        end=body.end, language=body.language, speaker_group=body.speaker_group, meta=body.meta)
    return dsvc.sample_out(s)


@router.post("/datasets/{dataset_id}/samples/upload", status_code=201)
def add_sample_upload(dataset_id: str, file: UploadFile = File(...), emotion: str | None = Form(None),
                      satisfaction: float | None = Form(None), speaker: str | None = Form(None),
                      speaker_group: str | None = Form(None), language: str | None = Form(None),
                      start: float = Form(0.0), end: float | None = Form(None),
                      user: User = AnalystUser, db: Session = Depends(get_db)):
    """Agrega una muestra subiendo un archivo de audio (grabación propia etiquetada)."""
    ds = dsvc.get_dataset(db, user, dataset_id)
    s = dsvc.add_sample(db, user, ds, emotion=(emotion or "").lower() or None, satisfaction=satisfaction, speaker=speaker,
                        speaker_group=speaker_group, language=language, start=start, end=end,
                        upload=(file.filename or "audio.wav", file.file))
    return dsvc.sample_out(s)


@router.patch("/datasets/{dataset_id}/samples/{sample_id}")
def update_sample(dataset_id: str, sample_id: str, body: SampleIn, user: User = AnalystUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    if ds.frozen:
        raise Conflict("El dataset está congelado.")
    s = db.get(TrainingSample, sample_id)
    if s is None or s.dataset_id != ds.id:
        raise AppError("La muestra no existe.", status_code=404)
    if body.emotion is not None:
        e = body.emotion.lower()
        if ds.labels and e not in ds.labels:
            raise AppError("La categoría no existe en el dataset.", status_code=422)
        s.emotion = e
    if body.satisfaction is not None:
        s.satisfaction = body.satisfaction
    if body.speaker_group is not None:
        s.speaker_group = body.speaker_group
    dsvc.recompute_stats(db, ds)
    db.commit()
    return dsvc.sample_out(s)


@router.delete("/datasets/{dataset_id}/samples/{sample_id}", response_model=Msg)
def delete_sample(dataset_id: str, sample_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    dsvc.delete_sample(db, dsvc.get_dataset(db, user, dataset_id), sample_id)
    return Msg(message="Muestra eliminada.")


@router.get("/datasets/{dataset_id}/samples/{sample_id}/audio")
def sample_audio(dataset_id: str, sample_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    s = db.get(TrainingSample, sample_id)
    if s is None or s.dataset_id != ds.id or not s.audio_key:
        raise AppError("La muestra no tiene audio.", status_code=404)
    st = get_storage()
    return StreamingResponse(st.open_range(s.audio_key), media_type="audio/wav", headers={"Content-Length": str(st.size(s.audio_key))})


@router.post("/datasets/{dataset_id}/validate")
def validate(dataset_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    return dsvc.validate_dataset(db, dsvc.get_dataset(db, user, dataset_id))


@router.post("/datasets/{dataset_id}/split")
def split(dataset_id: str, body: SplitIn, user: User = AnalystUser, db: Session = Depends(get_db)):
    if min(body.train, body.validation, body.test) < 0 or body.train <= 0 or body.validation <= 0:
        raise AppError("Los porcentajes de los conjuntos no son válidos.", status_code=422)
    info = dsvc.split_dataset(db, dsvc.get_dataset(db, user, dataset_id),
                              {"train": body.train, "validation": body.validation, "test": body.test}, body.seed)
    return info


@router.post("/datasets/{dataset_id}/import")
def import_manifest(dataset_id: str, manifest: UploadFile = File(..., description="CSV: audio,speaker,emotion,satisfaction,start,end,language,speaker_group"),
                    audio_zip: UploadFile | None = File(None), user: User = AnalystUser, db: Session = Depends(get_db)):
    """Importa datos propios: un CSV de manifiesto y un ZIP con los audios referenciados (rutas relativas)."""
    ds = dsvc.get_dataset(db, user, dataset_id)
    rows = dsvc.parse_csv(manifest.file)
    with tempfile.TemporaryDirectory(dir=get_settings().temp_dir) as td:
        base = Path(td)
        if audio_zip is not None:
            with zipfile.ZipFile(audio_zip.file) as z:
                for m in z.infolist():                    # anti zip-slip
                    target = (base / m.filename).resolve()
                    if base.resolve() not in target.parents and target != base.resolve():
                        raise AppError("El archivo ZIP contiene rutas no válidas.", status_code=422)
                z.extractall(base)
        res = dsvc.import_rows(db, user, ds, rows, base)
    audit(db, org_id=user.org_id, user_id=user.id, action="dataset.import", entity="dataset", entity_id=ds.id,
          details={"imported": res["imported"], "errors": res["error_count"]})
    return res


@router.post("/datasets/{dataset_id}/import-feedback")
def import_feedback(dataset_id: str, user: User = AnalystUser, db: Session = Depends(get_db)):
    """Crea muestras de satisfacción a partir de la satisfacción real (CSAT/encuesta) registrada en llamadas."""
    n = tsvc.dataset_from_feedback(db, user, dsvc.get_dataset(db, user, dataset_id))
    return {"imported": n}


@router.get("/datasets/{dataset_id}/export")
def export_manifest(dataset_id: str, user: User = AnyUser, db: Session = Depends(get_db)):
    ds = dsvc.get_dataset(db, user, dataset_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sample_id", "audio", "speaker", "emotion", "satisfaction", "start", "end", "language", "speaker_group", "split", "duration"])
    for s in ds.samples:
        w.writerow([s.id, f"{s.id}.wav", s.speaker, s.emotion, s.satisfaction, 0, s.duration, s.language, s.speaker_group, s.split, s.duration])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{ds.name}_v{ds.version}.csv"'})
