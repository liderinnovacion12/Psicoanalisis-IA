"""Pipeline de análisis por ETAPAS independientes e idempotentes con checkpoints.

audio_processing → diarization → transcription → emotion_analysis → satisfaction → report

* Cada etapa es una función `stage_*(ctx)`; el mismo código lo ejecutan Celery (una cola por etapa) o el
  modo `inline` (hilo). Si una etapa ya está COMPLETED se omite: reanudar = volver a encolar la llamada.
* `emotion_analysis` guarda su avance por lote (mismo commit que las predicciones) y reanuda desde ahí.
* Los errores técnicos se registran internamente; el usuario solo ve mensajes genéricos.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import get_config_store, get_settings
from app.core.errors import ProcessingError
from app.core.logging import get_logger
from app.database.base import SessionLocal
from app.ml.audio import ffmpeg
from app.ml.audio.processing import load_vad, process_audio
from app.ml.diarization import DiarizationResult, Turn, get_diarizer
from app.ml.diarization.spectral import SpectralDiarizer
from app.ml.emotion.factory import build_model, get_production_model_row
from app.ml.emotion.inference import iter_window_results
from app.ml.emotion.preprocessing import plan_windows
from app.ml.privacy.pii import redact_text
from app.ml.prosody.features import build_baseline, tension_from_prosody
from app.ml.prosody.interaction import interaction_metrics
from app.ml.satisfaction.engine import SatisfactionEngine, confidence_level
from app.ml.satisfaction.events import detect_events
from app.ml.satisfaction.features import build_series
from app.ml.text.sentiment import get_text_analyzer
from app.ml.transcription.base import (RawTranscript, assign_words_to_speakers, group_utterances)
from app.ml.transcription.whisper import get_transcriber
from app.models import (AudioSegment, Call, CallStatus, CriticalEvent, EmotionPrediction, MLModel,
                        SatisfactionScore, Speaker, Transcription)
from app.services import narrative
from app.services.cache import cache_get, cache_put, config_hash
from app.services.config_service import get_all
from app.services.storage import call_key, get_storage

log = get_logger(__name__)

STAGES = ["audio_processing", "diarization", "transcription", "emotion_analysis", "satisfaction", "report"]
STAGE_STATUS = {
    "audio_processing": CallStatus.PROCESSING_AUDIO, "diarization": CallStatus.DIARIZING,
    "transcription": CallStatus.TRANSCRIBING, "emotion_analysis": CallStatus.ANALYZING_EMOTIONS,
    "satisfaction": CallStatus.CALCULATING_SATISFACTION, "report": CallStatus.GENERATING_REPORT,
}
STAGE_RANGE = {"audio_processing": (0, 15), "diarization": (15, 28), "transcription": (28, 52),
               "emotion_analysis": (52, 85), "satisfaction": (85, 95), "report": (95, 100)}
STAGE_LABELS = {"audio_processing": "Audio", "diarization": "Diarización", "transcription": "Transcripción",
                "emotion_analysis": "Emociones", "satisfaction": "Satisfacción", "report": "Informe"}


class CallGone(Exception):
    """La llamada fue eliminada durante el procesamiento."""


def _check_alive(db: Session, call: Call) -> None:
    if db.scalar(select(Call.id).where(Call.id == call.id)) is None:
        raise CallGone()


def work_dir(call_id: str) -> Path:
    return get_settings().data_dir / "work" / call_id


@dataclass
class StageCtx:
    db: Session
    call: Call
    stage: str
    cfg: dict
    wd: Path
    _last: float = 0.0
    _last_pct: float = -1.0

    def progress(self, pct: float, msg: str = "") -> None:
        pct = float(max(0, min(100, pct)))
        now = time.time()
        if pct < 100 and now - self._last < 1.5 and abs(pct - self._last_pct) < 2:
            return
        self._last, self._last_pct = now, pct
        _check_alive(self.db, self.call)
        set_checkpoint(self.db, self.call, self.stage, status="RUNNING", progress=round(pct, 1), message=msg)
        lo, hi = STAGE_RANGE[self.stage]
        self.call.progress = round(lo + (hi - lo) * pct / 100, 1)
        self.db.commit()


def set_checkpoint(db: Session, call: Call, stage: str, **fields) -> None:
    cps = dict(call.checkpoints or {})
    cur = dict(cps.get(stage, {}))
    cur.update(fields)
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    cps[stage] = cur
    call.checkpoints = cps


def stage_done(call: Call, stage: str) -> bool:
    return (call.checkpoints or {}).get(stage, {}).get("status") == "COMPLETED"


def invalidate_from(db: Session, call: Call, stage: str) -> None:
    """Marca como pendientes la etapa dada y las posteriores (p. ej. al re-analizar)."""
    cps = dict(call.checkpoints or {})
    for st in STAGES[STAGES.index(stage):]:
        cps.pop(st, None)
    call.checkpoints = cps


# ------------------------------------------------------------------------------------------------------
# Etapa 1: audio
# ------------------------------------------------------------------------------------------------------
def stage_audio(ctx: StageCtx) -> None:
    call, cfg = ctx.call, ctx.cfg["audio"]
    storage = get_storage()
    ctx.wd.mkdir(parents=True, exist_ok=True)
    row = get_production_model_row(ctx.db, call.org_id, "emotion")     # el modelo se fija al empezar (trazabilidad)
    call.model_id, call.model_version = row.id, f"{row.name}:{row.version}"
    with storage.local_path(call.storage_key, suffix=Path(call.filename).suffix) as src:
        res = process_audio(src, ctx.wd, cfg, ctx.progress)
        ctx.progress(97, "Generando copia de reproducción")
        pb = ctx.wd / "playback.mp3"
        ffmpeg.make_playback(src, pb)
    key = call_key(call.org_id, call.id, "playback.mp3")
    storage.put_file(key, pb)
    pb.unlink(missing_ok=True)
    (ctx.wd / "audio_result.json").write_text(json.dumps(res, default=str), encoding="utf-8")
    m = res["meta"]
    call.playback_key = key
    call.duration = float(res["duration"])
    call.sample_rate, call.channels, call.bitrate = m["sample_rate"], m["channels"], m["bitrate"]
    call.audio_quality = res["quality"]
    call.warnings = list(dict.fromkeys((call.warnings or []) + (
        ["El resultado puede presentar menor precisión debido a la calidad del audio."] if res["quality"]["low"] else [])
        + res["quality"]["warnings"]))
    set_checkpoint(ctx.db, call, "audio_processing", status="COMPLETED", progress=100, mode=res["mode"],
                   files=res["files"])


def _audio_result(wd: Path) -> dict:
    return json.loads((wd / "audio_result.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------
# Etapa 2: diarización
# ------------------------------------------------------------------------------------------------------
def stage_diarize(ctx: StageCtx) -> None:
    call, db = ctx.call, ctx.db
    cfg = ctx.cfg["audio"]
    ar, vad = _audio_result(ctx.wd), load_vad(ctx.wd)
    diarizer = get_diarizer(ar["mode"], cfg)
    ch = config_hash(diarizer.name, cfg["diarization"], cfg["segmentation"]["merge_gap"], cfg["segmentation"]["min_turn"],
                     cfg["vad"], cfg["channels"])
    cached = cache_get(call.org_id, call.file_hash, "diarization", ch)
    if cached:
        result = DiarizationResult.from_dict(cached)
        log.info("diarización desde caché", extra={"call_id": call.id})
    else:
        try:
            result = diarizer.diarize(ctx.wd, ar["files"], vad, cfg, ctx.progress)
        except Exception as e:
            if diarizer.name in ("spectral", "channels"):
                raise
            log.error("pyannote falló; se usa el respaldo espectral", extra={"call_id": call.id, "error": str(e)})
            result = SpectralDiarizer().diarize(ctx.wd, ar["files"], vad, cfg, ctx.progress)
            result.warnings.append("pyannote no pudo ejecutarse; se utilizó la diarización de respaldo.")
        cache_put(call.org_id, call.file_hash, "diarization", ch, result.to_dict())
    if not result.turns:
        raise ProcessingError("diarization", "no se detectó voz",
                              "No se detectó voz suficiente en la grabación para analizarla.")

    # reconstruir filas (idempotente)
    db.execute(delete(Speaker).where(Speaker.call_id == call.id))
    db.flush()
    speakers = {}
    for i, label in enumerate(("SPEAKER_00", "SPEAKER_01")):
        talk = sum(t.duration for t in result.turns if t.speaker == label)
        sp = Speaker(org_id=call.org_id, call_id=call.id, label=label, role="other",
                     channel=i if ar["mode"] == "stereo_split" else None, talk_time=round(talk, 2))
        db.add(sp)
        speakers[label] = sp
    db.flush()
    db.add_all([AudioSegment(org_id=call.org_id, call_id=call.id, speaker_id=speakers[t.speaker].id,
                             start=round(t.start, 3), end=round(t.end, 3), source=result.engine, overlap=t.overlap)
                for t in result.turns if t.speaker in speakers])
    call.diarization_mode = result.engine
    call.warnings = list(dict.fromkeys((call.warnings or []) + result.warnings))
    set_checkpoint(db, call, "diarization", status="COMPLETED", progress=100, engine=result.engine,
                   quality=result.quality, n_speakers_detected=result.n_speakers_detected, details=result.details)


# ------------------------------------------------------------------------------------------------------
# Etapa 3: transcripción (+ texto + PII)
# ------------------------------------------------------------------------------------------------------
def _speaker_maps(db: Session, call: Call):
    sps = db.scalars(select(Speaker).where(Speaker.call_id == call.id).order_by(Speaker.label)).all()
    return {s.label: s for s in sps}


def stage_transcribe(ctx: StageCtx) -> None:
    call, db, cfg = ctx.call, ctx.db, ctx.cfg
    ar, vad = _audio_result(ctx.wd), load_vad(ctx.wd)
    speakers = _speaker_maps(db, call)
    segs = db.scalars(select(AudioSegment).where(AudioSegment.call_id == call.id).order_by(AudioSegment.start)).all()
    id2label = {s.id: s.label for s in speakers.values()}
    turns = [Turn(id2label[s.speaker_id], s.start, s.end, s.overlap) for s in segs]
    transcriber = get_transcriber(cfg["audio"])
    tcfg = cfg["audio"]["transcription"]
    ch = config_hash(transcriber.name, tcfg, cfg["audio"]["vad"])
    files = ar["files"]
    keys = list(files) if ar["mode"] == "stereo_split" else ["mono"]
    lang = None if tcfg["language"] == "auto" else tcfg["language"]
    raws: dict[str, RawTranscript] = {}
    for i, key in enumerate(keys):
        cached = cache_get(call.org_id, call.file_hash, f"transcript_{key}", ch)
        if cached:
            raw = RawTranscript.from_dict(cached)
        else:
            def prog(p, m, i=i):
                ctx.progress((i * 100 + p) / len(keys), m)
            raw = transcriber.transcribe_file(ctx.wd / files[key], vad.get(key, []), cfg["audio"], lang, prog)
            cache_put(call.org_id, call.file_hash, f"transcript_{key}", ch, raw.to_dict())
        raws[key] = raw
        lang = lang or raw.language
    main = max(raws.values(), key=lambda r: len(r.words))
    call.language, call.language_confidence = (lang or main.language), main.language_confidence

    utts = []
    if ar["mode"] == "stereo_split":
        for i, key in enumerate(keys):
            label = f"SPEAKER_0{i}"
            utts += group_utterances([(label, w) for w in raws[key].words], call.language)
    else:
        utts = group_utterances(assign_words_to_speakers(raws["mono"].words, turns), call.language)

    analyzer = get_text_analyzer(cfg["emotion"])
    pii_entities = cfg["app"]["privacy"]["pii_entities"]
    seg_by_spk: dict[str, list[AudioSegment]] = {}
    for s in segs:
        seg_by_spk.setdefault(id2label[s.speaker_id], []).append(s)
    db.execute(delete(Transcription).where(Transcription.call_id == call.id))
    rows = []
    for u in utts:
        sig = analyzer.analyze(u.text, call.language)
        red, spans = redact_text(u.text, pii_entities)
        mid = (u.start + u.end) / 2
        seg = next((s for s in seg_by_spk.get(u.speaker, []) if s.start - 0.3 <= mid <= s.end + 0.3), None)
        rows.append(Transcription(
            org_id=call.org_id, call_id=call.id, speaker_id=speakers[u.speaker].id, segment_id=seg.id if seg else None,
            start=round(u.start, 3), end=round(u.end, 3), text=u.text, text_redacted=red, confidence=round(u.confidence, 4),
            language=call.language, words=[[round(w.start, 2), round(w.end, 2), w.word.strip(), round(w.prob, 3)] for w in u.words],
            sentiment=sig.to_dict(), pii=spans))
    db.add_all(rows)
    total_w = sum(len(u.words) for u in utts)
    mean_conf = float(np.average([u.confidence for u in utts], weights=[max(u.end - u.start, 0.1) for u in utts])) if utts else 0.0
    if total_w < 5:
        call.warnings = list(dict.fromkeys((call.warnings or []) + ["La transcripción contiene muy poco texto; el análisis contextual será limitado."]))
    set_checkpoint(db, call, "transcription", status="COMPLETED", progress=100, engine=transcriber.name,
                   model=main.model, utterances=len(utts), words=total_w, mean_confidence=round(mean_conf, 4),
                   language=call.language, language_confidence=call.language_confidence)


# ------------------------------------------------------------------------------------------------------
# Etapa 4: emociones (por ventanas, con lotes y reanudación)
# ------------------------------------------------------------------------------------------------------
def stage_emotions(ctx: StageCtx) -> None:
    call, db, cfg = ctx.call, ctx.db, ctx.cfg
    acfg, seg_cfg = cfg["audio"], cfg["audio"]["segmentation"]
    ar = _audio_result(ctx.wd)
    speakers = _speaker_maps(db, call)
    id2label = {s.id: s.label for s in speakers.values()}
    segs = db.scalars(select(AudioSegment).where(AudioSegment.call_id == call.id)
                      .order_by(AudioSegment.start, AudioSegment.end, AudioSegment.id)).all()
    turns = [Turn(id2label[s.speaker_id], s.start, s.end, s.overlap) for s in segs]
    windows = plan_windows(turns, seg_cfg["window_size"], seg_cfg["hop_size"], seg_cfg["min_window"])
    total = len(windows)
    mrow = db.get(MLModel, call.model_id)
    if mrow is None:
        raise ProcessingError("emotion_analysis", "modelo no encontrado", "El modelo de emociones seleccionado ya no existe.")
    cp = (call.checkpoints or {}).get("emotion_analysis", {})
    done = int(cp.get("done", 0)) if cp.get("total") == total else 0
    if done == 0:
        db.execute(delete(EmotionPrediction).where(EmotionPrediction.call_id == call.id))
        db.commit()
    if total == 0:
        set_checkpoint(db, call, "emotion_analysis", status="COMPLETED", progress=100, done=0, total=0, valid=0)
        call.warnings = list(dict.fromkeys((call.warnings or []) + ["No hubo segmentos suficientes para el análisis emocional."]))
        return

    model = build_model(mrow, cfg["emotion"])
    ckey = config_hash(mrow.id, mrow.path or mrow.hf_id, mrow.version, seg_cfg["window_size"], seg_cfg["hop_size"],
                       seg_cfg["min_window"], [(t.speaker, round(t.start, 2), round(t.end, 2)) for t in turns],
                       cfg["emotion"]["prosody"])
    cached = cache_get(call.org_id, call.file_hash, "emotions", ckey) if done == 0 else None
    speaker_channel = ({s.label: f"ch{s.channel}" for s in speakers.values()} if ar["mode"] == "stereo_split"
                       else {s.label: "mono" for s in speakers.values()})
    bs = seg_cfg["batch_size"]
    valid = int(cp.get("valid", 0)) if done else 0
    cache_rows: list = [] if cached is None else cached

    def make_row(w, out, pros) -> EmotionPrediction | None:
        if out is None:
            return None
        return EmotionPrediction(
            org_id=call.org_id, call_id=call.id, segment_id=segs[w.turn_index].id, speaker_id=speakers[w.speaker].id,
            model_id=mrow.id, model_name=mrow.name, model_version=mrow.version, start=round(w.start, 3), end=round(w.end, 3),
            duration=round(w.duration, 3), emotion=out.emotion, confidence=round(out.confidence, 5),
            probabilities={k: round(v, 6) for k, v in out.probabilities.items()}, prosody=pros or None)

    if cached is not None:
        log.info("emociones desde caché", extra={"call_id": call.id})
        by_idx = {r["i"]: r for r in cached}
        rows = []
        for w in windows:
            r = by_idx.get(w.index)
            if r and r["probs"]:
                from app.ml.emotion.base import EmotionOutput
                rows.append(make_row(w, EmotionOutput(r["emotion"], r["conf"], r["probs"]), r.get("prosody")))
        db.add_all([r for r in rows if r])
        valid = sum(1 for r in rows if r)
        done = total
    else:
        for res in iter_window_results(model, windows, ctx.wd, ar["files"], speaker_channel, bs, done,
                                       cfg["emotion"]["prosody"]["enabled"], cfg["emotion"]["prosody"]):
            objs = []
            for r in res:
                row = make_row(r.window, r.output, r.prosody)
                if row:
                    objs.append(row)
                cache_rows.append({"i": r.window.index, "emotion": r.output.emotion if r.output else None,
                                   "conf": r.output.confidence if r.output else None,
                                   "probs": r.output.probabilities if r.output else None, "prosody": r.prosody})
            db.add_all(objs)
            done += len(res)
            valid += len(objs)
            set_checkpoint(db, call, "emotion_analysis", status="RUNNING", done=done, total=total, valid=valid,
                           progress=round(100 * done / total, 1), message=f"Ventanas {done}/{total}")
            lo, hi = STAGE_RANGE["emotion_analysis"]
            call.progress = round(lo + (hi - lo) * done / total, 1)
            _check_alive(db, call)
            db.commit()                                     # predicciones + checkpoint atómicos
        cache_put(call.org_id, call.file_hash, "emotions", ckey, cache_rows)
    set_checkpoint(db, call, "emotion_analysis", status="COMPLETED", progress=100, done=total, total=total,
                   valid=valid, model=f"{mrow.name}:{mrow.version}", device=getattr(model, "device", None))


# ------------------------------------------------------------------------------------------------------
# Etapa 5: satisfacción + eventos + métricas
# ------------------------------------------------------------------------------------------------------
def _load_regressor(db: Session, call: Call, mode: str):
    if mode == "rules":
        return None
    try:
        from app.ml.satisfaction.model import SatisfactionRegressor
        row = get_production_model_row(db, call.org_id, "satisfaction")
        return SatisfactionRegressor.load(Path(row.path))
    except Exception as e:
        log.warning("modelo de satisfacción no disponible; se usan reglas", extra={"error": str(e)})
        return None


def stage_satisfaction(ctx: StageCtx) -> None:
    call, db, cfg = ctx.call, ctx.db, ctx.cfg
    sat_cfg, emo_cfg, audio_cfg = cfg["satisfaction"], cfg["emotion"], cfg["audio"]
    speakers = _speaker_maps(db, call)
    id2label = {s.id: s.label for s in speakers.values()}
    dur = float(call.duration or 0)
    preds = db.scalars(select(EmotionPrediction).where(EmotionPrediction.call_id == call.id)
                       .order_by(EmotionPrediction.start)).all()
    trs = db.scalars(select(Transcription).where(Transcription.call_id == call.id).order_by(Transcription.start)).all()
    mrow = db.get(MLModel, call.model_id)
    labels = (mrow.labels if mrow and mrow.labels else emo_cfg["canonical_emotions"])
    if preds:
        labels = list(preds[0].probabilities.keys())

    series, utts, wps = {}, {}, {}
    for label, sp in speakers.items():
        p_sp = [p for p in preds if p.speaker_id == sp.id]
        base = build_baseline([p.prosody for p in p_sp if p.prosody])
        wins = [{"start": p.start, "end": p.end, "probabilities": p.probabilities, "confidence": p.confidence,
                 "tension": tension_from_prosody(p.prosody or {}, base)} for p in p_sp]
        utts[label] = [{"start": t.start, "end": t.end, "text": t.text, "sentiment": t.sentiment}
                       for t in trs if t.speaker_id == sp.id]
        wps[label] = sum(len(t.words or []) for t in trs if t.speaker_id == sp.id)
        series[label] = build_series(label, labels, wins, utts[label], 1.0)

    regressor = _load_regressor(db, call, sat_cfg.get("mode", "rules"))
    engine = SatisfactionEngine(sat_cfg, emo_cfg, regressor)
    aq = (call.audio_quality or {}).get("score")
    mismatch = bool(mrow and mrow.language and call.language and mrow.language != call.language)
    analyses = {lab: engine.analyze_speaker(series[lab], dur, aq, lab, mismatch) for lab in speakers}

    segs = db.scalars(select(AudioSegment).where(AudioSegment.call_id == call.id)).all()
    turns = [Turn(id2label[s.speaker_id], s.start, s.end, s.overlap) for s in segs]
    inter_metrics, overlaps = interaction_metrics(turns, dur, audio_cfg, wps)
    events = detect_events(series, analyses, utts, sat_cfg, emo_cfg, overlaps)

    # persistencia
    db.execute(delete(SatisfactionScore).where(SatisfactionScore.call_id == call.id))
    db.execute(delete(CriticalEvent).where(CriticalEvent.call_id == call.id))
    summary_sp = {}
    for lab, a in analyses.items():
        if a is None:
            continue
        db.add(SatisfactionScore(
            org_id=call.org_id, call_id=call.id, speaker_id=speakers[lab].id, scope="speaker", score=a.score,
            confidence=a.confidence, trend=a.trend, initial_score=a.initial_score, final_score=a.final_score,
            engine_version=a.details.get("engine_version"),
            details={"timeline": a.timeline, "components": a.components, "contributions": a.contributions,
                     "factors": a.factors, "metrics": a.metrics, "interpretation": a.interpretation, **a.details}))
        summary_sp[lab] = {"score": a.score, "confidence": a.confidence, "trend": a.trend,
                           "initial_score": a.initial_score, "final_score": a.final_score,
                           "interpretation": a.interpretation, "metrics": a.metrics, "factors": a.factors,
                           "components": a.components, "confidence_level": confidence_level(a.confidence)}
    ini = [a.initial_score for a in analyses.values() if a and a.initial_score is not None]
    fin = [a.final_score for a in analyses.values() if a and a.final_score is not None]
    interaction = {"initial": round(float(np.mean(ini)), 1) if ini else None,
                   "final": round(float(np.mean(fin)), 1) if fin else None}
    interaction["variation"] = (round(interaction["final"] - interaction["initial"], 1) if ini and fin else None)
    interaction["message"] = narrative.interaction_message(interaction["variation"], sat_cfg["trend"]["stable_delta"])
    scores = [a.score for a in analyses.values() if a]
    if scores:
        db.add(SatisfactionScore(org_id=call.org_id, call_id=call.id, speaker_id=None, scope="interaction",
                                 score=round(float(np.mean(scores)), 1),
                                 confidence=round(float(np.mean([a.confidence for a in analyses.values() if a])), 3),
                                 trend="improving" if (interaction["variation"] or 0) > sat_cfg["trend"]["stable_delta"]
                                 else "declining" if (interaction["variation"] or 0) < -sat_cfg["trend"]["stable_delta"] else "stable",
                                 initial_score=interaction["initial"], final_score=interaction["final"],
                                 details={"interaction": interaction}))
    for e in events:
        sp = speakers.get(e["speaker"]) if e["speaker"] else None
        db.add(CriticalEvent(org_id=call.org_id, call_id=call.id, speaker_id=sp.id if sp else None,
                             timestamp=e["timestamp"], end_timestamp=e["end_timestamp"], event_type=e["event_type"],
                             severity=e["severity"], confidence=e["confidence"], emotion=e["emotion"],
                             satisfaction=e["satisfaction"], description=e["description"], evidence=e["evidence"]))
    call.summary = {"speakers": summary_sp, "interaction": interaction, "events_count": len(events)}
    call.satisfaction_p1 = summary_sp.get("SPEAKER_00", {}).get("score")
    call.satisfaction_p2 = summary_sp.get("SPEAKER_01", {}).get("score")
    call.interaction = inter_metrics
    if not summary_sp:
        call.warnings = list(dict.fromkeys((call.warnings or []) + ["No hubo datos suficientes para calcular la satisfacción."]))
    set_checkpoint(db, call, "satisfaction", status="COMPLETED", progress=100, events=len(events),
                   mode=sat_cfg.get("mode", "rules"))


# ------------------------------------------------------------------------------------------------------
# Etapa 6: calidad del análisis + resumen ejecutivo
# ------------------------------------------------------------------------------------------------------
def stage_report(ctx: StageCtx) -> None:
    call, db, cfg = ctx.call, ctx.db, ctx.cfg
    aq_cfg = cfg["audio"]["analysis_quality"]
    w = aq_cfg["weights"]
    cps = call.checkpoints or {}
    audio_q = (call.audio_quality or {}).get("score", 0)
    speech = float(np.clip((call.audio_quality or {}).get("speech_ratio", 0) / 0.4, 0, 1)) * 100
    em = cps.get("emotion_analysis", {})
    segments = 100 * em.get("valid", 0) / em["total"] if em.get("total") else 0
    confs = [s["metrics"]["mean_model_confidence"] for s in (call.summary or {}).get("speakers", {}).values()]
    model_q = float(np.clip((np.mean(confs) - 0.3) / 0.5, 0, 1)) * 100 if confs else 0
    mrow = db.get(MLModel, call.model_id)
    warnings = list(call.warnings or [])
    if mrow and mrow.language and call.language and mrow.language != call.language:
        model_q *= aq_cfg.get("language_mismatch_penalty", 0.8)
        warnings.append(f"El modelo de emociones activo está entrenado para '{mrow.language}' y la llamada está en "
                        f"'{call.language}': el rendimiento puede ser inferior. Considere un modelo específico del idioma.")
    diar_q = cps.get("diarization", {}).get("quality", 0)
    tr_q = 100 * cps.get("transcription", {}).get("mean_confidence", 0)
    comps = {"audio": audio_q, "speech": speech, "segments": segments, "model": model_q, "diarization": diar_q,
             "transcription": tr_q}
    score = sum(comps[k] * w[k] for k in comps) / sum(w.values())
    low = score < aq_cfg["low_threshold"]
    if low:
        warnings.append("El resultado puede presentar menor precisión debido a la calidad del análisis.")
    call.warnings = list(dict.fromkeys(warnings))
    call.analysis_quality = {"score": round(score, 1), "components": {k: round(v, 1) for k, v in comps.items()},
                             "low": low, "note": ("Estimación heurística basada en calidad de audio, cobertura y confianza "
                                                  "de los modelos; no equivale a la exactitud real.")}

    events = [{"event_type": e.event_type} for e in db.scalars(select(CriticalEvent).where(CriticalEvent.call_id == call.id))]
    names = {s.label: narrative.person_name(s.label, s.role, cfg["app"]["general"]["roles_labels"])
             for s in call.speakers}
    if call.summary and call.summary.get("speakers"):
        s = dict(call.summary)
        s["executive_summary"] = narrative.executive_summary(s["speakers"], s["interaction"], events, names)
        call.summary = s
    set_checkpoint(db, call, "report", status="COMPLETED", progress=100)


STAGE_FUNCS: dict[str, Callable[[StageCtx], None]] = {
    "audio_processing": stage_audio, "diarization": stage_diarize, "transcription": stage_transcribe,
    "emotion_analysis": stage_emotions, "satisfaction": stage_satisfaction, "report": stage_report,
}


# ------------------------------------------------------------------------------------------------------
# Ejecución de etapas
# ------------------------------------------------------------------------------------------------------
def _finalize(db: Session, call: Call) -> None:
    call.status = CallStatus.COMPLETED.value
    call.progress = 100.0
    call.completed_at = datetime.now(timezone.utc)
    call.error_code = call.error_detail = None
    db.commit()
    app_cfg = get_all(db, call.org_id)["app"]
    d = app_cfg["privacy"]["retention_days"]
    if d:
        from datetime import timedelta
        call.retention_until = datetime.now(timezone.utc) + timedelta(days=d)
    if app_cfg["privacy"]["encrypt_at_rest"] or not app_cfg["storage"]["keep_normalized"]:
        shutil.rmtree(work_dir(call.id), ignore_errors=True)       # no dejar audio en claro
    db.commit()


def run_stage(call_id: str, stage: str) -> bool:
    """Ejecuta una etapa. True si terminó (o ya estaba completa); False si falló o la llamada ya no existe."""
    db = SessionLocal()
    try:
        call = db.get(Call, call_id)
        if call is None:
            return False
        if stage_done(call, stage):
            return True
        cfg = call.config_snapshot
        if not cfg or stage == "audio_processing":
            cfg = get_all(db, call.org_id)
            call.config_snapshot = cfg                                  # la configuración usada queda registrada
        call.status = STAGE_STATUS[stage].value
        call.error_code = None
        set_checkpoint(db, call, stage, status="RUNNING", progress=0, started_at=datetime.now(timezone.utc).isoformat())
        db.commit()
        ctx = StageCtx(db, call, stage, cfg, work_dir(call_id))
        t0 = time.time()
        log.info("etapa iniciada", extra={"call_id": call_id, "stage": stage})
        STAGE_FUNCS[stage](ctx)
        set_checkpoint(db, call, stage, seconds=round(time.time() - t0, 1))
        db.commit()
        log.info("etapa completada", extra={"call_id": call_id, "stage": stage, "seconds": round(time.time() - t0, 1)})
        if stage == STAGES[-1]:
            _finalize(db, call)
        return True
    except CallGone:
        log.info("llamada eliminada durante el procesamiento", extra={"call_id": call_id})
        shutil.rmtree(work_dir(call_id), ignore_errors=True)
        return False
    except Exception as e:
        db.rollback()
        log.error("etapa fallida", extra={"call_id": call_id, "stage": stage, "trace": traceback.format_exc()})
        try:
            call = db.get(Call, call_id)
            if call:
                call.status = CallStatus.ERROR.value
                call.error_code = f"{stage}_failed"
                call.error_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
                if isinstance(e, ProcessingError):
                    call.error_code = f"{stage}:{e.code}:{e.user_message}"
                set_checkpoint(db, call, stage, status="FAILED")
                db.commit()
        except Exception:  # pragma: no cover
            db.rollback()
        return False
    finally:
        db.close()


def run_pipeline(call_id: str) -> bool:
    """Ejecuta (o reanuda) todas las etapas en secuencia (modo inline)."""
    for st in STAGES:
        if not run_stage(call_id, st):
            return False
    return True
