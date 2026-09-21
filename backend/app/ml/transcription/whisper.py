"""Transcripción con faster-whisper (motor de WhisperX) + WhisperX opcional para alineación forzada."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from app.core.config import get_settings
from app.core.device import torch_device
from app.core.logging import get_logger
from app.ml.audio.io import WavReader
from app.ml.transcription.base import RawTranscript, Transcriber, Word, plan_blocks

log = get_logger(__name__)
_MODELS: dict = {}


def _compute_type(cfg_ct: str, device: str) -> str:
    if cfg_ct != "auto":
        return cfg_ct
    return "float16" if device == "cuda" else "int8"


def resolve_model_size(size: str, device: str) -> str:
    """'auto': con GPU el modelo grande (mucho más preciso en español); en CPU 'small' (equilibrio precisión/tiempo)."""
    return ("large-v3" if device == "cuda" else "small") if size == "auto" else size


def _load_model(size: str, compute_type: str):
    from faster_whisper import WhisperModel
    s = get_settings()
    device = torch_device(s.device)
    size = resolve_model_size(size, device)
    ct = _compute_type(compute_type, device)
    key = (size, device, ct)
    if key not in _MODELS:
        log.info("Cargando faster-whisper", extra={"size": size, "device": device, "compute_type": ct})
        _MODELS[key] = WhisperModel(size, device=device, compute_type=ct,
                                    download_root=str(s.model_cache_dir / "whisper"),
                                    local_files_only=s.hf_offline)
    return _MODELS[key]


def detect_language(model, reader: WavReader, vad: list[tuple[float, float]], cfg: dict) -> tuple[str, float]:
    """Vota el idioma con hasta 3 muestras de ~25 s de voz repartidas por la llamada."""
    speech = [(s, e) for s, e in vad if e - s >= 2.0]
    if not speech:
        return cfg.get("preferred_language", "es"), 0.0
    picks = [speech[int(i * (len(speech) - 1) / 2)] for i in range(3)] if len(speech) >= 3 else speech
    agg: dict[str, float] = {}
    for s, e in picks:
        audio = reader.read(s, min(e, s + 25))
        if len(audio) < 16000:
            continue
        _, _, probs = model.detect_language(audio)
        for lang, p in probs:
            agg[lang] = agg.get(lang, 0.0) + p
    if not agg:
        return cfg.get("preferred_language", "es"), 0.0
    total = sum(agg.values())
    lang, sc = max(agg.items(), key=lambda kv: kv[1])
    conf = sc / total
    pref = cfg.get("preferred_language", "es")
    if conf < cfg.get("language_confidence_min", 0.6) and pref in agg and agg[pref] / total >= 0.15:
        return pref, agg[pref] / total          # detección dudosa: priorizar español
    return lang, conf


class FasterWhisperTranscriber(Transcriber):
    name = "faster-whisper"

    def transcribe_file(self, wav_path: Path, vad, cfg: dict, language: str | None = None, progress=None):
        tcfg = cfg["transcription"]
        model = _load_model(tcfg["model_size"], tcfg["compute_type"])
        words: list[Word] = []
        lang_conf: float | None = None
        with WavReader(wav_path) as rd:
            lang = language if language and language != "auto" else (
                None if tcfg["language"] == "auto" else tcfg["language"])
            if lang is None:
                lang, lang_conf = detect_language(model, rd, vad, tcfg)
            blocks = plan_blocks(rd.duration, vad, tcfg["block_seconds"])
            for bi, (b0, b1) in enumerate(blocks):
                audio = rd.read(b0, b1)
                if len(audio) < 1600 or not _has_speech(vad, b0, b1):
                    if progress:
                        progress(100 * (bi + 1) / len(blocks), "Transcribiendo")
                    continue
                kw = {}
                if tcfg.get("anti_hallucination", True):
                    kw.update(no_speech_threshold=0.6, compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                              hallucination_silence_threshold=2.0, temperature=[0.0, 0.2, 0.4])
                prompt = (tcfg.get("initial_prompt") or {}).get(lang or "")
                if prompt:
                    kw["initial_prompt"] = prompt              # fija vocabulario/registro (p. ej. atención telefónica en español)
                segs, _ = model.transcribe(
                    audio, language=lang, word_timestamps=True, beam_size=tcfg["beam_size"],
                    vad_filter=tcfg["vad_filter"], condition_on_previous_text=False, **kw)
                blk = max(b1 - b0, 1e-6)
                for sg in segs:
                    if progress:                                   # avance real dentro del bloque (los segmentos llegan en orden)
                        progress(100 * (bi + min(sg.end / blk, 1.0)) / len(blocks), "Transcribiendo")
                    if getattr(sg, "no_speech_prob", 0) > 0.85 and getattr(sg, "avg_logprob", 0) < -1.0:
                        continue
                    for w in (sg.words or []):
                        words.append(Word(b0 + w.start, b0 + w.end, w.word, float(w.probability)))
                if progress:
                    progress(100 * (bi + 1) / len(blocks), "Transcribiendo")
        return RawTranscript(words, lang, lang_conf, self.name, f"whisper-{resolve_model_size(tcfg['model_size'], torch_device(get_settings().device))}",
                             {"blocks": len(blocks)})


def _has_speech(vad, b0, b1) -> bool:
    return any(e > b0 and s < b1 for s, e in vad) if vad else True


class WhisperXTranscriber(Transcriber):
    """WhisperX (transcripción por lotes + alineación forzada wav2vec2). Opcional: `pip install whisperx`.
    Escrito contra la API pública de WhisperX; no fue ejecutado en el entorno de desarrollo de esta entrega."""
    name = "whisperx"

    def transcribe_file(self, wav_path: Path, vad, cfg: dict, language: str | None = None, progress=None):
        import whisperx
        tcfg = cfg["transcription"]
        s = get_settings()
        device = torch_device(s.device)
        ct = _compute_type(tcfg["compute_type"], device)
        model = whisperx.load_model(tcfg["model_size"], device, compute_type=ct,
                                    download_root=str(s.model_cache_dir / "whisperx"),
                                    language=None if tcfg["language"] == "auto" else tcfg["language"])
        words: list[Word] = []
        lang_out, lang_conf = language, None
        with WavReader(wav_path) as rd:
            blocks = plan_blocks(rd.duration, vad, tcfg["block_seconds"])
            align_cache = {}
            for bi, (b0, b1) in enumerate(blocks):
                audio = rd.read(b0, b1).astype(np.float32)
                if not _has_speech(vad, b0, b1):
                    continue
                res = model.transcribe(audio, batch_size=8, language=lang_out)
                lang_out = lang_out or res["language"]
                if lang_out not in align_cache:
                    align_cache[lang_out] = whisperx.load_align_model(language_code=lang_out, device=device)
                am, meta = align_cache[lang_out]
                aligned = whisperx.align(res["segments"], am, meta, audio, device, return_char_alignments=False)
                for sg in aligned["segments"]:
                    for w in sg.get("words", []):
                        if "start" in w and "end" in w:
                            words.append(Word(b0 + w["start"], b0 + w["end"], w["word"], float(w.get("score", 0.8))))
                if progress:
                    progress(100 * (bi + 1) / len(blocks), "Transcribiendo (WhisperX)")
        return RawTranscript(words, lang_out, lang_conf, self.name, f"whisperx-{tcfg['model_size']}")


def get_transcriber(cfg: dict) -> Transcriber:
    if cfg["transcription"]["engine"] == "whisperx":
        try:
            import whisperx  # noqa: F401
            return WhisperXTranscriber()
        except Exception:
            log.warning("WhisperX no está instalado; se usa faster-whisper")
    return FasterWhisperTranscriber()
