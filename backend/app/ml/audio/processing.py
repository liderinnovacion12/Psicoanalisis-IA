"""Etapa 1: validación + preprocesamiento de audio (FFmpeg) + calidad + VAD.

Salida en `work_dir`:
  * mode == "stereo_split": ch0.wav, ch1.wav (16 kHz mono, normalizados)   -> un hablante por canal
  * mode == "mono":         mono.wav (16 kHz mono, normalizado)             -> requiere diarización
  * vad.json: segmentos de voz por archivo.
Ningún paso carga el audio completo en memoria.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from app.core.errors import InvalidAudio
from app.core.logging import get_logger
from app.ml.audio import ffmpeg
from app.ml.audio.io import wav_info
from app.ml.audio.quality import analyze_levels, detect_channel_layout, quality_score
from app.ml.audio.vad import Segments, get_vad, merge_segments

log = get_logger(__name__)


def validate_source(path: Path, cfg: dict) -> dict:
    ext = path.suffix.lower().lstrip(".")
    if ext not in cfg["accepted_extensions"]:
        raise InvalidAudio(detail=f"extensión no soportada: {ext}")
    meta = ffmpeg.probe(path)
    if meta["duration"] > cfg["max_duration_hours"] * 3600:
        raise InvalidAudio("La grabación supera la duración máxima permitida.",
                           detail=f"duración {meta['duration']}s")
    return meta


def build_filter(norm: dict) -> str | None:
    parts = []
    if norm.get("highpass_hz"):
        parts.append(f"highpass=f={int(norm['highpass_hz'])}")
    if norm.get("loudnorm"):
        parts.append(f"loudnorm=I={norm.get('target_lufs', -23)}:LRA=11:TP=-2")
    return ",".join(parts) or None


def process_audio(src: Path, work_dir: Path, cfg: dict,
                  progress: Callable[[float, str], None] | None = None) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    p = progress or (lambda pct, msg: None)
    norm = cfg["normalization"]
    sr = norm["sample_rate"]

    p(2, "Validando archivo")
    meta = validate_source(src, cfg)

    p(8, "Decodificando a PCM 16 kHz")
    raw = work_dir / "raw.wav"
    n_ch = 2 if meta["channels"] >= 2 else 1
    ffmpeg.to_wav(src, raw, sample_rate=sr, channels=n_ch)

    p(30, "Analizando calidad y canales")
    levels = analyze_levels(raw, cfg["vad"]["block_seconds"], sr)
    layout = {"is_stereo_split": False, "reason": "mono", "correlation": None}
    if n_ch == 2 and cfg["channels"]["detect_separation"]:
        layout = detect_channel_layout(raw, cfg["channels"], cfg["vad"]["block_seconds"], sr)
    mode = "stereo_split" if layout["is_stereo_split"] else "mono"

    p(45, "Normalizando audio")
    af = build_filter(norm)
    files: dict[str, str] = {}
    if mode == "stereo_split":
        for ch in (0, 1):
            ffmpeg.extract_channel(raw, work_dir / f"ch{ch}.wav", ch, sample_rate=sr, af=af)
            files[f"ch{ch}"] = f"ch{ch}.wav"
    else:
        ffmpeg.to_wav(raw, work_dir / "mono.wav", sample_rate=sr, channels=1, af=af)
        files["mono"] = "mono.wav"
    raw.unlink(missing_ok=True)

    p(65, "Detectando voz (VAD)")
    vad = get_vad(cfg["vad"], sr)
    vad_segments: dict[str, Segments] = {k: vad.detect(work_dir / v) for k, v in files.items()}
    (work_dir / "vad.json").write_text(json.dumps(vad_segments), encoding="utf-8")

    duration = wav_info(work_dir / next(iter(files.values())))[2]
    union = merge_segments([s for segs in vad_segments.values() for s in segs], 0.0)
    speech_ratio = sum(e - s for s, e in union) / max(duration, 1e-6)

    p(90, "Calculando calidad")
    quality = quality_score(meta, levels, speech_ratio, cfg)
    quality.update({"levels": levels, "speech_ratio": round(speech_ratio, 4), "vad_engine": vad.name,
                    "channel_layout": layout})
    result = {
        "meta": {**meta, "normalized_duration": duration},
        "mode": mode, "files": files, "quality": quality, "speech_ratio": speech_ratio,
        "duration": duration,
    }
    p(100, "Audio listo")
    return result


def load_vad(work_dir: Path) -> dict[str, Segments]:
    data = json.loads((work_dir / "vad.json").read_text(encoding="utf-8"))
    return {k: [(float(s), float(e)) for s, e in v] for k, v in data.items()}
