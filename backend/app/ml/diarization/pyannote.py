"""Diarización con pyannote.audio (motor principal para audio mono).

* Procesa el audio en chunks (por defecto 30 min, con solape) para no cargar horas de audio en RAM.
* Reconcilia hablantes entre chunks comparando los centroides de embedding (distancia coseno).
* Fuerza un mínimo de 2 hablantes y sondea hasta `max_speakers_probe` para poder avisar de >2 voces.

Requiere: `pip install "pyannote.audio>=3.3,<4"`, aceptar las condiciones del modelo en Hugging Face y
definir HF_TOKEN. NOTA: este módulo fue escrito contra la API de pyannote 3.x/4.x pero NO pudo ejecutarse
en el entorno de desarrollo de esta entrega (sin token / sin acceso al modelo restringido).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.device import torch_device
from app.core.logging import get_logger
from app.ml.audio.io import WavReader
from app.ml.diarization.base import (DiarizationResult, Diarizer, cosine_distance, mark_overlaps,
                                     merge_turns, reduce_to_two, turn_quality_proxy)

log = get_logger(__name__)
_PIPE = None


def pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    return bool(get_settings().hf_token)


def _load_pipeline(model: str):
    global _PIPE
    if _PIPE is None:
        import torch
        from pyannote.audio import Pipeline
        s = get_settings()
        try:
            pipe = Pipeline.from_pretrained(model, token=s.hf_token, cache_dir=str(s.model_cache_dir))
        except TypeError:  # pyannote 3.x usa `use_auth_token`
            pipe = Pipeline.from_pretrained(model, use_auth_token=s.hf_token, cache_dir=str(s.model_cache_dir))
        if pipe is None:
            raise RuntimeError("No se pudo cargar pyannote (¿token/condiciones del modelo?).")
        pipe.to(torch.device(torch_device(s.device)))
        _PIPE = pipe
    return _PIPE


class PyannoteDiarizer(Diarizer):
    name = "pyannote"

    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict, progress=None, hint: int | None = None):
        import torch
        d = cfg["diarization"]
        pipe = _load_pipeline(d["pyannote_model"])
        path = work_dir / files["mono"]
        chunk_s, ovl = d["chunk_minutes"] * 60, d["chunk_overlap_seconds"]
        raw: dict[str, list[tuple[float, float]]] = {}
        centroids: list[np.ndarray] = []            # embeddings globales
        counts: list[int] = []
        with WavReader(path) as rd:
            dur, sr = rd.duration, rd.sample_rate
            starts = list(np.arange(0, dur, chunk_s - ovl)) or [0.0]
            for ci, t0 in enumerate(starts):
                t1 = min(dur, t0 + chunk_s)
                wav = rd.read(t0, t1)
                inp = {"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": sr}
                out = pipe(inp, min_speakers=2 if hint == 2 else 1, max_speakers=1 if hint == 1 else d["max_speakers_probe"],
                           return_embeddings=True)
                if hasattr(out, "speaker_diarization"):          # pyannote 4.x
                    diar, emb = out.speaker_diarization, getattr(out, "speaker_embeddings", None)
                else:                                              # pyannote 3.x
                    diar, emb = out if isinstance(out, tuple) else (out, None)
                labels = list(diar.labels())
                # reconciliar hablantes locales con los globales
                mapping: dict[str, str] = {}
                for li, lab in enumerate(labels):
                    e = None if emb is None else np.asarray(emb[li], dtype=np.float32)
                    if e is None or not np.isfinite(e).all():
                        gi = li if li < len(centroids) else len(centroids)
                        if gi == len(centroids):
                            centroids.append(np.zeros(1, dtype=np.float32)); counts.append(0)
                    else:
                        dists = [cosine_distance(e, c) if c.shape == e.shape else 9 for c in centroids]
                        j = int(np.argmin(dists)) if dists else -1
                        if j >= 0 and dists[j] <= d["match_threshold"] and j not in mapping.values():
                            gi = j
                            centroids[j] = (centroids[j] * counts[j] + e) / (counts[j] + 1)
                            counts[j] += 1
                        else:
                            centroids.append(e); counts.append(1); gi = len(centroids) - 1
                    mapping[lab] = f"G{gi}"
                lo = t0 + (ovl / 2 if ci > 0 else 0)
                hi = t1 - (ovl / 2 if ci < len(starts) - 1 else 0)
                for turn, _, lab in diar.itertracks(yield_label=True):
                    b, e_ = t0 + turn.start, t0 + turn.end
                    b, e_ = max(b, lo), min(e_, hi)
                    if e_ - b > 0.05:
                        raw.setdefault(mapping[lab], []).append((b, e_))
                if progress:
                    progress(100 * (ci + 1) / len(starts), f"Diarización chunk {ci + 1}/{len(starts)}")
        embeds = {f"G{i}": c for i, c in enumerate(centroids) if c.size > 1}
        turns, warnings, info = reduce_to_two(raw, embeds or None, d)
        seg = cfg["segmentation"]
        turns = merge_turns(turns, seg["merge_gap"], seg["min_turn"])
        mark_overlaps(turns)
        sep = None
        if len(embeds) >= 2:
            top = sorted(raw, key=lambda s: -sum(e - b for b, e in raw[s]))[:2]
            if all(t in embeds for t in top):
                sep = cosine_distance(embeds[top[0]], embeds[top[1]])
        q = turn_quality_proxy(turns, dur, sep, info.get("minor_ratio", 0.0))
        return DiarizationResult(turns, self.name, info.get("n_detected", 2), q, warnings,
                                 {**info, "centroid_distance": None if sep is None else round(sep, 4)})
