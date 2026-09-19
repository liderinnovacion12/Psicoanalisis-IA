"""Detección de actividad de voz (VAD) por bloques. Silero (neuronal) con respaldo por energía."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.ml.audio.io import iter_blocks

log = get_logger(__name__)
Segments = list[tuple[float, float]]


def merge_segments(segs: Segments, gap: float = 0.3, min_len: float = 0.0) -> Segments:
    out: Segments = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(s, e) for s, e in out if e - s >= min_len]


class VAD(ABC):
    name = "vad"

    @abstractmethod
    def detect(self, wav_path: str | Path, channel: int | None = None) -> Segments: ...


class EnergyVAD(VAD):
    name = "energy"

    def __init__(self, cfg: dict, sample_rate: int = 16000):
        self.thr_db = cfg.get("energy_threshold_db", -42)
        self.min_speech = cfg.get("min_speech_ms", 250) / 1000
        self.min_silence = cfg.get("min_silence_ms", 300) / 1000
        self.block = cfg.get("block_seconds", 300)
        self.sr = sample_rate

    def detect(self, wav_path, channel=None) -> Segments:
        frame = int(0.03 * self.sr)
        segs: Segments = []
        for off, blk in iter_blocks(wav_path, self.block):
            x = blk[:, channel] if channel is not None else blk.mean(axis=1)
            n = len(x) // frame
            if n == 0:
                continue
            f = x[: n * frame].reshape(n, frame)
            db = 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-10)
            # umbral adaptativo: no menor que el suelo de ruido + 10 dB
            thr = max(self.thr_db, float(np.percentile(db, 10)) + 10)
            act = db > thr
            i = 0
            while i < n:
                if act[i]:
                    j = i
                    while j < n and act[j]:
                        j += 1
                    segs.append((off + i * 0.03, off + j * 0.03))
                    i = j
                else:
                    i += 1
        return merge_segments(segs, self.min_silence, self.min_speech)


class SileroVAD(VAD):
    name = "silero"

    def __init__(self, cfg: dict, sample_rate: int = 16000):
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        self._torch = torch
        self._gst = get_speech_timestamps
        self._model = load_silero_vad()
        self.cfg = cfg
        self.sr = sample_rate

    def detect(self, wav_path, channel=None) -> Segments:
        segs: Segments = []
        for off, blk in iter_blocks(wav_path, self.cfg.get("block_seconds", 300)):
            x = blk[:, channel] if channel is not None else blk.mean(axis=1)
            t = self._torch.from_numpy(np.ascontiguousarray(x))
            ts = self._gst(
                t, self._model, sampling_rate=self.sr, threshold=self.cfg.get("threshold", 0.5),
                min_speech_duration_ms=self.cfg.get("min_speech_ms", 250),
                min_silence_duration_ms=self.cfg.get("min_silence_ms", 300), return_seconds=True)
            segs.extend((off + d["start"], off + d["end"]) for d in ts)
        return merge_segments(segs, self.cfg.get("min_silence_ms", 300) / 1000,
                              self.cfg.get("min_speech_ms", 250) / 1000)


def get_vad(cfg: dict, sample_rate: int = 16000) -> VAD:
    if cfg.get("engine", "silero") == "silero":
        try:
            return SileroVAD(cfg, sample_rate)
        except Exception as e:
            log.warning("Silero VAD no disponible; se usa VAD por energía", extra={"error": str(e)})
    return EnergyVAD(cfg, sample_rate)
