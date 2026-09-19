"""Lectura de audio por bloques / ventanas sin cargar el archivo completo en memoria."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


def iter_blocks(path: str | Path, block_seconds: float = 300.0) -> Iterator[tuple[float, np.ndarray]]:
    """Genera (offset_s, ndarray[frames, canales] float32) por bloques."""
    with sf.SoundFile(str(path)) as f:
        frames = int(block_seconds * f.samplerate)
        pos = 0
        while True:
            data = f.read(frames, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            yield pos / f.samplerate, data
            pos += len(data)


def wav_info(path: str | Path) -> tuple[int, int, float]:
    """(sample_rate, channels, duration_s)"""
    i = sf.info(str(path))
    return i.samplerate, i.channels, i.frames / i.samplerate


class WavReader:
    """Lector de ventanas con manejador persistente (seek + read)."""

    def __init__(self, path: str | Path):
        self._f = sf.SoundFile(str(path))
        self.sample_rate = self._f.samplerate
        self.channels = self._f.channels
        self.duration = self._f.frames / self._f.samplerate

    def read(self, start: float, end: float, channel: int | None = None) -> np.ndarray:
        start = max(0.0, start)
        n = max(0, int((end - start) * self.sample_rate))
        self._f.seek(min(int(start * self.sample_rate), self._f.frames))
        data = self._f.read(n, dtype="float32", always_2d=True)
        if channel is not None:
            return np.ascontiguousarray(data[:, channel])
        return np.ascontiguousarray(data.mean(axis=1)) if data.shape[1] > 1 else np.ascontiguousarray(data[:, 0])

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
