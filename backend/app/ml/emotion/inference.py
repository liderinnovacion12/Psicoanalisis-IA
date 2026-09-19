"""Análisis emocional de una llamada completa por ventanas, con lotes y reanudación.

Nunca carga el audio completo: lee cada ventana con seek sobre el WAV del hablante.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from app.ml.audio.io import WavReader
from app.ml.emotion.base import EmotionModel, EmotionOutput
from app.ml.emotion.preprocessing import Window, rms_dbfs
from app.ml.prosody.features import window_prosody

SILENCE_DBFS = -60.0


@dataclass
class WindowResult:
    window: Window
    output: EmotionOutput | None          # None = ventana silenciosa/omitida
    prosody: dict


def iter_window_results(model: EmotionModel, windows: list[Window], work_dir, files: dict[str, str],
                        speaker_channel: dict[str, str], batch_size: int = 16, start_index: int = 0,
                        with_prosody: bool = True, prosody_cfg: dict | None = None
                        ) -> Iterator[list[WindowResult]]:
    """Genera listas de resultados por lote. `start_index` permite reanudar (ventanas ya guardadas).

    speaker_channel: {"SPEAKER_00": "ch0"|"mono", ...} -> clave de `files`.
    """
    readers: dict[str, WavReader] = {}
    pcfg = prosody_cfg or {}
    try:
        pending = windows[start_index:]
        for b in range(0, len(pending), batch_size):
            batch = pending[b:b + batch_size]
            waves, keep = [], []
            for w in batch:
                key = speaker_channel[w.speaker]
                if key not in readers:
                    readers[key] = WavReader(work_dir / files[key])
                x = readers[key].read(w.start, w.end, channel=0 if readers[key].channels > 1 else None)
                waves.append(x)
                keep.append(len(x) >= 4000 and rms_dbfs(x) > SILENCE_DBFS)
            outs = model.predict([x for x, k in zip(waves, keep) if k]) if any(keep) else []
            it = iter(outs)
            res = []
            for w, x, k in zip(batch, waves, keep):
                pros = window_prosody(x, model.sample_rate, pcfg.get("f0_min", 65), pcfg.get("f0_max", 500)) \
                    if (with_prosody and k) else {}
                res.append(WindowResult(w, next(it) if k else None, pros))
            yield res
    finally:
        for r in readers.values():
            r.close()
