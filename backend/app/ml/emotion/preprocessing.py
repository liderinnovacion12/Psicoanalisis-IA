"""Segmentación en ventanas deslizantes (window/hop) DENTRO de cada turno de hablante."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    index: int
    turn_index: int
    speaker: str
    start: float
    end: float
    overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_windows(turns: list, window: float = 5.0, hop: float = 2.5, min_window: float = 1.0) -> list[Window]:
    """`turns`: objetos con speaker/start/end/(overlap). Determinista para poder reanudar por índice.

    * turno más corto que `window` -> una ventana con el turno completo (si dura >= min_window);
    * turno largo -> ventanas cada `hop`; el último tramo se ancla al final para no perder emociones breves.
    """
    out: list[Window] = []
    for ti, t in enumerate(turns):
        dur = t.end - t.start
        ov = bool(getattr(t, "overlap", False))
        if dur < min_window:
            continue
        if dur <= window:
            out.append(Window(len(out), ti, t.speaker, t.start, t.end, ov))
            continue
        starts = list(np.arange(t.start, t.end - window + 1e-6, hop))
        for s in starts:
            out.append(Window(len(out), ti, t.speaker, float(s), float(s + window), ov))
        last_end = starts[-1] + window
        if t.end - last_end >= min_window:                  # cola sin cubrir
            out.append(Window(len(out), ti, t.speaker, t.end - window, t.end, ov))
    return out


def rms_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))
