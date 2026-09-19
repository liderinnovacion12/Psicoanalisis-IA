"""Interfaz de diarización + post-proceso común. Cambiar de motor = implementar `Diarizer`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SPEAKERS = ("SPEAKER_00", "SPEAKER_01")


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    turns: list[Turn]
    engine: str
    n_speakers_detected: int
    quality: float                      # 0-100 (estimación heurística, ver README)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"turns": [asdict(t) for t in self.turns], "engine": self.engine,
                "n_speakers_detected": self.n_speakers_detected, "quality": self.quality,
                "warnings": self.warnings, "details": self.details}

    @staticmethod
    def from_dict(d: dict) -> "DiarizationResult":
        return DiarizationResult([Turn(**t) for t in d["turns"]], d["engine"], d["n_speakers_detected"],
                                 d["quality"], d.get("warnings", []), d.get("details", {}))


class Diarizer(ABC):
    name = "diarizer"

    @abstractmethod
    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict,
                progress=None) -> DiarizationResult: ...


def merge_turns(turns: list[Turn], gap: float, min_len: float = 0.0) -> list[Turn]:
    """Une turnos consecutivos del mismo hablante separados por < gap; descarta turnos muy cortos."""
    out: list[Turn] = []
    last: dict[str, Turn] = {}
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        prev = last.get(t.speaker)
        if prev is not None and t.start - prev.end <= gap:
            prev.end = max(prev.end, t.end)
            prev.overlap = prev.overlap or t.overlap
        else:
            nt = Turn(t.speaker, t.start, t.end, t.overlap)
            out.append(nt)
            last[t.speaker] = nt
    return [t for t in out if t.duration >= min_len]


def mark_overlaps(turns: list[Turn]) -> None:
    """Marca turnos que se solapan con turnos de otro hablante."""
    ts = sorted(turns, key=lambda t: t.start)
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            if b.start >= a.end:
                break
            if b.speaker != a.speaker:
                a.overlap = b.overlap = True


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def reduce_to_two(raw: dict[str, list[tuple[float, float]]], embeddings: dict[str, np.ndarray] | None,
                  cfg: dict) -> tuple[list[Turn], list[str], dict]:
    """Deja exactamente dos hablantes (los de mayor tiempo de habla).

    Voces minoritarias se reasignan al hablante principal más cercano (embedding si existe; si no,
    por vecindad temporal). Devuelve (turns, warnings, info). Etiquetas por orden de aparición.
    """
    warnings: list[str] = []
    talk = {s: sum(e - b for b, e in segs) for s, segs in raw.items()}
    total = sum(talk.values()) or 1.0
    ranked = sorted(talk, key=talk.get, reverse=True)
    n_detected = len([s for s in ranked if talk[s] / total >= cfg.get("minor_speaker_ratio", 0.05)])
    if len(ranked) >= 2 and n_detected > 2:
        warnings.append("Se detectaron más de dos posibles hablantes. Revise la diarización.")
    majors = ranked[:2]
    if not majors:
        return [], ["No se detectó voz para diarizar."], {"n_detected": 0}
    first_seen = {s: min(b for b, _ in raw[s]) for s in majors}
    majors.sort(key=lambda s: first_seen[s])
    label_of = {s: SPEAKERS[i] for i, s in enumerate(majors)}
    turns: list[Turn] = []
    for s in majors:
        turns += [Turn(label_of[s], b, e) for b, e in raw[s]]
    minors = [s for s in ranked[2:]]
    if minors:
        major_ts = sorted(turns, key=lambda t: t.start)
        starts = np.array([t.start for t in major_ts])
        for s in minors:
            tgt = None
            if embeddings and s in embeddings and all(m in embeddings for m in majors):
                tgt = min(majors, key=lambda m: cosine_distance(embeddings[s], embeddings[m]))
                tgt = label_of[tgt]
            for b, e in raw[s]:
                if tgt is None:
                    i = int(np.searchsorted(starts, b)) - 1
                    tgt_i = major_ts[max(i, 0)].speaker if major_ts else SPEAKERS[0]
                else:
                    tgt_i = tgt
                turns.append(Turn(tgt_i, b, e))
    info = {"n_detected": len(ranked), "talk_time": {label_of.get(s, s): round(talk[s], 2) for s in ranked},
            "minor_ratio": round(sum(talk[s] for s in minors) / total, 4)}
    return turns, warnings, info


def turn_quality_proxy(turns: list[Turn], duration: float, sep: float | None, minor_ratio: float) -> float:
    """Proxy 0-100 de calidad de diarización sin ground truth (heurístico, NO es DER)."""
    if not turns or duration <= 0:
        return 0.0
    switches = sum(1 for a, b in zip(turns, turns[1:]) if a.speaker != b.speaker)
    per_min = switches / (duration / 60)
    plaus = 1.0 if 0.5 <= per_min <= 25 else 0.6
    both = len({t.speaker for t in turns}) == 2
    q_sep = 0.7 if sep is None else float(np.clip(sep / 0.7, 0, 1))
    q_assigned = 1 - min(minor_ratio * 2, 1)
    q = 0.5 * q_sep + 0.3 * q_assigned + 0.2 * plaus
    return round(100 * q * (1.0 if both else 0.5), 1)
