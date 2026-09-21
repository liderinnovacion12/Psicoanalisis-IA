"""Interfaz de transcripción + utilidades: bloques cortados en silencios y asignación palabra→hablante."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.ml.diarization.base import Turn


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float = 1.0


@dataclass
class RawTranscript:
    words: list[Word]
    language: str | None
    language_confidence: float | None
    engine: str
    model: str
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"words": [asdict(w) for w in self.words], "language": self.language,
                "language_confidence": self.language_confidence, "engine": self.engine,
                "model": self.model, "info": self.info}

    @staticmethod
    def from_dict(d: dict) -> "RawTranscript":
        return RawTranscript([Word(**w) for w in d["words"]], d["language"], d["language_confidence"],
                             d["engine"], d["model"], d.get("info", {}))


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str
    confidence: float
    language: str | None = None
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "confidence": round(self.confidence, 4), "language": self.language}


class Transcriber(ABC):
    name = "transcriber"

    @abstractmethod
    def transcribe_file(self, wav_path: Path, vad: list[tuple[float, float]], cfg: dict,
                        language: str | None = None,
                        progress: Callable[[float, str], None] | None = None) -> RawTranscript: ...


def plan_blocks(duration: float, vad: list[tuple[float, float]], block_s: float, search_s: float = 30.0
                ) -> list[tuple[float, float]]:
    """Divide [0,duration] en bloques de ~block_s, cortando en el silencio más largo cercano al objetivo."""
    if duration <= block_s * 1.15:
        return [(0.0, duration)]
    gaps = [(vad[i][1], vad[i + 1][0]) for i in range(len(vad) - 1) if vad[i + 1][0] - vad[i][1] > 0.3]
    blocks, cur = [], 0.0
    while duration - cur > block_s * 1.15:
        target = cur + block_s
        cands = [(g1 - g0, (g0 + g1) / 2) for g0, g1 in gaps if abs((g0 + g1) / 2 - target) <= search_s]
        cut = max(cands)[1] if cands else target
        blocks.append((cur, cut))
        cur = cut
    blocks.append((cur, duration))
    return blocks


def assign_words_to_speakers(words: list[Word], turns: list[Turn], max_gap: float = 1.0) -> list[tuple[str, Word]]:
    """Asigna cada palabra al hablante cuyo turno contiene su punto medio (o el más cercano)."""
    ts = sorted(turns, key=lambda t: t.start)
    starts = np.array([t.start for t in ts])
    out: list[tuple[str, Word]] = []
    last_spk = ts[0].speaker if ts else "SPEAKER_00"
    for w in words:
        mid = (w.start + w.end) / 2
        i = int(np.searchsorted(starts, mid, side="right"))
        # candidatos: turnos que empiezan antes de `mid` (revisar unos cuantos hacia atrás)
        cands = [t for t in ts[max(0, i - 6):i] if t.start <= mid <= t.end]
        if cands:
            spk = min(cands, key=lambda t: t.duration).speaker    # el turno más corto (más específico)
        else:
            near = None
            best = max_gap
            for t in ts[max(0, i - 3):i + 3]:
                d = 0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end))
                if d < best:
                    best, near = d, t
            spk = near.speaker if near else last_spk
        last_spk = spk
        out.append((spk, w))
    return out


_REPEAT = re.compile(r"(\b[\wáéíóúñü¿?¡!,.\-]+(?:\s+[\wáéíóúñü¿?¡!,.\-]+){0,5}?)(?:\s+\1){3,}", re.IGNORECASE)


def collapse_repeats(text: str) -> str:
    """Whisper a veces entra en bucle repitiendo una frase; se deja una sola aparición (>=4 repeticiones seguidas)."""
    prev = None
    while prev != text:
        prev, text = text, _REPEAT.sub(r"\1", text)
    return text


def group_utterances(assigned: list[tuple[str, Word]], language: str | None, gap: float = 1.2,
                     max_len: float = 25.0) -> list[Utterance]:
    """Agrupa palabras en frases por hablante. Los tokens de Whisper traen un espacio inicial cuando empiezan una
    palabra nueva; los que no (puntuación, sufijos) se pegan a la anterior."""
    utts: list[Utterance] = []
    cur: Utterance | None = None
    for spk, w in sorted(assigned, key=lambda x: x[1].start):
        new = (cur is None or cur.speaker != spk or w.start - cur.end > gap
               or (w.end - cur.start > max_len and cur.text.rstrip().endswith((".", "?", "!"))))
        if new:
            if cur:
                utts.append(cur)
            cur = Utterance(spk, w.start, w.end, w.word.strip(), 0.0, language, [w])
        else:
            starts_word = w.word.startswith(" ") or not cur.text
            cur.text += (" " if starts_word else "") + w.word.strip()
            cur.end = max(cur.end, w.end)
            cur.words.append(w)
    if cur:
        utts.append(cur)
    for u in utts:
        u.text = collapse_repeats(" ".join(u.text.split()))
        u.confidence = float(np.mean([w.prob for w in u.words])) if u.words else 0.0
    return [u for u in utts if u.text]
