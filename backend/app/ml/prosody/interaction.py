"""Métricas de interacción: interrupciones, solapamientos, silencios y pausas, velocidad del habla.

Son señales adicionales. NO se interpretan automáticamente como algo negativo.
"""
from __future__ import annotations

from app.ml.diarization.base import Turn


def interaction_metrics(turns: list[Turn], duration: float, cfg: dict, words_by_speaker: dict[str, int] | None = None
                        ) -> tuple[dict, list]:
    ic = cfg["interruptions"]
    ts = sorted(turns, key=lambda t: t.start)
    speakers = sorted({t.speaker for t in ts})
    interruptions = {s: 0 for s in speakers}
    overlaps: list[tuple[float, float, str]] = []
    # interrupción: B empieza mientras A habla y A continúa después de que B empiece (solape >= mínimo)
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            if b.start >= a.end:
                break
            if b.speaker == a.speaker:
                continue
            ov = min(a.end, b.end) - b.start
            if ov >= ic["min_overlap_seconds"]:
                overlaps.append((b.start, min(a.end, b.end), f"{a.speaker}|{b.speaker}"))
                interruptions[b.speaker] += 1        # b interrumpe a (a ya estaba hablando)
    # silencios/pausas entre habla total
    busy: list[tuple[float, float]] = []
    for t in ts:
        if busy and t.start <= busy[-1][1]:
            busy[-1] = (busy[-1][0], max(busy[-1][1], t.end))
        else:
            busy.append((t.start, t.end))
    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)]
    long_sil = [g for g in gaps if g[1] - g[0] >= ic["long_silence_seconds"]]
    pauses = [g for g in gaps if ic["pause_seconds"] <= g[1] - g[0] < ic["long_silence_seconds"]]
    talk = {s: round(sum(t.duration for t in ts if t.speaker == s), 2) for s in speakers}
    total_talk = sum(talk.values()) or 1.0
    rate = {}
    for s in speakers:
        w = (words_by_speaker or {}).get(s)
        if w is not None and talk[s] > 0:
            rate[s] = round(60 * w / talk[s], 1)          # palabras por minuto de habla
    return {
        "interruptions": interruptions,
        "overlap_seconds": round(sum(e - b for b, e, _ in overlaps), 2),
        "overlap_events": len(overlaps),
        "long_silences": len(long_sil),
        "long_silence_seconds": round(sum(e - b for b, e in long_sil), 2),
        "pauses": len(pauses),
        "talk_time": talk,
        "talk_ratio": {s: round(talk[s] / total_talk, 4) for s in speakers},
        "speech_rate_wpm": rate,
        "turn_count": {s: sum(1 for t in ts if t.speaker == s) for s in speakers},
        "silence_events": [{"start": round(b, 2), "end": round(e, 2)} for b, e in long_sil][:200],
        "note": "Señales adicionales; no se interpretan automáticamente como algo negativo.",
    }, overlaps
