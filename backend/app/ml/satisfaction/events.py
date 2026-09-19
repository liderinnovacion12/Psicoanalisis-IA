"""Detección de momentos críticos (sistema de eventos). Umbrales en satisfaction_config.yaml > events.

Los eventos son *señales compatibles con* fenómenos emocionales; no afirman causas. Las frases del
texto asociadas a un evento se marcan como "posible factor asociado" (coincidencia temporal).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from app.ml.satisfaction.engine import SpeakerAnalysis
from app.ml.satisfaction.features import (Series, emotion_share, frustration_index, smooth, tension_index,
                                          weights_vector)

EVENT_LABELS = {
    "frustration_peak": "Frustración elevada",
    "anger_peak": "Pico de enojo",
    "positive_peak": "Pico de emoción positiva",
    "emotional_shift": "Cambio emocional importante",
    "recovery": "Recuperación emocional",
    "deterioration": "Deterioro emocional",
    "satisfaction_jump": "Cambio brusco de satisfacción",
    "possible_conflict": "Posible conflicto",
    "positive_ending": "Final positivo",
    "negative_ending": "Final negativo",
}


def _sat_at(a: SpeakerAnalysis | None, t: float) -> float | None:
    if not a or not a.timeline:
        return None
    ts = np.array([p["t"] for p in a.timeline]); vs = np.array([p["score"] for p in a.timeline])
    return round(float(np.interp(t, ts, vs)), 1)


def _emotion_at(s: Series, t: float) -> str:
    i = int(np.argmin(np.abs(s.t - t)))
    return s.labels[int(np.argmax(s.P[i]))]


def _evidence(utts: list[dict], t: float, window: float, kind: str) -> dict:
    """Frases próximas en el tiempo (posible factor asociado). Prioriza las que traen cues de frustración/queja."""
    cand = []
    for u in utts:
        c = (u["start"] + u["end"]) / 2
        if abs(c - t) <= window or (u["start"] <= t <= u["end"]):
            sent = u.get("sentiment") or {}
            cue_types = sorted({c_["type"] for c_ in sent.get("cues", [])})
            rel = sent.get("frustration", 0) if kind in ("frustration", "anger", "conflict") else abs(sent.get("score", 0))
            cand.append((rel, u, cue_types))
    cand.sort(key=lambda x: -x[0])
    out = [{"text": u["text"][:180], "start": round(u["start"], 2), "end": round(u["end"], 2), "cues": ct}
           for rel, u, ct in cand[:2] if rel > 0.05]
    return {"possible_factors": out,
            "note": "Coincidencia temporal; no implica causalidad."} if out else {}


def _debounce(idx_times: list[tuple[float, float]], sep: float) -> list[tuple[float, float]]:
    """Mantiene el de mayor valor dentro de `sep` segundos."""
    keep: list[tuple[float, float]] = []
    for t, v in sorted(idx_times, key=lambda x: -x[1]):
        if all(abs(t - k) >= sep for k, _ in keep):
            keep.append((t, v))
    return sorted(keep)


def detect_events(series: dict[str, Series], analyses: dict[str, SpeakerAnalysis | None],
                  utterances: dict[str, list[dict]], sat_cfg: dict, emo_cfg: dict,
                  overlaps: list[tuple[float, float, str]] | None = None) -> list[dict]:
    ev = sat_cfg["events"]
    sep = ev["min_separation_seconds"]
    w = sat_cfg["weights"]
    cues_win = emo_cfg.get("text_sentiment", {}).get("cues_window_seconds", 20)
    events: list[dict] = []

    def add(spk, t, typ, sev, conf, s: Series | None, a, end=None, desc="", kind="frustration", utt=None):
        conf = float(np.clip(conf, 0, 1))
        if conf < ev["min_event_confidence"]:
            return
        events.append({
            "speaker": spk, "timestamp": round(float(t), 2), "end_timestamp": None if end is None else round(float(end), 2),
            "event_type": typ, "label": EVENT_LABELS[typ], "severity": round(float(np.clip(sev, 0, 1)), 3),
            "confidence": round(conf, 3), "emotion": _emotion_at(s, t) if s is not None and s.n else None,
            "satisfaction": _sat_at(a, t) if a else None, "description": desc,
            "evidence": _evidence(utt or [], t, cues_win, kind)})

    tension_by = {}
    for spk, s in series.items():
        a = analyses.get(spk)
        if s is None or s.n < 5 or a is None:
            continue
        utt = utterances.get(spk, [])
        k = max(1, int(10 / s.dt))
        fr = smooth(frustration_index(s, emo_cfg, emo_cfg.get("fusion")), k)
        tens = smooth(tension_index(s, emo_cfg, emo_cfg.get("fusion")), k)
        tension_by[spk] = (s.t, tens)
        ang = smooth(s.P[:, s.labels.index("angry")], k) if "angry" in s.labels else np.zeros(s.n)
        pos = smooth(emotion_share(s, emo_cfg["signals"]["positive_emotions"]), k)
        dist = max(1, int(sep / s.dt))
        conf_local = smooth(s.conf, k)

        def peaks(x, thr):
            pk, _ = find_peaks(x, height=thr, distance=dist, prominence=0.08)
            return _debounce([(float(s.t[i]), float(x[i])) for i in pk], sep)

        for t, v in peaks(fr, ev["frustration_threshold"]):
            i = int(np.argmin(np.abs(s.t - t)))
            add(spk, t, "frustration_peak", v, 0.5 * v + 0.5 * conf_local[i], s, a, desc=(
                f"Se detectan señales compatibles con frustración elevada (índice {v:.2f})."), kind="frustration", utt=utt)
        for t, v in peaks(ang, ev["anger_threshold"]):
            i = int(np.argmin(np.abs(s.t - t)))
            add(spk, t, "anger_peak", v, 0.5 * v + 0.5 * conf_local[i], s, a, desc=(
                f"El modelo estima una probabilidad elevada de 'angry' (p={v:.2f})."), kind="anger", utt=utt)
        for t, v in peaks(pos, ev["positive_threshold"]):
            i = int(np.argmin(np.abs(s.t - t)))
            add(spk, t, "positive_peak", v, 0.5 * v + 0.5 * conf_local[i], s, a, desc=(
                f"Se detectan señales compatibles con emoción positiva (p={v:.2f})."), kind="positive", utt=utt)

        # cambio emocional importante: |media(valencia siguiente) - media(previa)|
        val = s.P @ weights_vector(s.labels, w)
        sw = max(3, int(ev["shift_window_seconds"] / s.dt))
        if s.n >= 2 * sw + 1:
            d = np.array([val[i:i + sw].mean() - val[i - sw:i].mean() if sw <= i <= s.n - sw else 0.0 for i in range(s.n)])
            pk, _ = find_peaks(np.abs(d), height=ev["shift_valence_delta"], distance=dist)
            for t, v in _debounce([(float(s.t[i]), float(abs(d[i]))) for i in pk], sep):
                i = int(np.argmin(np.abs(s.t - t)))
                direction = "hacia estados más positivos" if d[i] > 0 else "hacia estados más negativos"
                add(spk, t, "emotional_shift", min(1.0, v), 0.5 * min(1.0, v) + 0.5 * conf_local[i], s, a,
                    desc=f"Cambio sostenido de la señal emocional {direction} (Δ={d[i]:+.2f}).", kind="shift", utt=utt)

        # recuperación / deterioro / cambio brusco sobre la línea de tiempo de satisfacción
        tl_t = np.array([p["t"] for p in a.timeline]); tl_v = np.array([p["score"] for p in a.timeline])
        if len(tl_t) >= 4:
            step = float(np.median(np.diff(tl_t))) or 1.0
            back = max(1, int(120 / step)); jw = max(1, int(ev["jump_window_seconds"] / step))
            rec, det, jmp = [], [], []
            for i in range(1, len(tl_t)):
                lo = tl_v[max(0, i - back):i + 1]
                if tl_v[i] - lo.min() >= ev["recovery_delta"]:
                    rec.append((float(tl_t[i]), float(tl_v[i] - lo.min())))
                if lo.max() - tl_v[i] >= ev["deterioration_delta"]:
                    det.append((float(tl_t[i]), float(lo.max() - tl_v[i])))
                if i >= jw and abs(tl_v[i] - tl_v[i - jw]) >= ev["satisfaction_jump"]:
                    jmp.append((float(tl_t[i]), float(abs(tl_v[i] - tl_v[i - jw]))))
            used = []
            for lst, typ, txt in ((rec, "recovery", "aumento"), (det, "deterioration", "descenso")):
                for t, dv in _debounce(lst, 120):
                    used.append(t)
                    add(spk, t, typ, min(1.0, dv / 50), 0.4 + 0.5 * float(conf_local[int(np.argmin(np.abs(s.t - t)))]), s, a,
                        desc=f"La satisfacción estimada muestra un {txt} de {dv:.0f} puntos respecto al mínimo/máximo reciente.",
                        kind="shift", utt=utt)
            for t, dv in _debounce(jmp, 120):
                if any(abs(t - u) < 60 for u in used):
                    continue
                add(spk, t, "satisfaction_jump", min(1.0, dv / 50), 0.4 + 0.5 * float(conf_local[int(np.argmin(np.abs(s.t - t)))]),
                    s, a, desc=f"Variación brusca de la satisfacción estimada ({dv:.0f} puntos en {ev['jump_window_seconds']} s).",
                    kind="shift", utt=utt)

        # final positivo / negativo
        fs = a.components.get("final_state")
        if fs is not None and abs(fs) >= ev["ending_threshold"]:
            L = float(np.clip(sat_cfg["final_segment"]["fraction"] * (s.t[-1] - s.t[0]),
                              sat_cfg["final_segment"]["min_seconds"], sat_cfg["final_segment"]["max_seconds"]))
            t0 = max(float(s.t[0]), float(s.t[-1]) - L)
            add(spk, t0, "positive_ending" if fs > 0 else "negative_ending", min(1.0, abs(fs)),
                0.4 + 0.5 * min(1.0, abs(fs)), s, a, end=float(s.t[-1]), kind="positive" if fs > 0 else "frustration",
                desc=("Las señales del tramo final son predominantemente " + ("positivas." if fs > 0 else "negativas.")), utt=utt)

    # posible conflicto: tensión simultánea alta en ambos hablantes (o solapamiento con tensión)
    spks = [k for k in tension_by]
    if len(spks) == 2:
        (t0, x0), (t1, x1) = tension_by[spks[0]], tension_by[spks[1]]
        grid = np.arange(0, max(t0[-1], t1[-1]), 2.0)
        a0 = np.interp(grid, t0, x0, left=0, right=0); a1 = np.interp(grid, t1, x1, left=0, right=0)
        both = np.minimum(a0, a1)
        near0 = np.array([np.min(np.abs(t0 - g)) <= 6 for g in grid]); near1 = np.array([np.min(np.abs(t1 - g)) <= 6 for g in grid])
        hot = (both >= ev["conflict_threshold"]) & near0 & near1
        i = 0
        cands = []
        while i < len(grid):
            if hot[i]:
                j = i
                while j < len(grid) and hot[j]:
                    j += 1
                if grid[j - 1] - grid[i] >= 6:
                    cands.append((float(grid[i]), float(grid[j - 1]), float(both[i:j].max())))
                i = j
            else:
                i += 1
        for b, e, v in cands:
            olap = sum(min(e, oe) - max(b, os_) for os_, oe, _ in (overlaps or []) if min(e, oe) > max(b, os_))
            sev = min(1.0, v + (0.1 if olap > 1 else 0))
            add(None, b, "possible_conflict", sev, 0.4 + 0.4 * v, None, None, end=e,
                desc=("Señales de tensión simultáneas en ambos participantes" + (" con solapamiento de voces" if olap > 1 else "") +
                      f" (índice {v:.2f}). Puede indicar un intercambio conflictivo; requiere revisión humana."),
                kind="conflict", utt=sum(utterances.values(), []))
    events.sort(key=lambda e: e["timestamp"])
    return events
