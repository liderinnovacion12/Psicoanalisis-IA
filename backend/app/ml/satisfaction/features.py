"""Series temporales por hablante y features para el Satisfaction Engine.

Las ventanas de emoción (solapadas) se remuestrean a una rejilla regular de `dt` segundos
(promedio de las ventanas que cubren cada celda). Solo existen celdas en los tramos en que el
hablante fue analizado ("tiempo de habla"). Nada aquí depende de un modelo concreto: recibe
probabilidades por etiqueta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


@dataclass
class Series:
    speaker: str
    labels: list[str]
    t: np.ndarray                    # centro de cada celda (s)
    P: np.ndarray                    # [n, K] probabilidades promedio
    conf: np.ndarray                 # confianza del modelo (prob. máx.) por celda
    tension: np.ndarray              # tensión prosódica 0-1 (nan si no hay)
    text_score: np.ndarray           # sentimiento textual -1..1 (nan si no hay texto)
    text_frust: np.ndarray           # cue de frustración 0..1 (nan si no hay texto)
    dt: float = 1.0
    agree: np.ndarray | None = None  # concordancia audio-texto 0..1 (nan si no hay texto)

    @property
    def n(self) -> int:
        return len(self.t)

    def slice(self, t0: float, t1: float) -> "Series":
        m = (self.t >= t0) & (self.t < t1)
        return Series(self.speaker, self.labels, self.t[m], self.P[m], self.conf[m], self.tension[m],
                      self.text_score[m], self.text_frust[m], self.dt, None if self.agree is None else self.agree[m])


def build_series(speaker: str, labels: list[str], windows: list[dict], utterances: list[dict],
                 dt: float = 1.0) -> Series | None:
    """windows: [{start,end,probabilities,confidence,tension?}] ; utterances: [{start,end,sentiment?}]"""
    if not windows:
        return None
    K = len(labels)
    t_end = max(w["end"] for w in windows)
    n = int(np.ceil(t_end / dt)) + 1
    P = np.zeros((n, K)); cnt = np.zeros(n); conf = np.zeros(n)
    ten = np.zeros(n); tcnt = np.zeros(n)
    ag = np.zeros(n); acnt = np.zeros(n)
    for w in windows:
        i0 = int(np.floor(w["start"] / dt))
        i1 = max(i0 + 1, int(np.ceil(w["end"] / dt)))
        vec = np.array([w["probabilities"].get(l, 0.0) for l in labels])
        P[i0:i1] += vec
        conf[i0:i1] += w["confidence"]
        cnt[i0:i1] += 1
        if w.get("tension") is not None:
            ten[i0:i1] += w["tension"]
            tcnt[i0:i1] += 1
        if w.get("agreement") is not None:
            ag[i0:i1] += w["agreement"]
            acnt[i0:i1] += 1
    m = cnt > 0
    P[m] /= cnt[m, None]
    conf[m] /= cnt[m]
    tension = np.full(n, np.nan)
    tm = tcnt > 0
    tension[tm] = ten[tm] / tcnt[tm]
    t = (np.arange(n) + 0.5) * dt
    ts = np.full(n, np.nan); tf = np.full(n, np.nan)
    for u in utterances:
        s = u.get("sentiment") or {}
        if not s:
            continue
        i0 = int(np.floor(u["start"] / dt)); i1 = max(i0 + 1, int(np.ceil(u["end"] / dt)))
        # peso de confianza: el texto de baja evidencia apenas mueve la señal
        ts[i0:i1] = s.get("score", 0.0) * min(1.0, 0.4 + s.get("confidence", 0.0))
        tf[i0:i1] = s.get("frustration", 0.0)
    agree = np.full(n, np.nan)
    am = acnt > 0
    agree[am] = ag[am] / acnt[am]
    return Series(speaker, labels, t[m], P[m], conf[m], tension[m], ts[m], tf[m], dt, agree[m])


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Media móvil que ignora NaN; k en celdas."""
    if len(x) == 0 or k <= 1:
        return x.copy()
    k = min(k, len(x))
    v = np.where(np.isnan(x), 0.0, x)
    w = (~np.isnan(x)).astype(float)
    ker = np.ones(k)
    num = np.convolve(v, ker, mode="same")
    den = np.convolve(w, ker, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def weights_vector(labels: list[str], w: dict) -> np.ndarray:
    return np.array([w.get(f"{l}_weight", w.get("other_weight", 0.0)) for l in labels], dtype=float)


def frustration_index(series: Series, emo_cfg: dict, weights_fusion: dict | None = None) -> np.ndarray:
    """Índice de frustración 0-1 por celda: audio (+ prosodia y texto si existen), fusionados con pesos."""
    co = emo_cfg["signals"]["frustration_emotions"]
    a = np.zeros(series.n)
    for i, l in enumerate(series.labels):
        a += co.get(l, 0.0) * series.P[:, i]
    a = np.clip(a, 0, 1)
    f = weights_fusion or {"audio_weight": 1.0, "prosody_weight": 0.35, "text_weight": 0.5}
    num = f["audio_weight"] * a
    den = np.full(series.n, f["audio_weight"])
    pros = np.clip((series.tension - 0.5) * 2, 0, 1)
    has_p = ~np.isnan(pros)
    num = num + np.where(has_p, f["prosody_weight"] * np.nan_to_num(pros), 0.0)
    den = den + np.where(has_p, f["prosody_weight"], 0.0)
    has_t = ~np.isnan(series.text_frust)
    num = num + np.where(has_t, f["text_weight"] * np.nan_to_num(series.text_frust), 0.0)
    den = den + np.where(has_t, f["text_weight"], 0.0)
    return num / den


def tension_index(series: Series, emo_cfg: dict, weights_fusion: dict | None = None) -> np.ndarray:
    co = emo_cfg["signals"]["tension_emotions"]
    a = np.zeros(series.n)
    for i, l in enumerate(series.labels):
        a += co.get(l, 0.0) * series.P[:, i]
    a = np.clip(a, 0, 1)
    f = weights_fusion or {"audio_weight": 1.0, "prosody_weight": 0.35}
    has_p = ~np.isnan(series.tension)
    num = f["audio_weight"] * a + np.where(has_p, f["prosody_weight"] * np.nan_to_num(series.tension), 0.0)
    den = f["audio_weight"] + np.where(has_p, f["prosody_weight"], 0.0)
    return num / den


def emotion_share(series: Series, names: list[str]) -> np.ndarray:
    idx = [i for i, l in enumerate(series.labels) if l in names]
    return series.P[:, idx].sum(axis=1) if idx else np.zeros(series.n)


def call_features(series: Series, comps: dict, emo_cfg: dict) -> np.ndarray:
    """Vector de features fijo para el modelo de satisfacción entrenable (orden estable)."""
    P = np.zeros((series.n, len(CANONICAL)))
    for i, l in enumerate(series.labels):
        if l in CANONICAL:
            P[:, CANONICAL.index(l)] = series.P[:, i]
    mean_p = P.mean(axis=0)
    third = max(1, series.n // 3)
    first, last = P[:third].mean(axis=0), P[-third:].mean(axis=0)
    frust = frustration_index(series, emo_cfg)
    tens = tension_index(series, emo_cfg)
    ts = series.text_score
    return np.concatenate([
        mean_p, first, last, last - first,
        [comps.get("signal", 0), comps.get("final_state", 0), comps.get("trend", 0), comps.get("persistence", 0),
         comps.get("stability", 0), float(np.nanmean(ts)) if np.any(~np.isnan(ts)) else 0.0,
         float(np.nanmean(series.tension)) if np.any(~np.isnan(series.tension)) else 0.5,
         float(frust.mean()), float(tens.mean()), float(frust.max()), float(np.log1p(series.n))]])


FEATURE_DIM = 7 * 4 + 11
