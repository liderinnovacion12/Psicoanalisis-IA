"""Prosodia por ventana: energía, pitch (F0), variabilidad, proporción de pausas.

F0 se estima con `librosa.yin` (rápido) y una compuerta de energía como voicing. Son features
ADICIONALES: no producen por sí solas emociones ni satisfacción.
"""
from __future__ import annotations

import numpy as np


def window_prosody(x: np.ndarray, sr: int = 16000, f0_min: float = 65, f0_max: float = 500) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < sr // 4:
        return {}
    frame = int(0.032 * sr)
    hop = frame // 2
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    thr = max(db.max() - 30, -60)
    voiced_e = db > thr
    out = {
        "rms_db": float(np.mean(db[voiced_e])) if voiced_e.any() else float(db.mean()),
        "energy_std_db": float(np.std(db[voiced_e])) if voiced_e.sum() > 1 else 0.0,
        "pause_ratio": float(1 - voiced_e.mean()),
    }
    try:
        import librosa
        f0 = librosa.yin(x, fmin=f0_min, fmax=f0_max, sr=sr, frame_length=1024, hop_length=hop)
        m = min(len(f0), len(voiced_e))
        f0, ve = f0[:m], voiced_e[:m]
        f0v = f0[ve & (f0 > f0_min * 1.05) & (f0 < f0_max * 0.95)]
        if len(f0v) >= 5:
            f0v = f0v[(f0v > np.percentile(f0v, 5)) & (f0v < np.percentile(f0v, 95))]
            out["f0_mean"] = float(np.mean(f0v))
            out["f0_std"] = float(np.std(f0v))
            out["f0_range_st"] = float(12 * np.log2((np.percentile(f0v, 95) + 1e-6) / (np.percentile(f0v, 5) + 1e-6)))
    except Exception:  # pragma: no cover
        pass
    return out


def tension_from_prosody(p: dict, baseline: dict) -> float | None:
    """Proxy 0-1 de 'tensión prosódica' respecto a la línea base del propio hablante.

    z(energía) + z(pitch) + z(variabilidad de pitch). Heurístico: sin datos etiquetados no se
    puede afirmar que mida tensión real; es una señal auxiliar con peso configurable.
    """
    if not p or "f0_mean" not in p or not baseline:
        return None
    def z(v, key):
        m, s = baseline.get(key, (v, 1.0))
        return (v - m) / (s if s > 1e-6 else 1.0)
    score = 0.5 * z(p["rms_db"], "rms_db") + 0.4 * z(p["f0_mean"], "f0_mean") + 0.3 * z(p.get("f0_std", 0), "f0_std")
    return float(1 / (1 + np.exp(-1.2 * score)))


def build_baseline(items: list[dict]) -> dict:
    """{key: (mediana, MAD-escalado)} por hablante."""
    base = {}
    for key in ("rms_db", "f0_mean", "f0_std"):
        vals = np.array([i[key] for i in items if key in i], dtype=np.float64)
        if len(vals) >= 3:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)) * 1.4826)
            base[key] = (med, max(mad, 1e-3))
    return base
