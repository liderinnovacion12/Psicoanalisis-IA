"""Análisis de calidad de audio (volumen, silencios, ruido, clipping) y detección de canales.

Todo se calcula en streaming por bloques: memoria O(1) respecto a la duración.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ml.audio.io import iter_blocks

FRAME_S = 0.025
EPS = 1e-10


def _frame_db(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([], dtype=np.float32)
    f = x[: n * frame].reshape(n, frame)
    return (10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + EPS)).astype(np.float32)


def analyze_levels(path: str | Path, block_seconds: float = 300.0, sample_rate: int = 16000) -> dict:
    """Métricas globales por streaming. Trabaja sobre la mezcla mono de los canales presentes."""
    frame = int(FRAME_S * sample_rate)
    dbs: list[np.ndarray] = []
    peak, clip, total, sumsq = 0.0, 0, 0, 0.0
    for _, blk in iter_blocks(path, block_seconds):
        x = blk.mean(axis=1) if blk.shape[1] > 1 else blk[:, 0]
        peak = max(peak, float(np.max(np.abs(blk))))
        clip += int(np.sum(np.abs(blk) >= 0.999))
        total += blk.size
        sumsq += float(np.sum(x.astype(np.float64) ** 2))
        dbs.append(_frame_db(x, frame))
    db = np.concatenate(dbs) if dbs else np.array([-120.0])
    n_samples = max(1, total // max(1, blk.shape[1] if total else 1))
    rms_db = 10 * np.log10(sumsq / n_samples + EPS)
    p10, p50, p90 = np.percentile(db, [10, 50, 90])
    silence_thr = max(-50.0, float(p90) - 35.0)
    return {
        "peak_dbfs": round(20 * np.log10(peak + EPS), 2),
        "rms_dbfs": round(float(rms_db), 2),
        "noise_floor_dbfs": round(float(p10), 2),
        "speech_level_dbfs": round(float(p90), 2),
        "snr_db": round(float(p90 - p10), 2),                 # estimación por percentiles (aprox.)
        "silence_ratio": round(float(np.mean(db < silence_thr)), 4),
        "clipping_ratio": round(clip / max(total, 1), 6),
        "frame_count": int(len(db)),
    }


def detect_channel_layout(path: str | Path, cfg: dict, block_seconds: float = 300.0,
                          sample_rate: int = 16000) -> dict:
    """Decide si un WAV estéreo contiene un hablante por canal.

    Criterios (configurables en audio_config.yaml > channels):
      * correlación de muestras entre canales baja (no es dual-mono/mezcla);
      * ambos canales tienen voz suficiente;
      * el solapamiento de actividad es bajo respecto al canal menos activo.
    """
    frame = int(FRAME_S * sample_rate)
    sl = sr = sll = srr = slr = 0.0
    n = 0
    dbl, dbr = [], []
    for _, blk in iter_blocks(path, block_seconds):
        if blk.shape[1] < 2:
            return {"is_stereo_split": False, "reason": "mono", "correlation": None}
        l, r = blk[:, 0].astype(np.float64), blk[:, 1].astype(np.float64)
        sl += l.sum(); sr += r.sum(); sll += (l * l).sum(); srr += (r * r).sum(); slr += (l * r).sum()
        n += len(l)
        m = min(len(_frame_db(l, frame)), len(_frame_db(r, frame)))
        dbl.append(_frame_db(l.astype(np.float32), frame)[:m])
        dbr.append(_frame_db(r.astype(np.float32), frame)[:m])
    if n == 0:
        return {"is_stereo_split": False, "reason": "empty", "correlation": None}
    cov = slr / n - (sl / n) * (sr / n)
    vl, vr = sll / n - (sl / n) ** 2, srr / n - (sr / n) ** 2
    corr = float(cov / (np.sqrt(max(vl, 0) * max(vr, 0)) + EPS))
    dl, dr = np.concatenate(dbl), np.concatenate(dbr)

    def active(db: np.ndarray) -> np.ndarray:
        p95 = np.percentile(db, 95)
        return (db > max(-55.0, p95 - 25.0)) & (db > -60)

    al, ar = active(dl), active(dr)
    ratio_l, ratio_r = float(al.mean()), float(ar.mean())
    both = float(np.sum(al & ar))
    least = max(1.0, float(min(al.sum(), ar.sum())))
    overlap_ratio = both / least
    dual_mono = abs(corr) > cfg["max_correlation"] and overlap_ratio > 0.6
    ok = (not dual_mono and abs(corr) <= cfg["max_correlation"]
          and min(ratio_l, ratio_r) >= cfg["min_active_ratio"]
          and overlap_ratio <= cfg["max_overlap_ratio"])
    reason = ("un hablante por canal" if ok else
              "canales correlacionados (mezcla/dual-mono)" if abs(corr) > cfg["max_correlation"] else
              "un canal casi sin voz" if min(ratio_l, ratio_r) < cfg["min_active_ratio"] else
              "actividad simultánea alta en ambos canales")
    return {"is_stereo_split": bool(ok), "reason": reason, "correlation": round(corr, 4),
            "active_ratio": [round(ratio_l, 4), round(ratio_r, 4)], "overlap_ratio": round(overlap_ratio, 4)}


def quality_score(meta: dict, levels: dict, speech_ratio: float, cfg: dict) -> dict:
    """Score 0-100 con desglose. Cada componente 0-1; pesos en audio_config.yaml > quality.weights."""
    q = cfg["quality"]
    w = q["weights"]
    snr = levels["snr_db"]
    c_snr = float(np.clip((snr - q["min_snr_db"]) / max(q["good_snr_db"] - q["min_snr_db"], 1e-6), 0, 1))
    c_clip = float(np.clip(1 - levels["clipping_ratio"] / max(q["clipping_ratio_warn"] * 10, 1e-9), 0, 1))
    lvl = levels["rms_dbfs"]
    c_level = 1.0 if -30 <= lvl <= -12 else float(np.clip(1 - (min(abs(lvl + 30), abs(lvl + 12)) / 25), 0, 1))
    c_speech = float(np.clip(speech_ratio / max(q["min_speech_ratio"] * 3, 1e-6), 0, 1))
    sr = meta.get("sample_rate") or 16000
    c_bw = float(np.clip((sr - q["min_sample_rate"]) / (16000 - q["min_sample_rate"]), 0, 1))
    comps = {"snr": c_snr, "clipping": c_clip, "level": c_level, "speech": c_speech, "bandwidth": c_bw}
    score = 100 * sum(comps[k] * w[k] for k in comps) / sum(w.values())
    warnings = []
    if levels["silence_ratio"] > q["silence_ratio_warn"]:
        warnings.append("Proporción de silencio muy alta.")
    if levels["clipping_ratio"] > q["clipping_ratio_warn"]:
        warnings.append("Se detecta saturación (clipping) en el audio.")
    if snr < q["min_snr_db"]:
        warnings.append("Nivel de ruido elevado respecto a la voz.")
    if speech_ratio < q["min_speech_ratio"]:
        warnings.append("Se detecta muy poca voz en la grabación.")
    if sr < 16000:
        warnings.append("Ancho de banda reducido (muestreo original bajo).")
    return {"score": round(score, 1), "components": {k: round(v, 3) for k, v in comps.items()},
            "low": score < q["low_quality_threshold"], "warnings": warnings}
