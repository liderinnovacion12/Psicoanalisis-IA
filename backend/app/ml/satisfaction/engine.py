"""SATISFACTION ENGINE — independiente del modelo de emociones.

NO equivale a "happy = satisfecho / angry = insatisfecho". Combina, con pesos de configuración:
señal emocional (duración·intensidad·confianza), estado final, tendencia, persistencia de estados
negativos, estabilidad, sentimiento textual y tensión prosódica. Devuelve 0-100, una confianza PROPIA
(distinta de la probabilidad emocional) y factores explicables (puntos aportados por cada señal).

score = base + scale · clip( sensibilidad · Σ wᵢ·cᵢ / Σ wᵢ , -1, 1 )      (cᵢ ∈ [-1, 1])
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.satisfaction.features import (Series, call_features, emotion_share, frustration_index, smooth,
                                          tension_index, weights_vector)

ENGINE_VERSION = "rules-1.0"

COMPONENT_LABELS = {
    "signal": ("Predominio de señales emocionales positivas", "Predominio de señales emocionales negativas"),
    "final_state": ("Estado emocional positivo al final de la llamada", "Estado emocional negativo al final de la llamada"),
    "trend": ("Tendencia positiva durante la interacción", "Tendencia negativa durante la interacción"),
    "persistence": ("Persistencia de estados positivos", "Persistencia de estados negativos"),
    "stability": ("Conversación emocionalmente estable", "Alta variabilidad emocional"),
    "text": ("Contexto textual favorable", "Contexto textual desfavorable (posibles expresiones de queja)"),
    "prosody": ("Prosodia sin señales marcadas de tensión", "Prosodia con señales de tensión"),
}


@dataclass
class SpeakerAnalysis:
    speaker: str
    score: float
    confidence: float
    trend: str
    initial_score: float | None
    final_score: float | None
    timeline: list[dict]
    components: dict
    contributions: dict
    factors: dict
    metrics: dict
    interpretation: dict
    details: dict = field(default_factory=dict)


def interpret(score: float, ranges: list[dict]) -> dict:
    for r in ranges:
        if r["min"] <= round(score) <= r["max"]:
            return {"key": r["key"], "label": r["label"]}
    return {"key": "unknown", "label": "Sin interpretación"}


def confidence_level(conf: float, thresholds=(0.75, 0.5)) -> str:
    return "alta" if conf >= thresholds[0] else "moderada" if conf >= thresholds[1] else "baja"


def _runs(mask: np.ndarray, min_len: int) -> int:
    """Nº de celdas pertenecientes a rachas True de longitud >= min_len."""
    total, run = 0, 0
    for m in np.append(mask, False):
        if m:
            run += 1
        else:
            if run >= min_len:
                total += run
            run = 0
    return total


class SatisfactionEngine:
    def __init__(self, sat_cfg: dict, emo_cfg: dict, regressor=None):
        self.cfg = sat_cfg
        self.emo = emo_cfg
        self.w = sat_cfg["weights"]
        self.regressor = regressor

    # ---- componentes ------------------------------------------------------------------------------
    def components(self, s: Series, final_fraction: float | None = None, final_min: float | None = None,
                   final_max: float | None = None) -> dict:
        """Componentes cᵢ ∈ [-1,1] (None si no aplica) para una serie completa o un tramo."""
        w, c = self.w, self.cfg
        if s.n == 0:
            return {}
        wv = weights_vector(s.labels, w)
        v = s.P @ wv                                                     # valencia por celda
        p_neutral = s.P[:, s.labels.index("neutral")] if "neutral" in s.labels else np.zeros(s.n)
        inten = 1 - p_neutral
        a = (1 + w["intensity_gain"] * inten) * (0.5 + 0.5 * s.conf)
        signal = float(np.sum(a * v) / np.sum(a))

        fs = c["final_segment"]
        frac = fs["fraction"] if final_fraction is None else final_fraction
        span = float(s.t[-1] - s.t[0]) + s.dt
        L = float(np.clip(frac * span, final_min if final_min is not None else fs["min_seconds"],
                          final_max if final_max is not None else fs["max_seconds"]))
        fmask = s.t >= s.t[-1] - L
        if fmask.sum() < 3:
            fmask = np.zeros(s.n, bool); fmask[-min(s.n, 10):] = True
        final_state = float(np.sum(a[fmask] * v[fmask]) / np.sum(a[fmask]))

        k = max(1, int(c["trend"]["smoothing_seconds"] / s.dt))
        vs = smooth(v, k)
        if s.n >= 6 and span > 20:
            tn = (s.t - s.t[0]) / max(s.t[-1] - s.t[0], 1e-6)
            slope = float(np.polyfit(tn, vs, 1)[0])
        else:
            slope = 0.0
        trend = float(np.clip(slope * c["trend"].get("gain", 1.5), -1, 1))

        pc = c["persistence"]
        min_run = max(1, int(pc["min_run_seconds"] / s.dt))
        neg = _runs(vs < pc["negative_valence_threshold"], min_run) / s.n
        pos = _runs(vs > pc.get("positive_valence_threshold", 0.25), min_run) / s.n
        persistence = float(np.clip((pos - neg) * 1.5, -1, 1))

        std = float(np.nanstd(vs))
        jumps = float(np.mean(np.abs(np.diff(v)) > 0.5)) if s.n > 1 else 0.0
        B = float(np.clip(1 - std / 0.5 - jumps, 0, 1))
        stability = (2 * B - 1) * (0.5 + 0.5 * float(np.clip(signal * 2, -1, 1)))

        out = {"signal": signal, "final_state": final_state, "trend": trend, "persistence": persistence,
               "stability": stability, "_stability_raw": B}
        ts = s.text_score[~np.isnan(s.text_score)]
        if len(ts) >= 3:
            out["text"] = float(np.clip(np.mean(ts) * 1.5, -1, 1))
        tn_ = s.tension[~np.isnan(s.tension)]
        if len(tn_) >= 3:
            out["prosody"] = float(np.clip(-(np.mean(tn_) - 0.5) * 2, -1, 1))
        return out

    def _weight_for(self, comp: str) -> float:
        return self.w[{"signal": "signal_weight", "final_state": "final_state_weight", "trend": "trend_weight",
                       "persistence": "persistence_weight", "stability": "stability_weight",
                       "text": "text_weight", "prosody": "prosody_weight"}[comp]]

    def combine(self, comps: dict) -> tuple[float, dict]:
        sc = self.cfg["score"]
        active = {k: v for k, v in comps.items() if not k.startswith("_") and v is not None}
        wsum = sum(abs(self._weight_for(k)) for k in active) or 1.0
        comp = sum(self._weight_for(k) * v for k, v in active.items()) / wsum
        raw = sc["base"] + sc["scale"] * float(np.clip(sc["sensitivity"] * comp, -1, 1))
        contrib = {k: sc["scale"] * sc["sensitivity"] * self._weight_for(k) * v / wsum for k, v in active.items()}
        return float(np.clip(raw, 0, 100)), contrib

    # ---- línea de tiempo --------------------------------------------------------------------------
    def timeline(self, s: Series, duration: float) -> list[dict]:
        tl = self.cfg["timeline"]
        step, W = tl["step_seconds"], tl["window_seconds"]
        alpha = tl["smoothing_alpha"]
        out, ema = [], None
        for tk in np.arange(0.0, duration + step, step):
            sub = s.slice(tk - W, tk + 1e-6)
            if sub.n < 3:
                if ema is None:
                    continue
                score = ema
            else:
                comps = self.components(sub, final_fraction=0.4, final_min=5, final_max=W)
                score, _ = self.combine({k: v for k, v in comps.items() if k != "trend"} | {"trend": comps["trend"] * 0.5})
                ema = score if ema is None else alpha * score + (1 - alpha) * ema
                score = ema
            out.append({"t": round(float(min(tk, duration)), 2), "score": round(float(score), 2)})
        return out

    # ---- confianza propia ---------------------------------------------------------------------------
    def confidence(self, s: Series, audio_quality: float | None, language_mismatch: bool = False) -> tuple[float, dict]:
        cc = self.cfg["confidence"]
        w = cc["weights"]
        seconds = s.n * s.dt
        if seconds < cc["min_analyzed_seconds"]:
            cover = 0.2 * seconds / cc["min_analyzed_seconds"]
        else:
            cover = 0.4 + 0.6 * min(1.0, (seconds - cc["min_analyzed_seconds"]) /
                                    max(cc["full_confidence_seconds"] - cc["min_analyzed_seconds"], 1))
        model = float(np.clip((float(np.mean(s.conf)) - 0.3) / 0.5, 0, 1))
        q = 0.7 if audio_quality is None else float(np.clip(audio_quality / 100, 0, 1))
        txt = float(np.mean(~np.isnan(s.text_score))) if s.n else 0.0
        parts = {"model": model, "coverage": float(cover), "audio_quality": q, "text": txt}
        conf = sum(w[k] * parts[k] for k in parts) / sum(w.values())
        if language_mismatch:
            conf *= cc.get("language_mismatch_penalty", 0.6)
            parts["language_mismatch"] = True
        return float(np.clip(conf, 0, 1)), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in parts.items()}

    # ---- análisis de un hablante ------------------------------------------------------------------------
    def analyze_speaker(self, s: Series | None, duration: float, audio_quality: float | None = None,
                        speaker: str = "", language_mismatch: bool = False) -> SpeakerAnalysis | None:
        if s is None or s.n < 3:
            return None
        comps = self.components(s)
        score_rules, contrib = self.combine(comps)
        score = score_rules
        mode = self.cfg.get("mode", "rules")
        details: dict = {"engine_version": ENGINE_VERSION, "mode": mode, "score_rules": round(score_rules, 2)}
        if mode in ("model", "hybrid") and self.regressor is not None:
            pred = float(np.clip(self.regressor.predict(call_features(s, comps, self.emo)), 0, 100))
            details["score_model"] = round(pred, 2)
            a = 1.0 if mode == "model" else self.cfg.get("hybrid_alpha", 0.5)
            score = a * pred + (1 - a) * score_rules
        cal = self.cfg.get("calibration", {})
        if cal.get("enabled"):
            co = cal.get("coefficients", {"a": 1.0, "b": 0.0})
            details["score_uncalibrated"] = round(score, 2)
            score = float(np.clip(co["a"] * score + co["b"], 0, 100))

        tl = self.timeline(s, duration)
        ini_win = self.cfg["timeline"]["window_seconds"]
        first_t, last_t = float(s.t[0]), float(s.t[-1])
        sc_ini = self._local_score(s.slice(first_t - 1, first_t + ini_win))
        sc_fin = self._local_score(s.slice(last_t - ini_win, last_t + 1))
        delta = (sc_fin - sc_ini) if (sc_ini is not None and sc_fin is not None) else 0.0
        stable = self.cfg["trend"]["stable_delta"]
        trend = "improving" if delta > stable else "declining" if delta < -stable else "stable"

        conf, conf_parts = self.confidence(s, audio_quality, language_mismatch)
        details["confidence_parts"] = conf_parts
        pos_f, neg_f = [], []
        for k, pts in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            if abs(pts) < 1.0:
                continue
            (pos_f if pts > 0 else neg_f).append({"key": k, "label": COMPONENT_LABELS[k][0 if pts > 0 else 1],
                                                  "points": round(pts, 1)})
        metrics = self.metrics(s, comps)
        return SpeakerAnalysis(speaker or s.speaker, round(score, 1), round(conf, 3), trend,
                               None if sc_ini is None else round(sc_ini, 1),
                               None if sc_fin is None else round(sc_fin, 1), tl,
                               {k: (None if v is None else round(v, 4)) for k, v in comps.items()},
                               {k: round(v, 2) for k, v in contrib.items()},
                               {"positive": pos_f, "negative": neg_f}, metrics,
                               interpret(score, self.cfg["interpretation"]), details)

    def _local_score(self, s: Series) -> float | None:
        if s.n < 3:
            return None
        comps = self.components(s, final_fraction=0.4, final_min=5, final_max=self.cfg["timeline"]["window_seconds"])
        sc, _ = self.combine(comps | {"trend": comps["trend"] * 0.5})
        return sc

    # ---- métricas por hablante (tarjeta PERSONA) ------------------------------------------------------------
    def metrics(self, s: Series, comps: dict) -> dict:
        sig = self.emo["signals"]
        frust = frustration_index(s, self.emo, self.emo.get("fusion"))
        tens = tension_index(s, self.emo, self.emo.get("fusion"))
        pos = emotion_share(s, sig["positive_emotions"])
        neg = emotion_share(s, sig["negative_emotions"])
        mean_p = s.P.mean(axis=0)
        dom = s.labels[int(np.argmax(mean_p))]
        n_ini = max(3, int(30 / s.dt)); n_fin = max(3, int(min(60, 0.2 * s.n * s.dt) / s.dt))
        ini = s.labels[int(np.argmax(s.P[:n_ini].mean(axis=0)))]
        fin = s.labels[int(np.argmax(s.P[-n_fin:].mean(axis=0)))]
        return {
            "dominant_emotion": dom, "initial_emotion": ini, "final_emotion": fin,
            "frustration": round(float(frust.mean()), 4), "tension": round(float(tens.mean()), 4),
            "positive": round(float(pos.mean()), 4), "negative": round(float(neg.mean()), 4),
            "stability": round(float(comps.get("_stability_raw", 0.0)), 4),
            "mean_probabilities": {l: round(float(p), 4) for l, p in zip(s.labels, mean_p)},
            "analyzed_seconds": round(s.n * s.dt, 1),
            "mean_model_confidence": round(float(np.mean(s.conf)), 4),
        }
