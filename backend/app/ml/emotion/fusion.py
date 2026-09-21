"""Fusión de la emoción de AUDIO con la de TEXTO por ventana, con pesos que dependen del idioma.

* Si el idioma de la llamada coincide con el del modelo de audio: más peso al audio.
* Si NO coincide (p. ej. modelo inglés, llamada en español): más peso al texto, cuyo modelo es nativo del idioma.
Se guarda cada fuente por separado y la CONCORDANCIA audio-texto (misma emoción principal + masa del texto sobre las 2 emociones
top del audio): si discrepan mucho, la confianza baja en lugar de esconder el desacuerdo.
Los pesos son configuración (emotion_config.yaml > label_fusion), no verdad calibrada: no hay datos etiquetados de llamadas
en español con los que ajustarlos todavía.
"""
from __future__ import annotations

import numpy as np


def temper(probs: dict[str, float], T: float) -> dict[str, float]:
    """Equivale a softmax(logits / T): como p ∝ exp(z), softmax(z/T) ∝ p^(1/T). T > 1 aplana probabilidades sobreconfiadas."""
    if T is None or abs(T - 1.0) < 1e-6:
        return probs
    v = {k: max(float(p), 1e-12) ** (1.0 / T) for k, p in probs.items()}
    s = sum(v.values())
    return {k: x / s for k, x in v.items()}


def weights_for(language_match: bool, cfg: dict) -> tuple[float, float]:
    lf = cfg.get("label_fusion", {})
    w = lf.get("same_language" if language_match else "language_mismatch", {"audio": 0.7 if language_match else 0.35,
                                                                             "text": 0.3 if language_match else 0.65})
    return float(w["audio"]), float(w["text"])


def fuse(audio: dict[str, float], text: dict[str, float] | None, wa: float, wt: float) -> tuple[dict[str, float], dict]:
    labels = list(audio)
    a = np.array([audio[l] for l in labels], dtype=float)
    base_src = {"audio": {k: round(float(v), 5) for k, v in audio.items()}, "text": None,
                "weights": {"audio": 1.0, "text": 0.0}, "agreement": None}
    if text is None or wt <= 0:
        return dict(audio), base_src
    t = np.array([text.get(l, 0.0) for l in labels], dtype=float)
    if t.sum() <= 1e-6:
        return dict(audio), base_src
    t = t / t.sum()                                           # renormaliza sobre las etiquetas del modelo de audio
    f = wa * a + wt * t
    f = f / f.sum()
    # Concordancia: ¿coinciden en la emoción principal? y ¿cuánta probabilidad da el texto a las 2 emociones top del audio?
    # (no compara distribuciones completas: el audio puede estar aplanado a propósito por la corrección de temperatura)
    top2 = np.argsort(a)[::-1][:2]
    agreement = float(0.5 * (np.argmax(a) == np.argmax(t)) + 0.5 * t[top2].sum())
    src = {"audio": base_src["audio"], "text": {l: round(float(v), 5) for l, v in zip(labels, t)},
           "weights": {"audio": round(wa / (wa + wt), 3), "text": round(wt / (wa + wt), 3)}, "agreement": round(agreement, 4)}
    return {l: float(v) for l, v in zip(labels, f)}, src


class TextTrack:
    """Emociones de texto de UN hablante, consultables por intervalo (media ponderada por solape)."""

    def __init__(self, items: list[tuple[float, float, dict[str, float]]]):
        items = [i for i in items if i[2]]
        items.sort(key=lambda x: x[0])
        self.s = np.array([i[0] for i in items]) if items else np.zeros(0)
        self.e = np.array([i[1] for i in items]) if items else np.zeros(0)
        self.labels = sorted({k for i in items for k in i[2]}) if items else []
        self.P = np.array([[i[2].get(l, 0.0) for l in self.labels] for i in items]) if items else np.zeros((0, 0))

    def query(self, t0: float, t1: float) -> dict[str, float] | None:
        if not len(self.s):
            return None
        ov = np.clip(np.minimum(self.e, t1) - np.maximum(self.s, t0), 0, None)
        tot = ov.sum()
        if tot < 0.25:                                            # menos de 0.25 s de texto en la ventana: sin evidencia
            return None
        p = (self.P * ov[:, None]).sum(0) / tot
        return {l: float(v) for l, v in zip(self.labels, p)}
