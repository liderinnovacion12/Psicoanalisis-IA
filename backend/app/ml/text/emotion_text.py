"""Emoción a partir del TEXTO transcrito (por idioma) en el mismo espacio de 7 emociones que el modelo de audio.

Motivación: el modelo de audio base está entrenado en inglés; en español su señal es poco fiable. Un modelo de texto
NATIVO del idioma da una segunda evidencia independiente. Se FUSIONA con la de audio (ver satisfaction/fusion.py) y se
mide su concordancia; nunca se presenta como verdad.

Modelos por defecto (públicos, sin token): es -> pysentimiento/robertuito-emotion-analysis (RoBERTa, tweets en español);
en -> j-hartmann/emotion-english-distilroberta-base. Limitación: entrenados con texto escrito de redes/otros corpora, no con
transcripciones de llamadas; funcionan mejor con enunciados explícitos ("estoy molesto", "muchas gracias") que con ironía o
contexto implícito.
"""
from __future__ import annotations

import re
import threading

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
_LABEL_MAP = {"anger": "angry", "joy": "happy", "sadness": "sad", "others": "neutral", "other": "neutral", "neutral": "neutral",
              "disgust": "disgust", "fear": "fear", "surprise": "surprise", "angry": "angry", "happy": "happy", "sad": "sad",
              "happiness": "happy", "love": "happy"}
_PIPES: dict[str, object] = {}
_LOCK = threading.Lock()


def _pipeline(model_id: str):
    with _LOCK:
        if model_id not in _PIPES:
            from transformers import pipeline
            s = get_settings()
            if s.hf_offline:
                import os
                os.environ["HF_HUB_OFFLINE"] = "1"
            _PIPES[model_id] = pipeline("text-classification", model=model_id, top_k=None,
                                        model_kwargs={"cache_dir": str(s.model_cache_dir)})
            log.info("Modelo de emoción de texto cargado", extra={"model": model_id})
    return _PIPES[model_id]


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


class TextEmotionAnalyzer:
    """analyze_batch(textos, idioma) -> lista de {emoción: prob} (None si no hay modelo o el texto es muy corto)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("text_emotion", {})

    def model_for(self, language: str | None) -> str | None:
        if not self.cfg.get("enabled", True) or not language:
            return None
        return (self.cfg.get("models") or {}).get(language[:2].lower())

    def analyze_batch(self, texts: list[str], language: str | None) -> list[dict[str, float] | None]:
        mid = self.model_for(language)
        out: list[dict[str, float] | None] = [None] * len(texts)
        if mid is None:
            return out
        min_words = int(self.cfg.get("min_words", 3))
        idx = [i for i, t in enumerate(texts) if len(t.split()) >= min_words]
        if not idx:
            return out
        try:
            pipe = _pipeline(mid)
            bs = int(self.cfg.get("batch_size", 32))
            for k in range(0, len(idx), bs):
                chunk = idx[k:k + bs]
                res = pipe([_clean(texts[i])[:1500] for i in chunk], truncation=True, max_length=int(self.cfg.get("max_length", 256)),
                           batch_size=bs)
                for i, r in zip(chunk, res):
                    d = {c: 0.0 for c in CANONICAL}
                    for x in r:
                        lab = _LABEL_MAP.get(str(x["label"]).lower())
                        if lab:
                            d[lab] += float(x["score"])
                    tot = sum(d.values()) or 1.0
                    out[i] = {k2: v / tot for k2, v in d.items()}
        except Exception as e:                       # sin red / sin modelo: el análisis continúa solo con audio
            log.warning("emoción de texto no disponible", extra={"model": mid, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            return [None] * len(texts)
        return out
