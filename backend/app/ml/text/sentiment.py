"""Análisis de texto: sentimiento y señales de contexto (frustración, resolución, cortesía) ES/EN.

`LexiconTextAnalyzer` es una implementación determinista basada en léxico + negación + fórmulas de cortesía.
Sus límites son claros: no detecta sarcasmo ni ironía y depende de la calidad de la transcripción.
Por eso su peso en la satisfacción es bajo y configurable, y una frase positiva NO implica satisfacción:
las fórmulas de cortesía ("gracias por llamar") se tratan como neutras.
`HFTextAnalyzer` (opcional) usa un modelo de sentimiento de Hugging Face.
Interfaz: cualquier `TextAnalyzer` (p. ej. un LLM local) puede sustituirlos.
"""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class TextSignal:
    score: float = 0.0              # -1 (negativo) .. +1 (positivo)
    confidence: float = 0.0         # 0..1 (según cantidad de evidencia)
    frustration: float = 0.0        # 0..1 (cues de frustración)
    resolution: float = 0.0         # 0..1 (cues de resolución/agradecimiento genuino)
    label: str = "neutral"
    cues: list[dict] = field(default_factory=list)   # [{type, phrase}]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("score", "confidence", "frustration", "resolution"):
            d[k] = round(d[k], 4)
        return d


class TextAnalyzer(ABC):
    name = "text"

    @abstractmethod
    def analyze(self, text: str, language: str | None = None) -> TextSignal: ...


def norm(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'¿?¡!]", " ", t)).strip()


# (patrón regex sobre texto normalizado sin acentos, peso)
FRUSTRATION = [
    (r"\bcansad[oa]s? de\b", 0.9), (r"\bharto\b|\bharta\b", 0.9), (r"\bllevo (?:\w+ ){0,3}(?:dias?|semanas?|meses|horas|anos)\b", 0.8),
    (r"\bnadie me\b", 0.8), (r"\bninguna solucion\b|\bnadie (?:me )?(?:soluciona|resuelve|ayuda)", 0.9),
    (r"\botra vez\b|\bde nuevo\b|\bnuevamente\b", 0.4), (r"\bya (?:llame|intente|escribi|reporte)\b", 0.6),
    (r"\binaceptable\b|\bpesimo\b|\bterrible\b|\bvergonzoso\b|\bfatal\b|\bimpresentable\b", 0.9),
    (r"\bno (?:me )?(?:sirve|funciona|ayuda|soluciona|resuelve)\b", 0.6),
    (r"\b(?:muy |bastante )?(?:enojad[oa]|molest[oa]|inconforme|decepcionad[oa]|indignad[oa]|furios[oa])\b", 0.9),
    (r"\bqueja\b|\breclamo\b|\bcancelar\b|\bme voy a quejar\b|\bquiero hablar con (?:un |el )?supervisor\b", 0.6),
    (r"\bmala atencion\b|\bmal servicio\b|\bpesim[oa] servicio\b|\bperdiendo (?:mi )?tiempo\b", 0.8),
    (r"\bnadie (?:me )?(?:da|dio|ha dado|responde|contesta)\b|\bme (?:tienen|traen|dan) (?:de un lado a otro|vueltas|largas)\b|\bsiempre (?:lo )?mismo\b", 0.8),
    (r"\bno (?:es|puede ser) posible\b|\bque (?:falta de|poca) (?:respeto|seriedad|atencion|verguenza)\b|\bes una burla\b|\bpesima atencion\b|\bme (?:cobraron|cobran) (?:de mas|dos veces|doble)\b", 0.85),
    (r"\bya no (?:aguanto|soporto|puedo mas)\b|\bse (?:pasaron|pasan)\b|\bhasta cuando\b|\bnunca (?:me )?(?:llaman|responden|solucionan|resuelven)\b", 0.85),
    (r"\bfed up\b|\bsick of\b|\bunacceptable\b|\bterrible\b|\bworst\b|\bridiculous\b|\bfrustrat(?:ed|ing)\b", 0.9),
    (r"\bstill (?:not|haven't|hasn't|no)\b|\bnobody\b|\bno one\b|\bwaiting for (?:\w+ ){0,2}(?:days?|hours?|weeks?)\b", 0.7),
    (r"\bangry\b|\bupset\b|\bdisappointed\b|\bcomplain(?:t)?\b|\bwasting (?:my )?time\b", 0.8),
]
POSITIVE = [
    (r"\bmuchas gracias\b|\bmil gracias\b|\bse lo agradezco\b|\bagradezco\b", 0.7),
    (r"\bperfecto\b|\bexcelente\b|\bgenial\b|\bmagnifico\b|\bestupendo\b", 0.8),
    (r"\bresuelto\b|\bsolucionado\b|\bquedo (?:resuelto|solucionado)\b|\bya funciona\b", 0.9),
    (r"\bme ayud(?:o|aron|aste)\b|\bmuy amable\b|\bbuen servicio\b|\bexcelente atencion\b", 0.9),
    (r"\bsatisfech[oa]\b|\bcontent[oa]\b|\bfeliz\b|\bencantad[oa]\b", 0.8),
    (r"\bthank you (?:so|very) much\b|\bthanks a lot\b|\bappreciate\b|\bperfect\b|\bgreat\b|\bawesome\b|\bexcellent\b", 0.8),
    (r"\b(?:issue|problem) (?:is )?(?:resolved|solved|fixed)\b|\bthat (?:works|helped)\b|\bvery helpful\b|\bhappy with\b", 0.9),
]
# Fórmulas de cortesía / guion: NO cuentan como satisfacción
POLITENESS = [
    r"\bgracias por (?:llamar|comunicarse|contactar|esperar|su llamada|su paciencia)\b", r"\ben que (?:le )?puedo ayudar\b",
    r"\bque tenga (?:un )?(?:buen|excelente|lindo) dia\b", r"\bcon (?:mucho )?gusto\b", r"\bhasta luego\b|\bbuenos dias\b|\bbuenas tardes\b|\bbuenas noches\b",
    r"\bthank you for (?:calling|holding|waiting)\b", r"\bhow (?:may|can) i help\b", r"\bhave a (?:nice|great|good) day\b",
]
NEGATIVE_WORDS = {"malo", "mala", "peor", "problema", "problemas", "error", "falla", "fallo", "demora", "demoras", "lento",
                  "lenta", "nunca", "jamas", "tarde", "molestia", "queja", "bad", "worse", "problem", "issue", "broken",
                  "slow", "never", "late", "wrong", "fail", "failed"}
POSITIVE_WORDS = {"bien", "bueno", "buena", "gracias", "claro", "correcto", "listo", "rapido", "rapida", "amable",
                  "excelente", "good", "great", "thanks", "thank", "clear", "fast", "nice", "helpful", "right", "okay"}
NEGATORS = {"no", "nunca", "jamas", "ni", "sin", "not", "never", "n't", "cannot", "dont", "didnt", "doesnt", "wasnt"}


class LexiconTextAnalyzer(TextAnalyzer):
    name = "lexicon"

    def analyze(self, text: str, language: str | None = None) -> TextSignal:
        t = norm(text)
        if len(t.split()) < 2:
            return TextSignal()
        cues: list[dict] = []
        polite_spans = [m.span() for p in POLITENESS for m in re.finditer(p, t)]

        def in_polite(span):
            return any(a <= span[0] and span[1] <= b for a, b in polite_spans)

        frust = pos = 0.0
        for pat, w in FRUSTRATION:
            for m in re.finditer(pat, t):
                frust = max(frust, w) if frust else w
                frust = min(1.0, frust + 0.15 * (1 if cues else 0))
                cues.append({"type": "frustration", "phrase": m.group(0)})
        for pat, w in POSITIVE:
            for m in re.finditer(pat, t):
                if in_polite(m.span()):
                    continue
                # negación inmediata ("no me ayudó") invierte la señal
                before = t[:m.start()].split()[-3:]
                if any(b in NEGATORS for b in before):
                    frust = max(frust, 0.5 * w)
                    cues.append({"type": "negated_positive", "phrase": " ".join(before) + " " + m.group(0)})
                else:
                    pos = max(pos, w)
                    cues.append({"type": "resolution", "phrase": m.group(0)})
        # léxico básico con negación
        toks = t.split()
        lex = 0.0
        for i, tk in enumerate(toks):
            neg = any(x in NEGATORS for x in toks[max(0, i - 3):i])
            if tk in NEGATIVE_WORDS:
                lex += 0.4 if not neg else -0.2
            elif tk in POSITIVE_WORDS:
                lex += (0.25 if not neg else -0.3)
        lex /= max(3.0, len(toks) ** 0.5)
        score = float(np.tanh(1.2 * (pos - frust) + lex))
        evidence = len(cues) + min(abs(lex), 1.0)
        conf = float(min(1.0, 0.25 + 0.25 * evidence)) if evidence > 0.2 else 0.15
        label = "negative" if score < -0.2 else "positive" if score > 0.2 else "neutral"
        return TextSignal(score, conf, float(min(frust, 1.0)), float(min(pos, 1.0)), label, cues[:8])


class HFTextAnalyzer(TextAnalyzer):
    """Sentimiento con un modelo de Hugging Face (p. ej. cardiffnlp/twitter-xlm-roberta-base-sentiment).
    Se combina con las cues del léxico para frustración/resolución."""
    name = "hf"

    def __init__(self, model_id: str):
        from transformers import pipeline
        from app.core.config import get_settings
        s = get_settings()
        self._pipe = pipeline("text-classification", model=model_id, top_k=None,
                              model_kwargs={"cache_dir": str(s.model_cache_dir)})
        self._lex = LexiconTextAnalyzer()

    def analyze(self, text: str, language: str | None = None) -> TextSignal:
        base = self._lex.analyze(text, language)
        if len(text.split()) < 3:
            return base
        res = self._pipe(text[:512])[0]
        p = {r["label"].lower(): r["score"] for r in res}
        pos = next((v for k, v in p.items() if "pos" in k), 0.0)
        neg = next((v for k, v in p.items() if "neg" in k), 0.0)
        score = float(pos - neg)
        base.score = 0.6 * score + 0.4 * base.score
        base.confidence = max(base.confidence, float(max(p.values())))
        base.label = "negative" if base.score < -0.2 else "positive" if base.score > 0.2 else "neutral"
        return base


_ANALYZERS: dict[str, TextAnalyzer] = {}


def get_text_analyzer(cfg: dict) -> TextAnalyzer:
    ts = cfg.get("text_sentiment", {})
    key = ts.get("engine", "lexicon") + ":" + ts.get("hf_model", "")
    if key not in _ANALYZERS:
        if ts.get("engine") == "hf":
            try:
                _ANALYZERS[key] = HFTextAnalyzer(ts["hf_model"])
            except Exception:
                _ANALYZERS[key] = LexiconTextAnalyzer()
        else:
            _ANALYZERS[key] = LexiconTextAnalyzer()
    return _ANALYZERS[key]
