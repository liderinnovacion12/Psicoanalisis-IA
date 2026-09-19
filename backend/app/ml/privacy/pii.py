"""Detección y anonimización de PII en transcripciones (ES/EN) por reglas.

Entidades: PERSON, PHONE, EMAIL, ID, ADDRESS, ACCOUNT. Límites conocidos: la detección de nombres
por reglas es heurística (cobertura parcial); se puede sustituir por NER (spaCy) implementando
`PIIDetector`. NO se anonimiza el audio (solo texto).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

PLACEHOLDERS = {"PERSON": "[PERSONA]", "PHONE": "[TELEFONO]", "EMAIL": "[CORREO]", "ID": "[DOCUMENTO]",
                "ADDRESS": "[DIRECCION]", "ACCOUNT": "[CUENTA]"}


@dataclass
class PIISpan:
    entity: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict:
        return {"entity": self.entity, "start": self.start, "end": self.end}   # no se guarda el texto sensible


class PIIDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> list[PIISpan]: ...


def _luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s += d
        alt = not alt
    return s % 10 == 0


NAME = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"
_WORD_NUM = r"(?:cero|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|zero|one|two|three|four|five|six|seven|eight|nine)"

_RULES: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("EMAIL", re.compile(r"\b[\w.+-]+\s+(?:arroba|at)\s+[\w-]+\s+(?:punto|dot)\s+\w+\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]{0,2})?(?:\(?\d{2,4}\)?[\s.-]{0,2})\d{3,4}[\s.-]{0,2}\d{3,4}(?!\d)")),
    ("PHONE", re.compile(rf"\b(?:{_WORD_NUM}[\s,-]+){{6,}}{_WORD_NUM}\b", re.I)),          # dígitos dictados
    ("ID", re.compile(r"\b(?:dni|cedula|cédula|cc|curp|rfc|nif|nie|pasaporte|passport|ssn|documento)\b[\s:#nºo.]*[A-Z0-9-]{5,}", re.I)),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ACCOUNT", re.compile(r"\b(?:cuenta|tarjeta|account|card|iban|clabe)\b[\s:#nºo.]*(?:\d[\s-]?){6,}", re.I)),
    ("ADDRESS", re.compile(rf"\b(?:calle|avenida|av\.|carrera|cra\.?|diagonal|transversal|street|st\.|avenue|road|boulevard)\s+"
                           rf"[\w\s#°º.-]{{2,40}}?\d+[\w-]*", re.I)),
    ("PERSON", re.compile(rf"\b(?i:me llamo|mi nombre es|soy|habla|le habla|mi apellido es|my name is|this is|i am|i'm)\s+"
                          rf"({NAME}(?:\s+(?:de|del|la|los|las)?\s*{NAME}){{0,3}})")),
    ("PERSON", re.compile(rf"\b(?i:se[nñ]or(?:a|ita)?|sr\.?|sra\.?|srta\.?|don|do[nñ]a|mr\.?|mrs\.?|ms\.?|dr\.?|dra\.?)\s+({NAME}(?:\s+{NAME}){{0,2}})")),
]


class RegexPIIDetector(PIIDetector):
    def __init__(self, entities: list[str] | None = None):
        self.entities = set(entities or PLACEHOLDERS)

    def detect(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for ent, rx in _RULES:
            if ent not in self.entities:
                continue
            for m in rx.finditer(text):
                s, e = (m.start(1), m.end(1)) if ent == "PERSON" and m.lastindex else (m.start(), m.end())
                raw = text[s:e]
                digits = re.sub(r"\D", "", raw)
                if ent == "PHONE" and not (7 <= len(digits) <= 15) and not re.search(_WORD_NUM, raw, re.I):
                    continue
                if ent == "ACCOUNT" and len(digits) >= 13 and not _luhn(digits) and len(digits) not in (16, 18, 20, 22):
                    continue
                spans.append(PIISpan(ent, s, e, raw))
        # resolver solapes: gana el más largo
        spans.sort(key=lambda x: (x.start, -(x.end - x.start)))
        merged: list[PIISpan] = []
        for sp in spans:
            if merged and sp.start < merged[-1].end:
                if sp.end - sp.start > merged[-1].end - merged[-1].start:
                    merged[-1] = sp
                continue
            merged.append(sp)
        return merged


def anonymize(text: str, spans: list[PIISpan]) -> str:
    out, last = [], 0
    for sp in sorted(spans, key=lambda s: s.start):
        out.append(text[last:sp.start])
        out.append(PLACEHOLDERS.get(sp.entity, "[DATO]"))
        last = sp.end
    out.append(text[last:])
    return "".join(out)


def redact_text(text: str, entities: list[str] | None = None) -> tuple[str, list[dict]]:
    spans = RegexPIIDetector(entities).detect(text)
    return anonymize(text, spans), [s.to_dict() for s in spans]
