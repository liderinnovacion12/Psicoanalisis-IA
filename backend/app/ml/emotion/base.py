"""Interfaz abstracta de modelos de emoción. La aplicación SOLO conoce esta interfaz.

    EmotionModel
        ├── Wav2VecEmotionModel      (baseline Hugging Face)          -> ml/emotion/wav2vec.py
        ├── FineTunedEmotionModel    (checkpoint local del registry)  -> ml/emotion/wav2vec.py
        └── <FutureEmotionModel>     (nuevo modelo: subclase + register_loader("nombre", Clase))
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionOutput:
    emotion: str                       # emoción dominante
    confidence: float                  # probabilidad de la dominante (NO es satisfacción)
    probabilities: dict[str, float]    # distribución COMPLETA


class EmotionModel(ABC):
    sample_rate: int = 16000

    def __init__(self, name: str, version: str = "v1", language: str | None = None):
        self.name = name
        self.version = version
        self.language = language
        self.labels: list[str] = []
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Carga pesos (desde caché local; descarga solo la primera vez)."""

    @abstractmethod
    def predict(self, waveforms: list[np.ndarray]) -> list[EmotionOutput]:
        """waveforms: float32 mono a `sample_rate`. Devuelve una salida por forma de onda."""

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "language": self.language,
                "labels": self.labels, "sample_rate": self.sample_rate}

    def ensure_loaded(self) -> "EmotionModel":
        if not self.loaded:
            self.load()
            self.loaded = True
        return self
