"""Implementación genérica sobre `transformers` (AutoModelForAudioClassification)."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from app.core.config import get_settings
from app.core.device import detect_device
from app.core.logging import get_logger
from app.ml.emotion.base import EmotionModel, EmotionOutput

log = get_logger(__name__)

_ALIASES = {"ang": "angry", "anger": "angry", "hap": "happy", "happiness": "happy", "joy": "happy",
            "neu": "neutral", "sadness": "sad", "sad": "sad", "fea": "fear", "dis": "disgust",
            "sur": "surprise", "surprised": "surprise", "calm": "calm", "fearful": "fear"}
MIN_SAMPLES = 8000       # 0.5 s a 16 kHz


def normalize_label(label: str) -> str:
    l = str(label).strip().lower()
    return _ALIASES.get(l, l)


class HFAudioEmotionModel(EmotionModel):
    def __init__(self, name: str, source: str, version: str = "v1", language: str | None = None,
                 max_seconds: float = 30.0, device: str | None = None, fp16: bool = True,
                 temperature: float = 1.0):
        super().__init__(name, version, language)
        self.source = source                  # id de Hugging Face o directorio local
        self.max_seconds = max_seconds
        self._device_pref = device
        self._fp16 = fp16
        self.temperature = max(float(temperature), 1e-3)   # >1 suaviza probabilidades sobreconfiadas
        self.model = None
        self.extractor = None
        self.device = "cpu"

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
        from app.ml.emotion.architectures import Wav2Vec2ForSpeechClassification, uses_speech_classification_head
        s = get_settings()
        t0 = time.time()
        kw = {"cache_dir": str(s.model_cache_dir), "local_files_only": s.hf_offline}
        self.extractor = AutoFeatureExtractor.from_pretrained(self.source, **kw)
        config = AutoConfig.from_pretrained(self.source, **kw)
        cls = Wav2Vec2ForSpeechClassification if uses_speech_classification_head(config) else AutoModelForAudioClassification
        self.model, info = cls.from_pretrained(self.source, output_loading_info=True, **kw)
        # Guarda anti-"cabeza aleatoria": si falta cualquier clave del clasificador, las probabilidades serían basura.
        bad = [k for k in info.get("missing_keys", []) if k.startswith(("classifier", "projector", "score"))]
        if bad:
            raise RuntimeError(f"El checkpoint '{self.source}' no coincide con la arquitectura ({cls.__name__}); "
                               f"faltan claves de la cabeza de clasificación: {bad}")
        self.arch = cls.__name__
        self.sample_rate = int(getattr(self.extractor, "sampling_rate", 16000))
        dev = detect_device(self._device_pref or s.device).device
        self.device = dev
        if dev == "cuda" and self._fp16:
            self.model.half()
        self.model.to(dev).eval()
        id2label = self.model.config.id2label
        self.labels = [normalize_label(id2label[i]) for i in range(len(id2label))]
        n_params = sum(p.numel() for p in self.model.parameters())
        self.loaded = True
        log.info("Modelo de emociones cargado", extra={"source": self.source, "device": dev,
                                                       "labels": self.labels, "params": n_params,
                                                       "secs": round(time.time() - t0, 1)})

    def info(self) -> dict:
        d = super().info()
        d.update({"source": self.source, "device": self.device, "max_seconds": self.max_seconds,
                  "architecture": getattr(self, "arch", None)})
        return d

    def _prep(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        max_n = int(self.max_seconds * self.sample_rate)
        if len(w) > max_n:
            w = w[:max_n]
        if len(w) < MIN_SAMPLES:
            w = np.pad(w, (0, MIN_SAMPLES - len(w)))
        return w

    def predict(self, waveforms: list[np.ndarray], batch_size: int = 16) -> list[EmotionOutput]:
        import torch
        self.ensure_loaded()
        prepped = [self._prep(w) for w in waveforms]
        # Se agrupan por longitud exacta: sin padding => no depende de attention_mask del extractor.
        groups: dict[int, list[int]] = defaultdict(list)
        for i, w in enumerate(prepped):
            groups[len(w)].append(i)
        out: list[EmotionOutput | None] = [None] * len(prepped)
        for _, idxs in groups.items():
            for b in range(0, len(idxs), batch_size):
                chunk = idxs[b:b + batch_size]
                inputs = self.extractor([prepped[i] for i in chunk], sampling_rate=self.sample_rate,
                                        return_tensors="pt", padding=True)
                x = inputs["input_values"].to(self.device)
                if self.device == "cuda" and self._fp16:
                    x = x.half()
                with torch.inference_mode():
                    logits = self.model(x).logits.float()
                probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()
                for i, p in zip(chunk, probs):
                    d = {lab: float(p[k]) for k, lab in enumerate(self.labels)}
                    top = max(d, key=d.get)
                    out[i] = EmotionOutput(top, d[top], d)
        return out  # type: ignore[return-value]
