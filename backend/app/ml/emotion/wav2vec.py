"""Modelos concretos: baseline wav2vec (Hugging Face) y fine-tuned (directorio local del registry)."""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.ml.emotion.hf import HFAudioEmotionModel

BASELINE_HF_ID = "r-f/wav2vec-english-speech-emotion-recognition"


class Wav2VecEmotionModel(HFAudioEmotionModel):
    """BASELINE: r-f/wav2vec-english-speech-emotion-recognition (inglés).

    La exactitud publicada por el modelo NO representa la precisión real en llamadas ni en español.
    """

    def __init__(self, hf_id: str = BASELINE_HF_ID, name: str = "emotion_baseline_wav2vec",
                 version: str = "baseline", language: str | None = "en", **kw):
        super().__init__(name, hf_id, version, language, **kw)


class FineTunedEmotionModel(HFAudioEmotionModel):
    """Checkpoint HF local (p. ej. data/models/emotion_model_v2 o models/custom_emotion_model)."""

    def __init__(self, path: str | Path, name: str, version: str = "v1", language: str | None = None, **kw):
        p = Path(path)
        if not p.is_absolute():
            p = (get_settings().models_dir / p) if (get_settings().models_dir / p).exists() else p.resolve()
        super().__init__(name, str(p), version, language, **kw)
