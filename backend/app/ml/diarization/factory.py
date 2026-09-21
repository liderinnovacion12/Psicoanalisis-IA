from __future__ import annotations

from app.core.logging import get_logger
from app.ml.diarization.base import Diarizer
from app.ml.diarization.channels import ChannelDiarizer
from app.ml.diarization.spectral import SpectralDiarizer

log = get_logger(__name__)


def get_diarizer(mode: str, cfg: dict) -> Diarizer:
    """mode: 'stereo_split' | 'mono'. engine (config): auto | pyannote | channels | spectral."""
    engine = cfg["diarization"]["engine"]
    if mode == "stereo_split" and engine in ("auto", "channels"):
        return ChannelDiarizer()
    if engine in ("auto", "pyannote"):
        from app.ml.diarization.pyannote import PyannoteDiarizer, pyannote_available
        if pyannote_available():
            return PyannoteDiarizer()
        if engine == "pyannote":
            log.warning("engine=pyannote pero no está disponible (paquete o HF_TOKEN); se usa el respaldo")
    if engine in ("auto", "pyannote", "embedding"):
        from app.ml.diarization.embedding import EmbeddingDiarizer, embedding_model_available
        if embedding_model_available():
            return EmbeddingDiarizer()
    return SpectralDiarizer()
