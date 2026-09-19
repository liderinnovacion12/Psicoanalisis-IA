"""Dataset de audio para entrenamiento, validación de muestras, estadísticas y desbalance."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.ml.training.augmentation import Augmenter


@dataclass
class Sample:
    id: str
    path: str                 # WAV local 16 kHz mono (recorte ya extraído)
    label: str
    start: float = 0.0
    end: float | None = None


def read_sample_audio(path: str, start: float = 0.0, end: float | None = None, sr: int = 16000) -> np.ndarray:
    with sf.SoundFile(path) as f:
        if f.samplerate != sr:
            raise ValueError(f"sample rate {f.samplerate} != {sr}")
        f.seek(int(start * sr))
        n = -1 if end is None else int((end - start) * sr)
        x = f.read(n, dtype="float32", always_2d=True)
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]


class AudioDataset:
    """Compatible con torch DataLoader (protocolo __len__/__getitem__)."""

    def __init__(self, samples: list[Sample], labels: list[str], max_seconds: float = 8.0, train: bool = False,
                 augment_cfg: dict | None = None, sr: int = 16000, seed: int = 0):
        self.samples, self.labels, self.sr = samples, labels, sr
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.max_n = int(max_seconds * sr)
        self.train = train
        self.aug = Augmenter(augment_cfg or {}, sr, seed) if train else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        x = read_sample_audio(s.path, s.start, s.end, self.sr)
        if self.aug is not None:
            x = self.aug(x)
        if len(x) > self.max_n:            # recorte aleatorio en train, centrado en eval (acota memoria)
            off = int(self.rng.integers(0, len(x) - self.max_n + 1)) if self.train else (len(x) - self.max_n) // 2
            x = x[off:off + self.max_n]
        return x, self.label2id[s.label]


def make_collate(extractor):
    import torch

    def collate(batch):
        waves = [b[0] for b in batch]
        inputs = extractor(waves, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        return inputs["input_values"], inputs.get("attention_mask"), torch.tensor([b[1] for b in batch], dtype=torch.long)
    return collate


# ---- estadísticas y desbalance ---------------------------------------------------------------------------
def class_distribution(labels: list[str | None]) -> dict[str, int]:
    return dict(Counter(l for l in labels if l))


def imbalance_report(dist: dict[str, int], alert_ratio: float = 0.5) -> dict:
    total = sum(dist.values())
    if not total:
        return {"imbalanced": False, "total": 0, "shares": {}, "strategies": []}
    shares = {k: v / total for k, v in dist.items()}
    top_k, top = max(shares.items(), key=lambda kv: kv[1])
    small = min(shares.values())
    imbalanced = top > alert_ratio or (len(dist) > 1 and top / max(small, 1e-9) > 8)
    msg = None
    if imbalanced:
        msg = (f"Desbalance de clases: '{top_k}' representa el {top:.0%} de las muestras "
               f"(clase minoritaria {small:.0%}). Use class weights u oversampling; no dependa de la accuracy.")
    return {"imbalanced": imbalanced, "total": total, "shares": {k: round(v, 4) for k, v in shares.items()},
            "message": msg, "strategies": ["class_weights", "oversample", "augmentation"] if imbalanced else []}


def compute_class_weights(train_labels: list[str], labels: list[str]) -> list[float]:
    """Pesos inversamente proporcionales a la frecuencia (normalizados a media 1 sobre las clases presentes)."""
    cnt = Counter(train_labels)
    n, k = len(train_labels), max(len([l for l in labels if cnt.get(l)]), 1)
    w = [n / (k * cnt[l]) if cnt.get(l) else 0.0 for l in labels]
    present = [x for x in w if x > 0]
    m = float(np.mean(present)) if present else 1.0
    return [x / m for x in w]


# ---- validación de muestras -----------------------------------------------------------------------------
def audio_fingerprint(x: np.ndarray) -> str:
    q = np.round(x[:: max(1, len(x) // 4000)] * 1000).astype(np.int16)
    return hashlib.sha1(q.tobytes()).hexdigest()


def validate_audio(path: str, start: float = 0.0, end: float | None = None, min_seconds: float = 0.5,
                   max_seconds: float = 30.0, sr: int = 16000) -> dict:
    """Devuelve {ok, issues[], duration, hash}. Detecta corrupto, demasiado corto/largo y sin voz."""
    issues: list[str] = []
    try:
        x = read_sample_audio(path, start, end, sr)
    except Exception:
        return {"ok": False, "issues": ["archivo_corrupto"], "duration": 0.0, "hash": None}
    dur = len(x) / sr
    if dur < min_seconds:
        issues.append("audio_demasiado_corto")
    if dur > max_seconds:
        issues.append("audio_demasiado_largo")
    if len(x) == 0 or not np.isfinite(x).all():
        issues.append("archivo_corrupto")
        return {"ok": False, "issues": issues, "duration": dur, "hash": None}
    rms_db = 10 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    fr = int(0.03 * sr)
    n = len(x) // fr
    if n:
        fdb = 10 * np.log10(np.mean(x[: n * fr].reshape(n, fr).astype(np.float64) ** 2, axis=1) + 1e-12)
        voiced = float(np.mean(fdb > max(-50.0, fdb.max() - 25)))
    else:
        voiced = 0.0
    if rms_db < -55 or voiced < 0.1:
        issues.append("audio_sin_voz")
    return {"ok": not issues, "issues": issues, "duration": round(dur, 3), "hash": audio_fingerprint(x),
            "rms_db": round(rms_db, 1)}
