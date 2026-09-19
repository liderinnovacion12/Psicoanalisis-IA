"""Data augmentation de audio conservadora (solo TRAIN).

Advertencia sobre la etiqueta: cambiar el pitch o la velocidad modifica cómo se percibe la emoción (activación/arousal),
por lo que esas transformaciones están DESACTIVADAS por defecto y limitadas a rangos pequeños (±5 % de velocidad,
±1 semitono). El ruido se mantiene con SNR alto (>= 20 dB) y el cambio de volumen es neutro respecto a la emoción
en la práctica (los modelos normalizan la amplitud). El time-masking es corto (0.2 s) para no borrar la señal emocional.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

SAFE_LIMITS = {"noise_snr_db": (20.0, 60.0), "gain_db": (-9.0, 9.0), "time_mask_seconds": 0.3,
               "speed_range": (0.95, 1.05), "pitch_semitones": 1.0}


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(x ** 2)) + 1e-12
    noise = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(p / (10 ** (snr_db / 10)))
    return x + noise


def change_gain(x: np.ndarray, db: float) -> np.ndarray:
    return np.clip(x * (10 ** (db / 20)), -1.0, 1.0)


def time_mask(x: np.ndarray, sr: int, max_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(rng.uniform(0.05, max_s) * sr)
    if n <= 0 or n >= len(x) // 2:
        return x
    i = int(rng.integers(0, len(x) - n))
    y = x.copy()
    y[i:i + n] = 0.0
    return y


def speed_perturb(x: np.ndarray, factor: float) -> np.ndarray:
    """Cambia velocidad (y tono) remuestreando; factor>1 = más rápido."""
    up, down = int(round(1000 / factor)), 1000
    return resample_poly(x, up, down).astype(np.float32)


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import librosa
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones).astype(np.float32)


class Augmenter:
    def __init__(self, cfg: dict, sr: int = 16000, seed: int = 0):
        self.cfg, self.sr = cfg or {}, sr
        self.rng = np.random.default_rng(seed)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled"))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        c, r = self.cfg, self.rng
        lo, hi = c.get("noise_snr_db", [20, 35])
        lo = max(lo, SAFE_LIMITS["noise_snr_db"][0])
        if r.random() < 0.4:
            x = add_noise(x, float(r.uniform(lo, max(hi, lo))), r)
        g = c.get("gain_db", [-6, 6])
        if r.random() < 0.5:
            x = change_gain(x, float(r.uniform(max(g[0], SAFE_LIMITS["gain_db"][0]), min(g[1], SAFE_LIMITS["gain_db"][1]))))
        if c.get("time_mask_seconds", 0) > 0 and r.random() < 0.3:
            x = time_mask(x, self.sr, min(c["time_mask_seconds"], SAFE_LIMITS["time_mask_seconds"]), r)
        sp = c.get("speed_range", [1.0, 1.0])
        if sp[0] != 1.0 or sp[1] != 1.0:
            if r.random() < 0.3:
                x = speed_perturb(x, float(r.uniform(max(sp[0], SAFE_LIMITS["speed_range"][0]), min(sp[1], SAFE_LIMITS["speed_range"][1]))))
        ps = float(c.get("pitch_semitones", 0.0))
        if ps > 0 and r.random() < 0.3:
            x = pitch_shift(x, self.sr, float(r.uniform(-min(ps, 1.0), min(ps, 1.0))))
        return x.astype(np.float32)
