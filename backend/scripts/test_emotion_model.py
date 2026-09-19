"""Demuestra que el modelo base de emociones funciona: carga, inferencia, sample rate,
duración máxima, memoria, GPU/CPU.

Uso:
    python scripts/test_emotion_model.py                      # audio sintético (solo valida la mecánica)
    python scripts/test_emotion_model.py --audio llamada.wav  # audio real (recomendado)
    python scripts/test_emotion_model.py --model models/custom_emotion_model

IMPORTANTE: con audio sintético las probabilidades NO tienen significado; solo prueban
que el modelo carga y produce distribuciones válidas. Para evaluar rendimiento real hay que
usar grabaciones etiquetadas (ver README > Entrenamiento).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil

from app.core.config import get_settings
from app.core.device import detect_device
from app.ml.emotion.wav2vec import BASELINE_HF_ID, FineTunedEmotionModel, Wav2VecEmotionModel


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def synthetic(seconds: float, sr: int = 16000) -> np.ndarray:
    """Señal 'tipo voz': armónicos modulados en amplitud + ruido leve."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    x *= 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) / 2 + 0.2
    x += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="WAV/MP3/etc. real (se convierte a 16 kHz mono con FFmpeg)")
    ap.add_argument("--model", help="directorio de un modelo fine-tuned; por defecto el baseline de HF")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="duración máxima a probar")
    args = ap.parse_args()

    s = get_settings()
    dev = detect_device(s.device)
    print(f"DEVICE : {dev.label}  (torch {dev.torch_version}, cuda {dev.cuda_version})")
    print(f"MODEL  : {args.model or s.model_name}")
    ok = True

    rss0 = rss_mb()
    t0 = time.time()
    model = FineTunedEmotionModel(args.model, "custom") if args.model else Wav2VecEmotionModel(s.model_name or BASELINE_HF_ID)
    model.load()
    print(f"[1] Carga            : OK  ({time.time() - t0:.1f}s, +{rss_mb() - rss0:.0f} MB RAM)")
    print(f"    labels           : {model.labels}")
    print(f"    sample_rate      : {model.sample_rate} Hz")
    if model.sample_rate != 16000:
        print("    ADVERTENCIA: el modelo no usa 16 kHz; la app normaliza a 16 kHz (configurable).")

    if args.audio:
        from app.ml.audio import ffmpeg
        import soundfile as sf
        tmp = Path(s.temp_dir) / "test_emotion.wav"
        ffmpeg.to_wav(args.audio, tmp, sample_rate=model.sample_rate, channels=1)
        wav, _ = sf.read(tmp, dtype="float32")
        print(f"[2] Audio real       : {len(wav) / model.sample_rate:.1f}s")
    else:
        wav = synthetic(args.max_seconds, model.sample_rate)
        print(f"[2] Audio sintético  : {len(wav) / model.sample_rate:.1f}s (probabilidades sin significado)")

    # Inferencia por duración
    print("[3] Inferencia por duración de ventana:")
    for sec in (1, 2.5, 5, 10, 30):
        if sec * model.sample_rate > len(wav):
            continue
        w = wav[: int(sec * model.sample_rate)]
        model.predict([w])                                  # calentamiento
        t = time.time()
        out = model.predict([w])[0]
        dt = time.time() - t
        tot = sum(out.probabilities.values())
        good = abs(tot - 1) < 1e-3
        ok &= good
        print(f"    {sec:>4}s -> {dt * 1000:7.0f} ms  top={out.emotion:<9} p={out.confidence:.3f}  suma={tot:.4f} {'OK' if good else 'FALLA'}")

    # Lote
    n = 16
    wins = [wav[i * 8000: i * 8000 + 5 * model.sample_rate] for i in range(n)]
    wins = [w for w in wins if len(w) == 5 * model.sample_rate] or [wav[: 5 * model.sample_rate]]
    t = time.time()
    outs = model.predict(wins)
    dt = time.time() - t
    print(f"[4] Lote {len(wins)}x5s        : {dt:.2f}s  ({dt / len(wins) * 1000:.0f} ms/ventana, "
          f"RTF ventana={dt / (len(wins) * 5):.3f})")
    print(f"    distribución completa (1ª ventana): { {k: round(v, 3) for k, v in outs[0].probabilities.items()} }")

    # Duración máxima
    long_sec = min(len(wav) / model.sample_rate, args.max_seconds)
    rss1 = rss_mb()
    t = time.time()
    model.predict([wav[: int(long_sec * model.sample_rate)]])
    print(f"[5] Entrada de {long_sec:.0f}s  : {time.time() - t:.1f}s, RAM +{rss_mb() - rss1:.0f} MB "
          f"(max_seconds configurado={model.max_seconds}; la app usa ventanas de 5 s)")

    if dev.device == "cuda":
        import torch
        print(f"[6] VRAM             : {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB pico")
    print(f"    RAM total proceso: {rss_mb():.0f} MB")
    print("RESULTADO:", "OK" if ok else "FALLÓ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
