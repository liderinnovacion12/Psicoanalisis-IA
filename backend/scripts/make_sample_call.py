"""Genera una llamada de EJEMPLO sintética (TTS local de Windows/SAPI) con dos hablantes.

    python scripts/make_sample_call.py [--out ../examples/call]

Produce:
  llamada_ejemplo_estereo.wav   (cliente en canal izquierdo, agente en el derecho)  -> ESCENARIO A
  llamada_ejemplo_mono.wav      (mezcla mono)                                       -> ESCENARIO B (diarización)
  llamada_ejemplo.mp3           (mono, para probar la subida)
  llamada_ejemplo_verdad.json   (turnos reales, para evaluar la diarización)

AVISO: es voz SINTÉTICA de emoción plana. Sirve para validar el pipeline (subida, audio, diarización,
transcripción, PII, reporte) — NO para evaluar el reconocimiento emocional ni la satisfacción.
Para una evaluación real use grabaciones reales etiquetadas (ver README).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.audio import ffmpeg  # noqa: E402

SR = 16000
VOICE = "Microsoft Sabina Desktop"

SCRIPT = [  # (rol, texto, pausa_previa_s) ; pausa negativa = solapamiento
    ("agent", "Buenos días, gracias por comunicarse con atención al cliente. ¿En qué puedo ayudarle?", 0.3),
    ("client", "Hola, llevo tres días esperando una solución y nadie me ayuda. Estoy muy molesto con el servicio.", 0.5),
    ("agent", "Entiendo la situación, permítame ayudarle. ¿Me puede indicar su nombre y su número de pedido?", 0.6),
    ("client", "Mi nombre es Juan Pérez y mi teléfono es 555 123 4567. Ya di el número tres veces, es el cuarenta y cinco doce. "
               "Estoy cansado de llamar otra vez.", 0.5),
    ("agent", "Lamento mucho el inconveniente. Ya estoy revisando su caso y veo cuál fue el problema.", 0.7),
    ("client", "Bueno, espero que ahora sí lo puedan resolver, porque esto es inaceptable.", 0.4),
    ("agent", "Ya lo solucioné. Su pedido saldrá hoy mismo y recibirá un correo de confirmación.", 0.9),
    ("client", "Ah, perfecto. Muchas gracias, eso me ayuda bastante. Excelente servicio.", 0.6),
    ("agent", "Con gusto. ¿Puedo ayudarle en algo más?", 0.5),
    ("client", "No, eso es todo. Muchas gracias, que tenga un buen día.", 0.5),
]


def tts(text: str, out_wav: Path, rate: int = 0) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$t = Get-Content -Raw -Encoding UTF8 $env:TTS_TEXT;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{VOICE}'); $s.Rate = {rate};"
        "$s.SetOutputToWaveFile($env:TTS_OUT); $s.Speak($t); $s.Dispose()"
    )
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tp = f.name
    env = {**os.environ, "TTS_TEXT": tp, "TTS_OUT": str(out_wav)}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env, capture_output=True)
    Path(tp).unlink(missing_ok=True)


def load16(path: Path, agent: bool) -> np.ndarray:
    tmp = path.with_suffix(".16k.wav")
    af = "asetrate=13600,aresample=16000,atempo=1.1765" if agent else None     # timbre más grave para el agente
    ffmpeg.to_wav(path, tmp, sample_rate=SR, channels=1, af=af)
    x, _ = sf.read(tmp, dtype="float32")
    tmp.unlink(missing_ok=True)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "examples" / "call"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    left, right, truth = [], [], []
    t = 0.0
    total = 0
    tracks = {"client": [], "agent": []}
    with tempfile.TemporaryDirectory() as td:
        for i, (role, text, pause) in enumerate(SCRIPT):
            wav = Path(td) / f"{i}.wav"
            tts(text, wav, rate=0 if role == "agent" else 1)
            x = load16(wav, role == "agent")
            start = max(0.0, t + pause)
            tracks[role].append((start, x))
            truth.append({"speaker": "SPEAKER_01" if role == "agent" else "SPEAKER_00", "role": role,
                          "start": round(start, 2), "end": round(start + len(x) / SR, 2), "text": text})
            t = start + len(x) / SR
    n = int((t + 1.0) * SR)
    L, R = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for role, arr in (("client", L), ("agent", R)):
        for start, x in tracks[role]:
            i = int(start * SR)
            arr[i:i + len(x)] += x
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    L, R = L / peak * 0.8, R / peak * 0.8
    sf.write(out / "llamada_ejemplo_estereo.wav", np.stack([L, R], axis=1), SR, subtype="PCM_16")
    mono = np.clip(L + R, -1, 1)
    sf.write(out / "llamada_ejemplo_mono.wav", mono, SR, subtype="PCM_16")
    ffmpeg.make_playback(out / "llamada_ejemplo_mono.wav", out / "llamada_ejemplo.mp3", bitrate="96k", sample_rate=16000)
    (out / "llamada_ejemplo_verdad.json").write_text(json.dumps(
        {"note": "Voz sintética (TTS). Cliente = SPEAKER_00 (canal izq.), Agente = SPEAKER_01 (canal der.).",
         "duration": round(n / SR, 2), "turns": truth}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generado en {out}  (duración {n / SR:.1f}s)")


if __name__ == "__main__":
    main()
