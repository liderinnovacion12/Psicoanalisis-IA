"""Audio: FFmpeg, formatos, calidad, detección de canales y procesamiento de llamada larga en streaming."""
import subprocess
import threading
import time

import numpy as np
import psutil
import pytest
import soundfile as sf

from app.core.config import get_config_store
from app.core.errors import InvalidAudio
from app.ml.audio import ffmpeg
from app.ml.audio.processing import process_audio
from app.ml.audio.quality import analyze_levels, detect_channel_layout
from tests.conftest import EXAMPLES

CFG = get_config_store().defaults("audio")


def synth_speechlike(seconds, sr=16000, seed=0, f0=140):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = np.clip(np.sin(2 * np.pi * 0.35 * t), 0, None) ** 0.5
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * env
    return (0.3 * x / (np.abs(x).max() + 1e-9) + 0.002 * rng.standard_normal(len(t))).astype("float32")


@pytest.mark.parametrize("ext,codec", [("wav", None), ("flac", None), ("mp3", "libmp3lame"), ("ogg", "libvorbis"),
                                       ("m4a", "aac"), ("aac", "aac")])
def test_all_required_formats_are_accepted_and_probed(tmp_path, tone_wav, ext, codec):
    dst = tmp_path / f"a.{ext}"
    cmd = [ffmpeg.ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(tone_wav)]
    if codec:
        cmd += ["-c:a", codec]
    subprocess.run(cmd + [str(dst)], check=True)
    meta = ffmpeg.probe(dst)
    assert abs(meta["duration"] - 3.0) < 0.3 and meta["sample_rate"] > 0 and meta["channels"] == 1
    out = tmp_path / "n.wav"
    ffmpeg.to_wav(dst, out, sample_rate=16000, channels=1)
    info = sf.info(out)
    assert info.samplerate == 16000 and info.channels == 1


def test_invalid_audio_is_rejected(tmp_path):
    p = tmp_path / "x.mp3"
    p.write_bytes(b"no soy audio" * 50)
    with pytest.raises(InvalidAudio):
        ffmpeg.probe(p)


def test_quality_metrics_detect_clipping_and_silence(tmp_path):
    x = np.clip(synth_speechlike(20) * 6, -1, 1)
    p = tmp_path / "clip.wav"
    sf.write(p, x, 16000)
    lv = analyze_levels(p)
    assert lv["clipping_ratio"] > 0.01 and lv["peak_dbfs"] > -0.5
    y = np.zeros(16000 * 10, "float32")
    sf.write(tmp_path / "sil.wav", y, 16000)
    assert analyze_levels(tmp_path / "sil.wav")["silence_ratio"] > 0.9


def test_channel_detection_stereo_split_vs_mono_mix(tmp_path):
    sr = 16000
    a = synth_speechlike(30, seed=1, f0=120)
    b = np.roll(synth_speechlike(30, seed=2, f0=200), sr * 15)         # habla en momentos distintos
    a[15 * sr:] = 0
    b[: 15 * sr] = 0
    sf.write(tmp_path / "split.wav", np.stack([a, b], 1), sr)
    sf.write(tmp_path / "mix.wav", np.stack([a + b, a + b], 1), sr)
    ch = CFG["channels"]
    assert detect_channel_layout(tmp_path / "split.wav", ch)["is_stereo_split"] is True
    m = detect_channel_layout(tmp_path / "mix.wav", ch)
    assert m["is_stereo_split"] is False and m["correlation"] > 0.9


def test_process_audio_pipeline_stage_outputs(tmp_path):
    sr = 16000
    sf.write(tmp_path / "in.wav", np.stack([synth_speechlike(20, seed=3), synth_speechlike(20, seed=4)], 1), sr)
    res = process_audio(tmp_path / "in.wav", tmp_path / "w", CFG)
    assert res["mode"] in ("mono", "stereo_split") and res["quality"]["score"] > 0
    for f in res["files"].values():
        i = sf.info(tmp_path / "w" / f)
        assert i.samplerate == 16000 and i.channels == 1
    assert not (tmp_path / "w" / "raw.wav").exists()          # temporales eliminados


@pytest.mark.slow
@pytest.mark.skipif(not (EXAMPLES / "llamada_ejemplo_estereo.wav").exists(), reason="falta la llamada de ejemplo")
def test_memory_does_not_grow_with_call_duration(tmp_path):
    """La RAM de trabajo depende del tamaño de bloque (5 min), NO de la duración: 60 min ≈ 17 min, y ambos << audio completo."""
    src = EXAMPLES / "llamada_ejemplo_estereo.wav"

    def build(copies, name):
        lst = tmp_path / f"{name}.txt"
        lst.write_text("".join(f"file '{src.resolve().as_posix()}'\n" for _ in range(copies)))
        out = tmp_path / f"{name}.wav"
        subprocess.run([ffmpeg.ffmpeg_path(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
        return out

    def measure(path, work):
        proc, peak, stop = psutil.Process(), [0], threading.Event()
        base = proc.memory_info().rss

        def watch():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.2)
        threading.Thread(target=watch, daemon=True).start()
        t0 = time.time()
        res = process_audio(path, tmp_path / work, CFG)
        stop.set()
        return res, peak[0] - base, time.time() - t0

    process_audio(src, tmp_path / "warmup", CFG)                  # calienta torch/Silero (su carga no depende de la duración)
    mid, hour = build(12, "mid"), build(43, "hour")
    r_mid, g_mid, _ = measure(mid, "w_mid")
    r_hour, g_hour, secs = measure(hour, "w_hour")
    full = r_hour["duration"] * 16000 * 2 * 4
    print(f"17 min: RAM +{g_mid / 1e6:.0f} MB | 60 min: RAM +{g_hour / 1e6:.0f} MB en {secs:.0f}s | audio completo en float32: {full / 1e6:.0f} MB")
    assert r_hour["mode"] == "stereo_split" and r_hour["duration"] > 3500
    assert g_hour < g_mid * 1.5 + 60e6          # no crece con la duración (4.3x más audio)
