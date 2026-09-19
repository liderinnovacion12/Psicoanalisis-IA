"""Envoltorio de FFmpeg/ffprobe. Usa el binario del PATH o el de `imageio-ffmpeg` como respaldo."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from app.core.errors import InvalidAudio
from app.core.logging import get_logger

log = get_logger(__name__)

_LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "2.1": 3, "3.0": 3, "quad": 4, "4.0": 4, "5.0": 5,
                    "5.1": 6, "6.1": 7, "7.1": 8}


@lru_cache
def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FFmpeg no está instalado (instálelo o `pip install imageio-ffmpeg`).") from e


@lru_cache
def ffprobe_path() -> str | None:
    p = shutil.which("ffprobe")
    if p:
        return p
    ff = Path(ffmpeg_path())
    cand = ff.with_name("ffprobe" + ff.suffix)
    return str(cand) if cand.exists() else None


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def probe(path: str | Path) -> dict:
    """Devuelve duration, sample_rate, channels, bitrate, codec. Lanza InvalidAudio si no es audio."""
    path = str(path)
    info: dict = {}
    fp = ffprobe_path()
    if fp:
        r = _run([fp, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
        if r.returncode == 0:
            data = json.loads(r.stdout or "{}")
            st = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
            if st:
                fmt = data.get("format", {})
                dur = float(st.get("duration") or fmt.get("duration") or 0)
                info = {
                    "duration": dur,
                    "sample_rate": int(st.get("sample_rate") or 0),
                    "channels": int(st.get("channels") or 0),
                    "bitrate": int(st.get("bit_rate") or fmt.get("bit_rate") or 0),
                    "codec": st.get("codec_name"),
                    "container": fmt.get("format_name"),
                }
    if not info:  # respaldo: parsear la salida de `ffmpeg -i`
        r = _run([ffmpeg_path(), "-hide_banner", "-i", path])
        text = r.stderr
        m_stream = re.search(r"Stream #\d+:\d+.*?: Audio: (\w+).*?, (\d+) Hz, ([^,]+)", text)
        if not m_stream:
            raise InvalidAudio(detail=f"sin stream de audio: {text[-300:]}")
        m_dur = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
        m_br = re.search(r"bitrate: (\d+) kb/s", text)
        layout = m_stream.group(3).strip()
        mch = re.match(r"(\d+) channels", layout)
        ch = int(mch.group(1)) if mch else _LAYOUT_CHANNELS.get(layout.split("(")[0], 2)
        dur = (int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))) if m_dur else 0
        info = {"duration": dur, "sample_rate": int(m_stream.group(2)), "channels": ch,
                "bitrate": int(m_br.group(1)) * 1000 if m_br else 0, "codec": m_stream.group(1),
                "container": None}
    if not info.get("duration") or info["duration"] <= 0:
        # Contenedores sin duración en cabecera (p. ej. AAC ADTS): decodificar para medir
        info["duration"] = _measure_duration(path)
    if info["duration"] <= 0:
        raise InvalidAudio(detail="duración no determinable")
    return info


def _measure_duration(path: str) -> float:
    r = _run([ffmpeg_path(), "-hide_banner", "-nostats", "-i", path, "-vn", "-f", "null", "-"])
    ms = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not ms:
        return 0.0
    h, m, s = ms[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def to_wav(src: str | Path, dst: str | Path, *, sample_rate: int = 16000, channels: int | None = None,
           af: str | None = None, timeout: int | None = None) -> None:
    """Convierte a WAV PCM s16le. `af` = cadena de filtros de audio."""
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn",
           "-ar", str(sample_rate)]
    if channels:
        cmd += ["-ac", str(channels)]
    if af:
        cmd += ["-af", af]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    r = _run(cmd, timeout)
    if r.returncode != 0:
        log.error("ffmpeg falló", extra={"stderr": r.stderr[-800:], "src": str(src)})
        raise InvalidAudio(detail=r.stderr[-500:])


def extract_channel(src: str | Path, dst: str | Path, channel: int, *, sample_rate: int = 16000,
                    af: str | None = None) -> None:
    pan = f"pan=mono|c0=c{channel}"
    filt = pan + ("," + af if af else "")
    to_wav(src, dst, sample_rate=sample_rate, channels=1, af=filt)


def cut_segment(src: str | Path, dst: str | Path, start: float, end: float, *, sample_rate: int = 16000,
                channels: int = 1, channel: int | None = None) -> None:
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
           "-t", f"{max(end - start, 0.05):.3f}", "-i", str(src), "-vn"]
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    cmd += ["-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)]
    r = _run(cmd)
    if r.returncode != 0:
        raise InvalidAudio(detail=r.stderr[-500:])


def make_playback(src: str | Path, dst: str | Path, *, bitrate: str = "64k", sample_rate: int = 24000,
                  channels: int = 1) -> None:
    """MP3 de reproducción liviano (mezcla a mono). Se sirve con Range al reproductor web."""
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn",
           "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    r = _run(cmd)
    if r.returncode != 0:
        raise InvalidAudio(detail=r.stderr[-500:])
