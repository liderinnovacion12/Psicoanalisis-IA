"""Diarización de respaldo sin modelos externos (mono): clustering de 2 hablantes sobre MFCC.

ES REAL PERO DE PRECISIÓN LIMITADA: sirve cuando no hay token de Hugging Face / pyannote. Para
resultados de calidad use `pyannote` o audio estéreo con un hablante por canal. La calidad
reportada se limita (<= 60) y la UI muestra una advertencia.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.ml.audio.io import iter_blocks
from app.ml.diarization.base import (DiarizationResult, Diarizer, Turn, SPEAKERS, mark_overlaps,
                                     merge_turns)

log = get_logger(__name__)
WIN, HOP = 1.5, 0.75


def _features(path: Path, vad: list[tuple[float, float]], sr: int = 16000, block_s: float = 600):
    import librosa
    feats, times = [], []
    vad_arr = np.array(vad) if vad else np.zeros((0, 2))
    for off, blk in iter_blocks(path, block_s):
        y = blk.mean(axis=1)
        if len(y) < sr:
            continue
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, n_fft=512, hop_length=160, n_mels=40)
        d1 = librosa.feature.delta(mf)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=160)
        fr = np.vstack([mf, d1[:8], cent / 1000.0]).T            # (frames, D) a 10 ms
        n_frames = fr.shape[0]
        w, h = int(WIN * 100), int(HOP * 100)
        for i in range(0, n_frames - w + 1, h):
            t0, t1 = off + i / 100, off + (i + w) / 100
            # exigir que la ventana sea mayormente voz
            if len(vad_arr):
                ov = np.clip(np.minimum(vad_arr[:, 1], t1) - np.maximum(vad_arr[:, 0], t0), 0, None).sum()
                if ov / WIN < 0.7:
                    continue
            seg = fr[i:i + w]
            feats.append(np.concatenate([seg.mean(0), seg.std(0)]))
            times.append((t0, t1))
    return np.array(feats), times


class SpectralDiarizer(Diarizer):
    name = "spectral"

    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict, progress=None):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        path = work_dir / files["mono"]
        segs = vad.get("mono", [])
        X, times = _features(path, segs)
        seg_cfg = cfg.get("segmentation", {})
        warnings = ["Diarización de respaldo (sin pyannote): precisión limitada. Configure HF_TOKEN para "
                    "utilizar pyannote.audio."]
        if len(X) < 4:
            t = [Turn(SPEAKERS[0], s, e) for s, e in segs]
            return DiarizationResult(merge_turns(t, seg_cfg.get("merge_gap", 0.8)), self.name, 1, 20.0,
                                     warnings + ["Muy poca voz para separar hablantes."], {})
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Z)
        labels = km.labels_.copy()
        # suavizado temporal (mediana móvil de 5 ventanas)
        k = 5
        pad = np.pad(labels, k // 2, mode="edge")
        labels = np.array([int(np.median(pad[i:i + k]) > 0.5) for i in range(len(labels))])
        sample = np.random.default_rng(0).choice(len(Z), size=min(len(Z), 2000), replace=False)
        try:
            sil = float(silhouette_score(Z[sample], labels[sample])) if len(set(labels[sample])) == 2 else 0.0
        except Exception:
            sil = 0.0
        # ventanas -> turnos (centro de cada ventana +- hop/2), recortados al VAD
        raw: list[Turn] = []
        for (t0, t1), lab in zip(times, labels):
            c = (t0 + t1) / 2
            raw.append(Turn(f"S{lab}", c - HOP / 2, c + HOP / 2))
        raw = merge_turns(raw, 0.05)
        vad_arr = segs
        clipped: list[Turn] = []
        for t in raw:
            for s, e in vad_arr:
                if e <= t.start:
                    continue
                if s >= t.end:
                    break
                b, en = max(s, t.start), min(e, t.end)
                if en - b > 0.1:
                    clipped.append(Turn(t.speaker, b, en))
        first = {}
        for t in sorted(clipped, key=lambda t: t.start):
            first.setdefault(t.speaker, len(first))
        for t in clipped:
            t.speaker = SPEAKERS[first[t.speaker]] if first[t.speaker] < 2 else SPEAKERS[1]
        turns = merge_turns(clipped, seg_cfg.get("merge_gap", 0.8), seg_cfg.get("min_turn", 0.6))
        mark_overlaps(turns)
        quality = round(float(np.clip((sil + 0.1) / 0.6, 0, 1)) * 60, 1)   # techo 60 (respaldo)
        return DiarizationResult(turns, self.name, len({t.speaker for t in turns}), quality, warnings,
                                 {"silhouette": round(sil, 4), "windows": int(len(X))})
