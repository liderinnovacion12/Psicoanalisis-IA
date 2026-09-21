"""Diarización de respaldo sin modelos externos (mono): decide si hay 1 o 2 hablantes y, si hay 2, los separa por MFCC.

ES REAL PERO DE PRECISIÓN LIMITADA: sirve cuando no hay token de Hugging Face / pyannote. Para
resultados de calidad use `pyannote` o audio estéreo con un hablante por canal. La calidad
reportada se limita (<= 60) y la UI muestra una advertencia.

Es el ÚLTIMO recurso (solo si no se puede cargar el modelo de embeddings, ver embedding.py). Con datos reales se comprobó
que estas estadísticas NO separan bien: una sola persona con entonación variable dio silhouette 0.196 / separación 1.15,
igual que dos voces sintéticas (0.208 / 1.12). Por eso el automático exige umbrales altos (tenderá a decidir "una persona") y el
usuario puede FORZAR 1 o 2 personas.
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
MIN_SILHOUETTE = 0.25     # último recurso: las estadísticas MFCC confunden "misma voz, distinta entonación" con "dos voces"
MIN_SEPARATION = 1.30


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


def cluster_stats(Z: np.ndarray, labels: np.ndarray) -> dict:
    """silhouette y separación de centroides (en unidades de dispersión intra-cluster) para 2 clusters."""
    from sklearn.metrics import silhouette_score
    if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
        return {"silhouette": 0.0, "separation": 0.0, "minor_share": 0.0}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(len(Z), 2000), replace=False)
    try:
        sil = float(silhouette_score(Z[idx], labels[idx])) if len(set(labels[idx])) == 2 else 0.0
    except Exception:
        sil = 0.0
    c0, c1 = Z[labels == 0].mean(0), Z[labels == 1].mean(0)
    within = np.sqrt((Z[labels == 0].var(0).mean() + Z[labels == 1].var(0).mean()) / 2) + 1e-9
    sep = float(np.linalg.norm(c0 - c1) / (within * np.sqrt(Z.shape[1])))
    return {"silhouette": sil, "separation": sep, "minor_share": float(min(labels.mean(), 1 - labels.mean()))}


def looks_like_two_speakers(stats: dict) -> bool:
    return stats["silhouette"] >= MIN_SILHOUETTE and stats["separation"] >= MIN_SEPARATION


class SpectralDiarizer(Diarizer):
    name = "spectral"

    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict, progress=None, hint: int | None = None):
        from sklearn.cluster import KMeans
        path = work_dir / files["mono"]
        segs = vad.get("mono", [])
        seg_cfg = cfg.get("segmentation", {})
        gap, min_turn = seg_cfg.get("merge_gap", 0.8), seg_cfg.get("min_turn", 0.6)
        warnings = ["Diarización de respaldo (sin pyannote): precisión limitada. Configure HF_TOKEN para "
                    "utilizar pyannote.audio."]

        def single(reason: str, stats: dict | None = None) -> DiarizationResult:
            turns = merge_turns([Turn(SPEAKERS[0], s, e) for s, e in segs], gap, min_turn)
            return DiarizationResult(turns, self.name, 1, 100.0, [], {"single_speaker": True, "reason": reason, **(stats or {})})

        if hint == 1:
            return single("indicado por el usuario")
        if progress:
            progress(5, "Extrayendo características de voz")
        X, times = _features(path, segs)
        if len(X) < 4:
            r = single("muy poca voz para separar hablantes")
            r.warnings = warnings + ["Muy poca voz para separar hablantes; se analiza como una sola persona."]
            return r
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Z).labels_.copy()
        k = 5                                                       # suavizado temporal (mediana móvil de 5 ventanas)
        pad = np.pad(labels, k // 2, mode="edge")
        labels = np.array([int(np.median(pad[i:i + k]) > 0.5) for i in range(len(labels))])
        stats = cluster_stats(Z, labels)
        if progress:
            progress(60, "Separando hablantes")
        if hint != 2 and not looks_like_two_speakers(stats):
            return single("no se detectó una segunda voz distinta", {k_: round(v, 4) for k_, v in stats.items()})

        # ventanas -> turnos (centro de cada ventana +- hop/2), recortados al VAD
        raw: list[Turn] = []
        for (t0, t1), lab in zip(times, labels):
            c = (t0 + t1) / 2
            raw.append(Turn(f"S{lab}", c - HOP / 2, c + HOP / 2))
        raw = merge_turns(raw, 0.05)
        clipped: list[Turn] = []
        for t in raw:
            for s, e in segs:
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
        turns = merge_turns(clipped, gap, min_turn)
        mark_overlaps(turns)
        quality = round(float(np.clip((stats["silhouette"] + 0.1) / 0.6, 0, 1)) * 60, 1)   # techo 60 (respaldo)
        return DiarizationResult(turns, self.name, len({t.speaker for t in turns}), quality, warnings,
                                 {**{k_: round(v, 4) for k_, v in stats.items()}, "windows": int(len(X)),
                                  "forced": hint == 2})
