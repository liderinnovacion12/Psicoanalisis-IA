"""Diarización de respaldo con EMBEDDINGS DE HABLANTE (WavLM-SV, microsoft/wavlm-base-plus-sv; sin token de Hugging Face).

Decide primero si hay 1 o 2 personas y, si hay 2, las separa. A diferencia de las estadísticas MFCC, un modelo
de verificación de hablante distingue "la misma voz con distinta entonación/emoción" de "voces distintas".

Decisión (automática): se agrupan las ventanas en 2 clusters y se exigen a la vez
    similitud coseno entre centroides  <= `max_centroid_similarity`   (misma voz => ~0.85-0.99; voces distintas => ~0.35)
    brecha (similitud intra - inter)   >= `min_gap`
    cluster minoritario                >= `min_minor_share`           (evita que un ruido/tos cree una "segunda persona")
Calibrado con 4 audios reales (ver README): 1 voz -> sim 0.85-0.99, brecha 0.01-0.09; 2 voces -> sim 0.35, brecha 0.63.
CALIBRACIÓN LIMITADA: solo hay un ejemplo de dos voces (sintético). El usuario puede FORZAR 1 o 2 personas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.device import torch_device
from app.core.logging import get_logger
from app.ml.audio.io import WavReader
from app.ml.diarization.base import (SPEAKERS, DiarizationResult, Diarizer, Turn, mark_overlaps, merge_turns)

log = get_logger(__name__)
_MODEL: tuple | None = None
SR = 16000

DEFAULTS = {"model": "microsoft/wavlm-base-plus-sv", "window": 3.0, "hop": 1.5, "min_window": 1.5, "batch_size": 16,
            "max_centroid_similarity": 0.72, "min_gap": 0.25, "min_minor_share": 0.08, "min_windows": 6}


def embedding_model_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _load(name: str):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        s = get_settings()
        kw = {"cache_dir": str(s.model_cache_dir), "local_files_only": s.hf_offline}
        dev = torch_device(s.device)
        fe = AutoFeatureExtractor.from_pretrained(name, **kw)
        model = WavLMForXVector.from_pretrained(name, **kw).to(dev).eval()
        _MODEL = (fe, model, dev)
        log.info("Modelo de embeddings de hablante cargado", extra={"model": name, "device": dev})
    return _MODEL


def plan_windows(vad: list[tuple[float, float]], win: float, hop: float, min_len: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in vad:
        if e - s < min_len:
            continue
        t = s
        while t + min_len <= e:
            out.append((t, min(t + win, e)))
            t += hop
    return out


def embed_windows(path: Path, windows: list[tuple[float, float]], cfg: dict, progress=None) -> np.ndarray:
    import torch
    fe, model, dev = _load(cfg["model"])
    by_len: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(windows):
        by_len.setdefault(int(round((b - a) * SR)), []).append(i)
    out = np.zeros((len(windows), 512), dtype=np.float32)
    done, total = 0, len(windows)
    with WavReader(path) as rd:
        for _, idxs in by_len.items():
            for k in range(0, len(idxs), cfg["batch_size"]):
                chunk = idxs[k:k + cfg["batch_size"]]
                waves = [rd.read(*windows[i]) for i in chunk]
                inp = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True)
                with torch.inference_mode():
                    v = model(**{a: b.to(dev) for a, b in inp.items()}).embeddings
                v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
                for j, i in enumerate(chunk):
                    out[i] = v[j]
                done += len(chunk)
                if progress:
                    progress(100 * done / max(total, 1), "Analizando voces")
    return out


def cluster_two(E: np.ndarray) -> tuple[np.ndarray, dict]:
    """KMeans(2) sobre embeddings normalizados + estadísticas de separación."""
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit(E).labels_
    if min(np.bincount(lab, minlength=2)) < 2:
        return lab, {"centroid_similarity": 1.0, "gap": 0.0, "minor_share": float(min(lab.mean(), 1 - lab.mean()))}
    c0, c1 = E[lab == 0].mean(0), E[lab == 1].mean(0)
    cs = float(c0 @ c1 / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-9))
    S = E @ E.T
    def intra(m):
        s = S[np.ix_(m, m)]
        return float(s[np.triu_indices(len(s), 1)].mean())
    inter = float(S[np.ix_(lab == 0, lab == 1)].mean())
    gap = (intra(lab == 0) + intra(lab == 1)) / 2 - inter
    return lab, {"centroid_similarity": cs, "gap": float(gap), "minor_share": float(min(lab.mean(), 1 - lab.mean()))}


def looks_like_two_speakers(stats: dict, cfg: dict) -> bool:
    return (stats["centroid_similarity"] <= cfg["max_centroid_similarity"] and stats["gap"] >= cfg["min_gap"]
            and stats["minor_share"] >= cfg["min_minor_share"])


class EmbeddingDiarizer(Diarizer):
    name = "embedding"

    def diarize(self, work_dir: Path, files: dict[str, str], vad: dict, cfg: dict, progress=None, hint: int | None = None):
        ecfg = {**DEFAULTS, **(cfg.get("diarization", {}).get("embedding") or {})}
        seg_cfg = cfg.get("segmentation", {})
        gap_m, min_turn = seg_cfg.get("merge_gap", 0.8), seg_cfg.get("min_turn", 0.6)
        segs = vad.get("mono", [])
        info_warn = ["Diarización de respaldo con embeddings de voz (sin pyannote): precisión moderada. Configure HF_TOKEN "
                     "para utilizar pyannote.audio, o use audio estéreo con una persona por canal."]

        def single(reason: str, stats: dict | None = None) -> DiarizationResult:
            turns = merge_turns([Turn(SPEAKERS[0], s, e) for s, e in segs], gap_m, min_turn)
            return DiarizationResult(turns, self.name, 1, 100.0, [], {"single_speaker": True, "reason": reason,
                                                                      **{k: round(v, 4) for k, v in (stats or {}).items()}})
        if hint == 1:
            return single("indicado por el usuario")
        windows = plan_windows(segs, ecfg["window"], ecfg["hop"], ecfg["min_window"])
        if len(windows) < ecfg["min_windows"]:
            r = single("muy poca voz para separar hablantes")
            r.warnings = ["Muy poca voz para separar hablantes; se analiza como una sola persona."]
            return r
        E = embed_windows(work_dir / files["mono"], windows, ecfg, progress)
        lab, stats = cluster_two(E)
        if hint != 2 and not looks_like_two_speakers(stats, ecfg):
            return single("no se detectó una segunda voz distinta", stats)

        # suavizado temporal (mediana móvil de 3 ventanas, en orden temporal) y ventanas -> turnos
        order = np.argsort([w[0] for w in windows])
        lab_o = lab[order]
        pad = np.pad(lab_o, 1, mode="edge")
        sm = np.array([int(np.median(pad[i:i + 3]) > 0.5) for i in range(len(lab_o))])
        half = ecfg["hop"] / 2
        raw = [Turn(f"S{l}", (windows[i][0] + windows[i][1]) / 2 - half, (windows[i][0] + windows[i][1]) / 2 + half)
               for i, l in zip(order, sm)]
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
        first: dict[str, int] = {}
        for t in sorted(clipped, key=lambda t: t.start):
            first.setdefault(t.speaker, len(first))
        for t in clipped:
            t.speaker = SPEAKERS[min(first[t.speaker], 1)]
        turns = merge_turns(clipped, gap_m, min_turn)
        mark_overlaps(turns)
        n_found = len({t.speaker for t in turns})
        quality = round(float(np.clip(stats["gap"] / 0.6, 0, 1)) * 70, 1)      # techo 70: aún no validado con llamadas reales
        return DiarizationResult(turns, self.name, n_found, quality, info_warn,
                                 {**{k: round(v, 4) for k, v in stats.items()}, "windows": len(windows), "forced": hint == 2})
