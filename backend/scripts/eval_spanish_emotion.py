"""Evalúa modelos de emoción de AUDIO sobre voz en ESPAÑOL con etiquetas reales (corpus MESD, CC-BY-4.0).

    python scripts/eval_spanish_emotion.py [--max N] [--models baseline,mesd]

Mide: exactitud, Macro-F1, matriz de confusión, sobreconfianza (ECE) y la temperatura que minimiza la NLL en VALIDATION
(evaluando en TEST, que ningún modelo usó para ajustar nada).

LÍMITES (importantes): MESD son ~1 200 clips de ~1 s, actuados, en estudio, de pocos hablantes y 6 emociones (sin 'surprise').
Es la única evidencia con etiquetas en español que se puede medir aquí, pero NO representa llamadas reales (ruido, canal
telefónico, emoción espontánea). Un buen resultado aquí no garantiza uno bueno en llamadas.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf


def load_split(split: str, cache: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("hackathon-pln-es/MESD", f"data/{split}-00000-of-00001.parquet", repo_type="dataset", cache_dir=cache)
    return pd.read_parquet(p)


def decode(row) -> np.ndarray:
    """MESD (versión Somos NLP): `audio_array` ya viene a 16 kHz (palabras sueltas de ~0.8 s)."""
    return np.asarray(row["audio_array"], dtype="float32")


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def best_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best, bt = 1e9, 1.0
    for T in np.arange(0.5, 15.01, 0.25):
        p = softmax(logits, T)[np.arange(len(y)), y]
        nll = -np.log(np.clip(p, 1e-9, 1)).mean()
        if nll < best:
            best, bt = nll, float(T)
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="máximo de clips por split (0 = todos)")
    ap.add_argument("--models", default="baseline,mesd")
    ap.add_argument("--out", default=str(ROOT.parent / "docs" / "eval_spanish_emotion.json"))
    a = ap.parse_args()
    import torch
    from sklearn.metrics import confusion_matrix, f1_score
    from app.core.config import get_settings
    from app.ml.emotion.hf import normalize_label
    from app.ml.emotion.wav2vec import Wav2VecEmotionModel
    s = get_settings()
    cache = str(s.model_cache_dir)

    data = {sp: load_split(sp, cache) for sp in ("validation", "test")}
    lab_col = "emotion"
    print("columnas:", list(data["test"].columns), "| clips val/test:", len(data["validation"]), len(data["test"]))
    ES2EN = {"anger": "angry", "disgust": "disgust", "fear": "fear", "happiness": "happy", "neutral": "neutral", "sadness": "sad"}

    def y_of(df):
        return [ES2EN.get(str(v).lower(), str(v).lower()) for v in df[lab_col].tolist()]

    results = {}
    for mname in a.models.split(","):
        if mname == "baseline":
            m = Wav2VecEmotionModel(); m.load()
            def run(waves):
                out = []
                for w in waves:
                    o = m.predict([w])[0]
                    out.append(o.probabilities)
                return out
            def logits_of(waves):
                import torch
                res = []
                for w in waves:
                    w = m._prep(w)
                    inp = m.extractor([w], sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.inference_mode():
                        res.append(m.model(inp["input_values"].to(m.device)).logits.float().cpu().numpy()[0])
                return np.array(res), m.labels
        else:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            rid = "somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-classification-MESD"
            fe = AutoFeatureExtractor.from_pretrained(rid, cache_dir=cache)
            mod = AutoModelForAudioClassification.from_pretrained(rid, cache_dir=cache).eval()
            lbls = [normalize_label(mod.config.id2label[i]) for i in range(len(mod.config.id2label))]
            lbls = [ES2EN.get(l, l) for l in lbls]
            def logits_of(waves, mod=mod, fe=fe, lbls=lbls):
                res = []
                for w in waves:
                    w = np.pad(w, (0, max(0, 8000 - len(w))))
                    inp = fe([w], sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.inference_mode():
                        res.append(mod(**inp).logits.float().numpy()[0])
                return np.array(res), lbls
        per = {}
        for sp, df in data.items():
            df2 = df.iloc[: a.max] if a.max else df
            waves = [decode(r) for _, r in df2.iterrows()]
            L, lbls = logits_of(waves)
            per[sp] = (L, y_of(df2), lbls)
            print(f"[{mname}] {sp}: {len(waves)} clips procesados")
        Lv, yv, lbls = per["validation"]
        Lt, yt, _ = per["test"]
        allowed = [l for l in lbls if l in set(ES2EN.values())]                    # 6 emociones de MESD (sin 'surprise')
        idx = [lbls.index(l) for l in allowed]
        yv_i = np.array([allowed.index(y) for y in yv]); yt_i = np.array([allowed.index(y) for y in yt])
        # (a) evaluación tal cual (el modelo puede predecir cualquiera de sus etiquetas)
        pred_full = [lbls[i] for i in Lt.argmax(1)]
        acc_full = float(np.mean([p == y for p, y in zip(pred_full, yt)]))
        f1_full = float(f1_score(yt, pred_full, average="macro", labels=allowed, zero_division=0))
        # (b) restringido a las 6 clases de MESD (para calibrar y comparar la confianza)
        Lt6, Lv6 = Lt[:, idx], Lv[:, idx]
        p1 = softmax(Lt6, 1.0)
        conf1, pred1 = p1.max(1), p1.argmax(1)
        T = best_temperature(Lv6, yv_i)
        pT = softmax(Lt6, T)
        confT = pT.max(1)
        cm = confusion_matrix(yt_i, pred1, labels=list(range(len(allowed))))
        results[mname] = {
            "n_test": len(yt), "labels": allowed, "accuracy_full": round(acc_full, 4), "macro_f1_full": round(f1_full, 4),
            "accuracy_6clases": round(float((pred1 == yt_i).mean()), 4),
            "macro_f1_6clases": round(float(f1_score(yt_i, pred1, average="macro", zero_division=0)), 4),
            "mean_confidence": round(float(conf1.mean()), 4), "ece": round(ece(conf1, (pred1 == yt_i).astype(float)), 4),
            "best_temperature_val": T, "ece_after_temperature": round(ece(confT, (pT.argmax(1) == yt_i).astype(float)), 4),
            "mean_confidence_after_T": round(float(confT.mean()), 4), "confusion_matrix": cm.tolist(),
        }
        print(json.dumps({mname: {k: v for k, v in results[mname].items() if k != "confusion_matrix"}}, ensure_ascii=False, indent=1))
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("guardado en", a.out)


if __name__ == "__main__":
    main()
