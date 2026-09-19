"""Fine-tuning por línea de comandos (sin API ni base de datos) a partir de un CSV + carpeta de audios.

    python scripts/train.py --manifest ../examples/dataset/manifest.csv --audio-dir ../examples/dataset/audio \
        --base r-f/wav2vec-english-speech-emotion-recognition --output data/models/mi_modelo_es \
        --epochs 10 --batch-size 8 --lr 3e-5 --imbalance class_weights

CSV (UTF-8, con cabecera): audio,speaker,emotion,satisfaction,start,end,language,speaker_group
  * `audio`: ruta relativa a --audio-dir (cualquier formato que FFmpeg lea);  `emotion`: etiqueta (angry, happy, …, o propia)
  * `start`/`end` (segundos o HH:MM:SS, opcionales): recorte dentro del archivo
  * `speaker_group`: persona/llamada. Se usa para separar TRAIN/VAL/TEST SIN FUGA DE DATOS.
Resultado: modelo Hugging Face en <output>/best + training_results.json (accuracy, F1, matriz de confusión).
Registrarlo en la app:  python scripts/register_model.py --path <output>/best --name mi_modelo_es --language es
"""
import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.audio import ffmpeg  # noqa: E402
from app.ml.training.dataset import Sample, imbalance_report, validate_audio  # noqa: E402
from app.ml.training.splits import group_stratified_split  # noqa: E402
from app.ml.training.trainer import Callbacks, EmotionTrainer, TrainParams  # noqa: E402


def ts(v):
    if not v:
        return None
    if ":" in v:
        s = 0.0
        for p in v.split(":"):
            s = s * 60 + float(p)
        return s
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--base", default="r-f/wav2vec-english-speech-emotion-recognition")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--imbalance", choices=["none", "class_weights", "oversample"], default="class_weights")
    ap.add_argument("--augment", action="store_true", help="augmentation segura (ruido, volumen, time-mask)")
    ap.add_argument("--unfreeze-encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    base_dir, out = Path(a.audio_dir), Path(a.output)
    tmp = Path(tempfile.mkdtemp(prefix="train_"))
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    samples, meta, bad = [], [], Counter()
    for i, r in enumerate(rows):
        emo = (r.get("emotion") or "").strip().lower()
        if not emo:
            bad["etiqueta_faltante"] += 1
            continue
        src = base_dir / r["audio"].strip()
        if not src.exists():
            bad["archivo_no_encontrado"] += 1
            continue
        dst = tmp / f"{i}.wav"
        try:
            s, e = ts(r.get("start")), ts(r.get("end"))
            if e is not None:
                ffmpeg.cut_segment(src, dst, s or 0.0, e)
            else:
                ffmpeg.to_wav(src, dst, sample_rate=16000, channels=1)
        except Exception:
            bad["archivo_corrupto"] += 1
            continue
        v = validate_audio(str(dst))
        if v["issues"]:
            for x in v["issues"]:
                bad[x] += 1
            continue
        samples.append(Sample(str(i), str(dst), emo))
        meta.append({"id": str(i), "label": emo, "group": (r.get("speaker_group") or r.get("speaker") or r["audio"]).strip()})
    print(f"Muestras válidas: {len(samples)} de {len(rows)}. Descartadas: {dict(bad)}")
    if len(samples) < 12:
        sys.exit("Muy pocas muestras válidas para entrenar.")
    dist = Counter(s.label for s in samples)
    rep = imbalance_report(dict(dist))
    print("Distribución:", dict(dist))
    if rep["imbalanced"]:
        print("ALERTA:", rep["message"])
    assign, info = group_stratified_split(meta, {"train": 1 - a.val - a.test, "validation": a.val, "test": a.test}, a.seed)
    print("Splits:", info["counts"], "| sin fuga:", info["leakage_free"], info["warnings"])
    by = {sp: [s for s in samples if assign[s.id] == sp] for sp in ("train", "validation", "test")}
    labels = sorted(dist)
    params = TrainParams(learning_rate=a.lr, batch_size=a.batch_size, epochs=a.epochs, weight_decay=a.weight_decay,
                         warmup_steps=a.warmup_steps, gradient_accumulation=a.grad_accum, early_stopping_patience=a.patience,
                         freeze_feature_encoder=not a.unfreeze_encoder, imbalance_strategy=a.imbalance, seed=a.seed,
                         augmentation={"enabled": a.augment, "noise_snr_db": [20, 35], "gain_db": [-6, 6], "time_mask_seconds": 0.2})
    cb = Callbacks(on_epoch_end=lambda i: print(f"Epoch {i['epoch']}/{i['epochs']}  train_loss={i['train_loss']:.3f}  val_loss={i['val_loss']:.3f}  "
                                                 f"macroF1={i['macro_f1']:.3f}  (mejor {i['best_macro_f1']:.3f})"))
    res = EmotionTrainer(a.base, labels, params, out, cb).fit(by["train"], by["validation"], by["test"] or None,
                                                              resume=(out / "checkpoint.pt").exists())
    t = res["test"] or res["validation"]
    print(f"\nTEST  accuracy={t['accuracy']:.3f}  macroF1={t['macro_f1']:.3f}  weightedF1={t['weighted_f1']:.3f}  "
          f"precision={t['precision']:.3f}  recall={t['recall']:.3f}")
    print("Matriz de confusión (filas=real):", t["labels"])
    for row in t["confusion_matrix"]:
        print("  ", row)
    print(f"Modelo guardado en {out / 'best'}")


if __name__ == "__main__":
    main()
