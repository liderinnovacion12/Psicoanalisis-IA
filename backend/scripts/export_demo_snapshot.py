"""Genera los datos del MODO DEMOSTRACIÓN del frontend (sin backend) a partir de ejecuciones REALES de la aplicación.

    python scripts/export_demo_snapshot.py

Qué hace (todo con el código real, contra una BD temporal en data/demo):
  1. Sube y analiza la llamada de ejemplo estéreo (Whisper + wav2vec baseline + Satisfaction Engine).
  2. Crea un dataset de juguete (tonos sintéticos, NO emociones reales), lo particiona y ejecuta un fine-tuning real de un
     wav2vec2 minúsculo para poblar las pantallas de Dataset / Entrenamiento / Modelos.
  3. Guarda las respuestas JSON de la API (mismas formas que en producción), el audio y los PDF/CSV/XLSX/JSON exportados en
     frontend/lib/demo-data.json y frontend/public/demo/.
Nada se inventa: solo se serializan respuestas reales. La voz de la llamada es sintética (TTS) y los tonos del dataset también:
la UI lo rotula como demostración.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEMO = ROOT / "data" / "demo"
shutil.rmtree(DEMO, ignore_errors=True)
DEMO.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "DATABASE_URL": f"sqlite:///{DEMO / 'demo.db'}", "DATA_DIR": str(DEMO), "UPLOAD_DIR": str(DEMO / "uploads"),
    "TEMP_DIR": str(DEMO / "tmp"), "CACHE_DIR": str(DEMO / "cache"), "REPORT_DIR": str(DEMO / "reports"),
    "MODELS_DIR": str(DEMO / "models"), "MODEL_CACHE_DIR": str(ROOT / "data" / "model_cache"), "TASK_MODE": "inline",
    "SECRET_KEY": "demo-secret", "LOG_JSON": "false", "LOG_LEVEL": "WARNING", "ALLOW_SIGNUP": "true",
})
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

API = "/api/v1"
OUT_JSON = REPO / "frontend" / "lib" / "demo-data.json"
OUT_DIR = REPO / "frontend" / "public" / "demo"
CLASSES = ["angry", "happy", "neutral", "sad"]


def tiny_model(path: Path) -> Path:
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
    from app.ml.emotion.architectures import Wav2Vec2ForSpeechClassification
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16,) * 7,
                         conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, feat_extract_norm="layer", do_stable_layer_norm=True,
                         num_labels=len(CLASSES), final_dropout=0.0, id2label=dict(enumerate(CLASSES)),
                         label2id={c: i for i, c in enumerate(CLASSES)}, architectures=["Wav2Vec2ForSpeechClassification"],
                         finetuning_task="wav2vec2_clf", pooling_mode="mean")
    Wav2Vec2ForSpeechClassification(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=True).save_pretrained(path)
    return path


def wait(fn, ok, timeout=1200, every=3):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if ok(r):
            return r
        time.sleep(every)
    raise TimeoutError


def main():
    snap: dict = {"routes": {}, "models": {}, "generated_from": "ejecución real de la aplicación (ver scripts/export_demo_snapshot.py)"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        r = c.post(f"{API}/auth/register", json={"organization": "Demostración", "name": "Usuario demo", "email": "demo@demo.com",
                                                 "password": "DemoDemo123!"})
        assert r.status_code == 200, r.text
        h = {"Authorization": f"Bearer {r.json()['access_token']}"}
        me = r.json()["user"]
        get = lambda p, **kw: c.get(API + p, headers=h, **kw)

        # ---- 1. llamada real ----
        print("Subiendo y analizando la llamada de ejemplo (modelos reales)…")
        with open(REPO / "examples" / "call" / "llamada_ejemplo_estereo.wav", "rb") as f:
            up = c.post(f"{API}/calls/upload", headers=h, files={"file": ("llamada_ejemplo_estereo.wav", f)},
                        data={"display_name": "Llamada de ejemplo (voz sintética)"})
        cid = up.json()["id"]
        st = wait(lambda: get(f"/calls/{cid}/status").json(), lambda s: s["status"] in ("COMPLETED", "ERROR"))
        assert st["status"] == "COMPLETED", st

        # ---- 2. dataset + entrenamiento de juguete (real, con datos sintéticos) ----
        print("Creando dataset de demostración y entrenando un modelo de juguete…")
        from app.database.base import SessionLocal
        from app.ml.training.versioning import register_model
        tiny = tiny_model(DEMO / "tiny_base")
        db = SessionLocal()
        org_id = me["org_id"]
        base = register_model(db, org_id=org_id, kind="emotion", loader="finetuned", path=str(tiny), base_model=None, dataset_id=None,
                              params={"note": "modelo minúsculo de demostración"}, labels=CLASSES, language="es", created_by=None,
                              status="VALIDATION")
        db.commit()
        base_id = base.id
        db.close()
        ds = c.post(f"{API}/datasets", headers=h, json={"name": "DEMO_TONOS_SINTETICOS", "language": "es", "labels": CLASSES,
                    "description": "Dataset de DEMOSTRACIÓN: tonos sintéticos, NO son emociones reales. Solo muestra la interfaz."}).json()
        rng = np.random.default_rng(0)
        t = np.arange(int(1.5 * 16000)) / 16000
        tmpd = Path(tempfile.mkdtemp())
        for person in range(8):
            for ci, cl in enumerate(CLASSES):
                for k in range(2):
                    x = 0.3 * np.sin(2 * np.pi * (120 + 90 * ci + rng.uniform(-8, 8)) * t) * (1 + 0.5 * np.sin(2 * np.pi * (2 + ci) * t)) \
                        + 0.01 * rng.standard_normal(len(t))
                    p = tmpd / f"{person}_{cl}_{k}.wav"
                    sf.write(p, x.astype("float32"), 16000)
                    with open(p, "rb") as fh:
                        assert c.post(f"{API}/datasets/{ds['id']}/samples/upload", headers=h, files={"file": (p.name, fh)},
                                      data={"emotion": cl, "speaker_group": f"persona_{person}", "language": "es"}).status_code == 201
        c.post(f"{API}/datasets/{ds['id']}/validate", headers=h)
        c.post(f"{API}/datasets/{ds['id']}/split", headers=h, json={"train": .7, "validation": .15, "test": .15})
        run = c.post(f"{API}/training/start", headers=h, json={"dataset_id": ds["id"], "kind": "emotion", "base_model_id": base_id,
                     "name": "Entrenamiento de demostración (tonos)", "params": {"epochs": 5, "batch_size": 8, "learning_rate": 2e-3,
                     "warmup_steps": 2, "gradient_accumulation": 1, "early_stopping_patience": 0, "freeze_feature_encoder": False}}).json()
        wait(lambda: get(f"/training/{run['id']}").json(), lambda r: r["status"] in ("COMPLETED", "FAILED"), 900, 2)

        # ---- 3. instantánea de respuestas ----
        print("Guardando respuestas de la API…")
        R = snap["routes"]
        R["/auth/me"] = me
        R["/auth/setup-status"] = {"needs_setup": False, "signup_enabled": False}
        R["/calls"] = get("/calls", params={"page_size": 200}).json()
        for sub in ("", "/status", "/speakers", "/emotions", "/satisfaction", "/events", "/interaction", "/segments", "/feedback"):
            R[f"/calls/:id{sub}"] = get(f"/calls/{cid}{sub}").json()
        R["/calls/:id/transcription"] = get(f"/calls/{cid}/transcription").json()
        R["/calls/:id/transcription?redact"] = get(f"/calls/{cid}/transcription", params={"redact": True}).json()
        R["/dashboard/summary"] = get("/dashboard/summary").json()
        info = get("/system/info").json()
        info["task_mode"] = "demo"
        R["/system/info"] = info
        R["/labels"] = get("/labels").json()
        R["/datasets"] = get("/datasets").json()
        R["/datasets/:id"] = get(f"/datasets/{ds['id']}").json()
        R["/datasets/:id/samples"] = get(f"/datasets/{ds['id']}/samples", params={"page_size": 15}).json()
        R["/training"] = get("/training").json()
        R["/settings"] = get("/settings").json()
        R["/settings/calibration/status"] = get("/settings/calibration/status").json()
        R["/users"] = get("/users").json()
        models = get("/models").json()
        R["/models"] = models
        for m in models:
            snap["models"][m["id"]] = get(f"/models/{m['id']}").json()
        trained = next(m for m in models if not m["is_builtin"] and m["id"] != base_id and m["kind"] == "emotion")
        R["/models/compare"] = get("/models/compare", params={"ids": [base_id, trained["id"]]}).json()
        snap["ids"] = {"call": cid, "dataset": ds["id"], "run": run["id"], "trained_model": trained["id"]}

        # ---- 4. archivos: audio y exportaciones reales ----
        (OUT_DIR / "audio.mp3").write_bytes(get(f"/calls/{cid}/audio").content)
        (OUT_DIR / "reporte.pdf").write_bytes(get(f"/reports/{cid}").content)
        for fmt in ("csv", "xlsx", "json"):
            (OUT_DIR / f"exportacion.{fmt}").write_bytes(get(f"/calls/{cid}/export", params={"format": fmt}).content)

    OUT_JSON.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {OUT_JSON} ({OUT_JSON.stat().st_size // 1024} KB) y archivos en {OUT_DIR}")


if __name__ == "__main__":
    main()
