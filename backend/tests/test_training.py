"""Entrenamiento real de un wav2vec2 minúsculo (sin descargas): dataset → splits sin fuga → fine-tuning → métricas →
registro → activación; además stop/continuar y modelo de satisfacción.  Las métricas NO miden calidad de un modelo útil."""
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import make_tiny_emotion_model

pytestmark = pytest.mark.slow
API = "/api/v1"
CLASSES = ["angry", "happy", "neutral", "sad"]


@pytest.fixture(scope="module")
def env(client):
    r = client.post(f"{API}/auth/register", json={"organization": "Org Train", "email": "t@acme-t.com", "password": "Passw0rd!x"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    org_id = r.json()["user"]["org_id"]
    from app.database.base import SessionLocal
    from app.ml.training.versioning import register_model
    tiny = make_tiny_emotion_model(Path(tempfile.mkdtemp()) / "tiny")
    db = SessionLocal()
    base = register_model(db, org_id=org_id, kind="emotion", loader="finetuned", path=str(tiny), base_model=None, dataset_id=None,
                          params={}, labels=CLASSES, language="es", created_by=None, status="VALIDATION")
    db.commit()
    bid = base.id
    db.close()
    return {"h": h, "base": bid, "org": org_id}


@pytest.fixture(scope="module")
def dataset(client, env, tmp_path_factory):
    h = env["h"]
    ds = client.post(f"{API}/datasets", headers=h, json={"name": "TRAIN_DS", "language": "es", "labels": CLASSES}).json()
    d = tmp_path_factory.mktemp("samples")
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * 16000)) / 16000
    for person in range(8):                                       # 8 personas x 4 clases x 2 muestras
        for ci, c in enumerate(CLASSES):
            for k in range(2):
                f0 = 120 + 90 * ci + rng.uniform(-8, 8)
                x = 0.3 * np.sin(2 * np.pi * f0 * t) * (1 + 0.5 * np.sin(2 * np.pi * (2 + ci) * t)) + 0.01 * rng.standard_normal(len(t))
                p = d / f"{person}_{c}_{k}.wav"
                sf.write(p, x.astype("float32"), 16000)
                with open(p, "rb") as fh:
                    r = client.post(f"{API}/datasets/{ds['id']}/samples/upload", headers=h, files={"file": (p.name, fh)},
                                    data={"emotion": c, "speaker_group": f"persona_{person}", "language": "es"})
                assert r.status_code == 201, r.text
    return ds["id"]


def wait_run(client, h, rid, states, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"{API}/training/{rid}", headers=h).json()
        if r["status"] in states:
            return r
        time.sleep(1.5)
    raise TimeoutError(rid)


def test_dataset_stats_validation_and_split_without_leakage(client, env, dataset):
    h = env["h"]
    d = client.get(f"{API}/datasets/{dataset}", headers=h).json()
    assert d["n_samples"] == 64 and d["stats"]["emotions"] == {c: 16 for c in CLASSES} and d["stats"]["groups"] == 8
    assert d["total_duration"] == pytest.approx(64 * 1.5, abs=1)
    v = client.post(f"{API}/datasets/{dataset}/validate", headers=h).json()
    assert v["invalid"] == 0
    s = client.post(f"{API}/datasets/{dataset}/split", headers=h, json={"train": .7, "validation": .15, "test": .15}).json()
    assert s["leakage_free"] and set(s["counts"]) == {"train", "validation", "test"}
    items = client.get(f"{API}/datasets/{dataset}/samples", headers=h, params={"page_size": 500}).json()["items"]
    groups = {}
    for it in items:
        groups.setdefault(it["speaker_group"], set()).add(it["split"])
    assert all(len(v) == 1 for v in groups.values())                          # ninguna persona en dos conjuntos


def test_duplicates_and_bad_samples_are_flagged(client, env, dataset, tmp_path):
    h = env["h"]
    ds = client.post(f"{API}/datasets", headers=h, json={"name": "DS_VAL", "labels": CLASSES}).json()["id"]
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(24000) / 16000)).astype("float32")
    sf.write(tmp_path / "a.wav", x, 16000)
    sf.write(tmp_path / "short.wav", x[:2000], 16000)
    sf.write(tmp_path / "silent.wav", np.zeros(24000, "float32"), 16000)
    up = lambda n, **kw: client.post(f"{API}/datasets/{ds}/samples/upload", headers=h, files={"file": (n, open(tmp_path / n, "rb"))}, data=kw)
    assert up("a.wav", emotion="happy").json()["validation"]["ok"]
    assert "duplicado" in up("a.wav", emotion="happy").json()["validation"]["issues"]
    assert "audio_demasiado_corto" in up("short.wav", emotion="sad").json()["validation"]["issues"]
    assert "audio_sin_voz" in up("silent.wav", emotion="sad").json()["validation"]["issues"]
    assert "etiqueta_faltante" in up("a.wav").json()["validation"]["issues"]
    assert client.post(f"{API}/datasets/{ds}/samples/upload", headers=h, files={"file": ("a.wav", open(tmp_path / "a.wav", "rb"))},
                       data={"emotion": "inventada"}).status_code == 422


def test_dataset_versioning(client, env, dataset):
    h = env["h"]
    v2 = client.post(f"{API}/datasets/{dataset}/version", headers=h).json()
    assert v2["version"] == 2 and v2["n_samples"] == 64 and v2["parent_id"] == dataset and v2["code"] == "TRAIN_DS_v2"
    client.post(f"{API}/datasets/{v2['id']}/freeze", headers=h)
    r = client.delete(f"{API}/datasets/{v2['id']}/samples/{client.get(f'{API}/datasets/{v2['id']}/samples', headers=h).json()['items'][0]['id']}", headers=h)
    assert r.status_code == 409


def test_finetune_registers_versioned_model_with_metrics_and_activation(client, env, dataset):
    h = env["h"]
    r = client.post(f"{API}/training/start", headers=h, json={"dataset_id": dataset, "kind": "emotion", "base_model_id": env["base"],
                    "params": {"epochs": 3, "batch_size": 8, "learning_rate": 2e-3, "warmup_steps": 2, "gradient_accumulation": 1,
                               "early_stopping_patience": 0, "freeze_feature_encoder": False}})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert client.post(f"{API}/training/start", headers=h, json={"dataset_id": dataset, "kind": "emotion", "base_model_id": env["base"]}).status_code == 409  # uno a la vez
    run = wait_run(client, h, rid, {"COMPLETED", "FAILED"})
    assert run["status"] == "COMPLETED", run
    assert len(run["metrics_history"]) == 3 and all(0 <= m["macro_f1"] <= 1 and m["train_loss"] > 0 for m in run["metrics_history"])
    m = client.get(f"{API}/models/{run['model_id']}", headers=h).json()
    assert m["name"] == "emotion_model_v2" and m["status"] == "VALIDATION" and m["loader"] == "finetuned" and m["labels"] == CLASSES
    splits = {x["split"]: x for x in m["metrics"]}
    assert {"validation", "test"} <= set(splits)
    t = splits["test"]
    assert t["macro_f1"] is not None and t["weighted_f1"] is not None and np.array(t["confusion_matrix"]).shape == (4, 4)
    assert set(t["per_class"]) == set(CLASSES) and t["n_samples"] > 0
    assert m["parameters"]["n_train"] > 0 and m["parameters"]["dataset"] == "TRAIN_DS_v1"

    cmp = client.get(f"{API}/models/compare", headers=h, params={"ids": [env["base"], m["id"]]}).json()
    assert len(cmp) == 2 and cmp[1]["metrics"]["macro_f1"] is not None

    # el modelo entrenado carga desde el registry y predice con las etiquetas del dataset
    from app.core.config import get_config_store
    from app.database.base import SessionLocal
    from app.ml.emotion.factory import build_model
    from app.models import MLModel
    db = SessionLocal()
    em = build_model(db.get(MLModel, m["id"]), get_config_store().defaults("emotion"))
    out = em.predict([np.random.default_rng(1).standard_normal(16000 * 2).astype("float32") * 0.1])[0]
    assert set(out.probabilities) == set(CLASSES) and abs(sum(out.probabilities.values()) - 1) < 1e-4
    db.close()

    # activar: las nuevas llamadas usarán este modelo (guarda model_version)
    assert client.post(f"{API}/models/{m['id']}/activate", headers=h).status_code == 200
    from app.ml.emotion.factory import get_production_model_row
    db = SessionLocal()
    assert get_production_model_row(db, env["org"], "emotion").id == m["id"]
    db.close()
    ms = {x["id"]: x for x in client.get(f"{API}/models", headers=h).json()}
    assert ms[m["id"]]["is_active"] and ms[m["id"]]["status"] == "PRODUCTION"


def test_stop_and_resume_training(client, env, dataset):
    h = env["h"]
    r = client.post(f"{API}/training/start", headers=h, json={"dataset_id": dataset, "kind": "emotion", "base_model_id": env["base"], "name": "stoppable",
                    "params": {"epochs": 5, "batch_size": 8, "learning_rate": 1e-3, "warmup_steps": 1, "gradient_accumulation": 1, "early_stopping_patience": 0}})
    rid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < 300:
        cur = client.get(f"{API}/training/{rid}", headers=h).json()
        if cur["current_epoch"] >= 1:
            break
        time.sleep(0.5)
    assert client.post(f"{API}/training/{rid}/stop", headers=h).status_code == 200
    stopped = wait_run(client, h, rid, {"STOPPED", "COMPLETED", "FAILED"})
    assert stopped["status"] == "STOPPED" and stopped["resumable"]
    assert client.post(f"{API}/training/{rid}/resume", headers=h).status_code == 200
    done = wait_run(client, h, rid, {"COMPLETED", "FAILED"})
    assert done["status"] == "COMPLETED" and done["current_epoch"] == 5 and len(done["metrics_history"]) == 5     # continuó desde el checkpoint


def test_satisfaction_regressor_training_and_calibration():
    from app.ml.satisfaction.model import SatisfactionRegressor, fit_linear_calibration, train_regressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 39))
    y = np.clip(50 + 12 * X[:, 0] - 8 * X[:, 5] + rng.normal(scale=4, size=120), 0, 100)
    reg, m = train_regressor(X, y, groups=[i // 3 for i in range(120)])
    assert m["mae"] < 8 and m["r2"] > 0.6 and m["cv"] == "grouped"
    p = Path(tempfile.mkdtemp()) / "m.joblib"
    reg.save(p)
    assert abs(SatisfactionRegressor.load(p).predict(X[0]) - reg.predict(X[0])) < 1e-9
    pred = y * 0.7 + 5                                             # sesgo sistemático
    cal = fit_linear_calibration(pred, y)
    assert cal["mae_after"] < cal["mae_before"] * 0.5
