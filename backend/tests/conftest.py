"""Configuración de pruebas: BD SQLite y carpetas temporales aisladas; las pruebas ML reutilizan la caché de modelos."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="callan_test_"))
_ROOT = Path(__file__).resolve().parents[1]
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}", "DATA_DIR": str(_TMP), "UPLOAD_DIR": str(_TMP / "uploads"),
    "TEMP_DIR": str(_TMP / "tmp"), "CACHE_DIR": str(_TMP / "cache"), "REPORT_DIR": str(_TMP / "reports"),
    "MODELS_DIR": str(_TMP / "models"), "MODEL_CACHE_DIR": str(_ROOT / "data" / "model_cache"),
    "TASK_MODE": "inline", "SECRET_KEY": "test-secret-key", "LOG_JSON": "false", "LOG_LEVEL": "WARNING",
    "ALLOW_SIGNUP": "true",
})

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

EXAMPLES = _ROOT.parent / "examples" / "call"
RUN_ML = os.environ.get("RUN_ML_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if RUN_ML:
        return
    skip = pytest.mark.skip(reason="defina RUN_ML_TESTS=1 (requiere modelos descargados)")
    for it in items:
        if "ml" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _register(client, org, email):
    r = client.post("/api/v1/auth/register", json={"organization": org, "email": email, "password": "Passw0rd!x", "name": email})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def admin_a(client):
    return {"Authorization": f"Bearer {_register(client, 'Org A', 'admin@acme-a.com')}"}


@pytest.fixture(scope="session")
def admin_b(client):
    return {"Authorization": f"Bearer {_register(client, 'Org B', 'admin@acme-b.com')}"}


@pytest.fixture(scope="session")
def roles_a(client, admin_a):
    out = {}
    for role in ("ANALYST", "VIEWER"):
        email = f"{role.lower()}@acme-a.com"
        r = client.post("/api/v1/users", headers=admin_a, json={"email": email, "password": "Passw0rd!x", "role": role})
        assert r.status_code == 201, r.text
        t = client.post("/api/v1/auth/login", json={"email": email, "password": "Passw0rd!x"}).json()["access_token"]
        out[role] = {"Authorization": f"Bearer {t}"}
    return out


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory):
    """Audio válido pero sin voz (tono): sirve para probar subida/validación sin depender de modelos."""
    p = tmp_path_factory.mktemp("aud") / "tone.wav"
    t = np.arange(16000 * 3) / 16000
    sf.write(p, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
    return p


def make_tiny_emotion_model(path: Path, labels=("angry", "happy", "neutral", "sad")):
    """Wav2Vec2 minúsculo (pesos aleatorios) con la MISMA arquitectura/cabeza que el baseline; sirve para probar
    el entrenamiento y el registro sin descargas. Sus predicciones no tienen significado."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
    from app.ml.emotion.architectures import Wav2Vec2ForSpeechClassification
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, feat_extract_norm="layer",
                         do_stable_layer_norm=True, num_labels=len(labels), final_dropout=0.0,
                         id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)},
                         architectures=["Wav2Vec2ForSpeechClassification"], finetuning_task="wav2vec2_clf", pooling_mode="mean")
    Wav2Vec2ForSpeechClassification(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=True).save_pretrained(path)
    return path
