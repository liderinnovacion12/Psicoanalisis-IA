"""Pruebas de API: autenticación, roles, multi-tenant, subida/validación, errores sin detalle técnico."""
import pytest

API = "/api/v1"


@pytest.fixture(autouse=True)
def no_processing(monkeypatch):
    """Estas pruebas no ejecutan el pipeline ML (se cubre en test_pipeline_ml.py)."""
    monkeypatch.setattr("app.api.calls.enqueue_call", lambda *a, **k: None)


def upload(client, headers, path, name=None, allow=False):
    with open(path, "rb") as f:
        return client.post(f"{API}/calls/upload", headers=headers, files={"file": (name or path.name, f)},
                           data={"allow_training": str(allow).lower()})


def test_requires_authentication(client):
    assert client.get(f"{API}/calls").status_code == 401
    r = client.post(f"{API}/auth/login", json={"email": "nadie@acme-x.com", "password": "mala"})
    assert r.status_code == 401 and "Traceback" not in r.text and r.json()["message"]


def test_first_run_signup_then_closed(client, admin_a):
    st = client.get(f"{API}/auth/setup-status").json()
    assert st["needs_setup"] is False


def test_role_permissions(client, admin_a, roles_a, tone_wav):
    assert upload(client, roles_a["VIEWER"], tone_wav).status_code == 403          # VIEWER solo visualiza
    assert client.get(f"{API}/calls", headers=roles_a["VIEWER"]).status_code == 200
    assert client.post(f"{API}/users", headers=roles_a["ANALYST"], json={"email": "x@acme-a.com", "password": "Passw0rd!x"}).status_code == 403
    assert client.put(f"{API}/settings/audio", headers=roles_a["ANALYST"], json={"value": {}}).status_code == 403
    r = upload(client, roles_a["ANALYST"], tone_wav)
    assert r.status_code == 201


def test_upload_validates_and_hides_technical_errors(client, admin_a, tmp_path):
    bad = tmp_path / "falso.mp3"
    bad.write_bytes(b"esto no es audio" * 100)
    r = upload(client, admin_a, bad)
    assert r.status_code == 422
    assert r.json()["message"] == "No fue posible procesar el audio. Verifique que el archivo sea válido."
    assert "ffmpeg" not in r.text.lower() and "stderr" not in r.text.lower()
    txt = tmp_path / "doc.txt"
    txt.write_text("hola")
    assert upload(client, admin_a, txt).status_code == 422
    empty = tmp_path / "vacio.wav"
    empty.write_bytes(b"")
    assert upload(client, admin_a, empty).status_code == 422


def test_upload_ok_defaults_and_status(client, admin_a, tone_wav):
    r = upload(client, admin_a, tone_wav)
    assert r.status_code == 201
    j = r.json()
    assert j["status"] == "QUEUED" and abs(j["duration"] - 3.0) < 0.2 and j["channels"] == 1
    d = client.get(f"{API}/calls/{j['id']}", headers=admin_a).json()
    assert d["allow_training"] is False                                             # privacidad por defecto
    st = client.get(f"{API}/calls/{j['id']}/status", headers=admin_a).json()
    assert [s["key"] for s in st["stages"]][:3] == ["audio_processing", "diarization", "transcription"]
    assert client.get(f"{API}/calls/{j['id']}/export?format=csv", headers=admin_a).status_code == 409   # aún sin resultados


def test_multi_tenant_isolation(client, admin_a, admin_b, tone_wav):
    cid = upload(client, admin_a, tone_wav).json()["id"]
    assert client.get(f"{API}/calls/{cid}", headers=admin_b).status_code == 404
    assert client.get(f"{API}/calls/{cid}/audio", headers=admin_b).status_code == 404
    assert client.delete(f"{API}/calls/{cid}", headers=admin_b).status_code == 404
    ids_b = [c["id"] for c in client.get(f"{API}/calls", headers=admin_b).json()["items"]]
    assert cid not in ids_b
    ds = client.post(f"{API}/datasets", headers=admin_a, json={"name": "DS_A"}).json()
    assert client.get(f"{API}/datasets/{ds['id']}", headers=admin_b).status_code == 404
    assert client.get(f"{API}/datasets", headers=admin_b).json() == []


def test_list_filters_search_pagination(client, admin_a, tone_wav):
    for n in ("alfa_uno.wav", "beta_dos.wav"):
        upload(client, admin_a, tone_wav, name=n)
    r = client.get(f"{API}/calls", headers=admin_a, params={"q": "alfa_uno"}).json()
    assert r["total"] >= 1 and all("alfa_uno" in c["filename"] for c in r["items"])
    r = client.get(f"{API}/calls", headers=admin_a, params={"page_size": 2, "page": 1}).json()
    assert len(r["items"]) == 2 and r["total"] >= 3
    assert client.get(f"{API}/calls", headers=admin_a, params={"min_satisfaction": 50, "speaker": "SPEAKER_00"}).status_code == 200
    assert client.get(f"{API}/calls", headers=admin_a, params={"status": "COMPLETED"}).json()["items"] == []


def test_audio_range_and_delete(client, admin_a, tone_wav):
    """El reproductor pide el original si aún no hay copia de reproducción; Range → 206."""
    cid = upload(client, admin_a, tone_wav).json()["id"]
    r = client.get(f"{API}/calls/{cid}/audio", headers={**admin_a, "Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100 and r.headers["content-range"].startswith("bytes 0-99/")
    assert client.delete(f"{API}/calls/{cid}", headers=admin_a).status_code == 200
    assert client.get(f"{API}/calls/{cid}", headers=admin_a).status_code == 404


def test_config_validation_and_override(client, admin_a):
    r = client.put(f"{API}/settings/audio", headers=admin_a, json={"value": {"segmentation": {"window_size": 999}}})
    assert r.status_code == 422
    r = client.put(f"{API}/settings/audio", headers=admin_a, json={"value": {"segmentation": {"window_size": 4.0, "hop_size": 2.0}}})
    assert r.status_code == 200 and r.json()["segmentation"]["window_size"] == 4.0
    cur = client.get(f"{API}/settings", headers=admin_a).json()
    assert cur["config"]["audio"]["segmentation"]["window_size"] == 4.0 and cur["defaults"]["audio"]["segmentation"]["window_size"] == 5.0
    client.delete(f"{API}/settings/audio", headers=admin_a)
    assert client.get(f"{API}/settings", headers=admin_a).json()["config"]["audio"]["segmentation"]["window_size"] == 5.0


def test_models_registry_baseline_and_activation_rules(client, admin_a, roles_a):
    ms = client.get(f"{API}/models", headers=admin_a).json()
    base = next(m for m in ms if m["is_builtin"])
    assert base["hf_id"] == "r-f/wav2vec-english-speech-emotion-recognition" and base["status"] == "PRODUCTION" and base["is_active"]
    assert base["language"] == "en"
    assert client.post(f"{API}/models/{base['id']}/activate", headers=roles_a["ANALYST"]).status_code == 403


def test_dashboard_and_system_info(client, admin_a):
    d = client.get(f"{API}/dashboard/summary", headers=admin_a).json()
    assert d["totals"]["analyzed_calls"] == 0 and d["totals"]["total_calls"] >= 1
    info = client.get(f"{API}/system/info", headers=admin_a).json()
    assert info["device"]["label"] in ("CPU MODE",) or info["device"]["label"].startswith("GPU")
    assert info["ffmpeg"] is True


def test_dataset_privacy_gate_and_custom_labels(client, admin_a, tone_wav):
    cid = upload(client, admin_a, tone_wav).json()["id"]
    ds = client.post(f"{API}/datasets", headers=admin_a, json={"name": "DS_PRIV"}).json()
    r = client.post(f"{API}/datasets/{ds['id']}/samples", headers=admin_a,
                    json={"call_id": cid, "speaker": "SPEAKER_00", "emotion": "angry", "start": 0, "end": 2})
    assert r.status_code == 409 and "mejorar el modelo" in r.json()["message"]        # sin permiso explícito no se usa
    assert client.post(f"{API}/labels", headers=admin_a, json={"label": "Ironía leve"}).status_code in (201, 422)
    assert client.post(f"{API}/labels", headers=admin_a, json={"label": "confusion"}).json()["labels"][-1] == "confusion"
