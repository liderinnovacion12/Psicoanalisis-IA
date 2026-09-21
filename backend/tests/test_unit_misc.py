import numpy as np
import pytest

from app.core.config import get_config_store
from app.ml.diarization.base import Turn, merge_turns, reduce_to_two
from app.ml.emotion.preprocessing import plan_windows
from app.ml.privacy.pii import redact_text
from app.ml.prosody.interaction import interaction_metrics
from app.ml.text.sentiment import LexiconTextAnalyzer
from app.ml.training.dataset import compute_class_weights, imbalance_report
from app.ml.training.evaluator import metrics_from_predictions
from app.ml.training.splits import group_stratified_split
from app.ml.transcription.base import Word, assign_words_to_speakers, group_utterances, plan_blocks
from app.utils import crypto


def test_windows_cover_long_turn_with_overlap_and_tail():
    w = plan_windows([Turn("SPEAKER_00", 0, 3600)], 5.0, 2.5, 1.0)          # 60 min
    assert len(w) >= 1439 and w[0].start == 0 and w[-1].end == 3600
    assert all(abs(b.start - a.start - 2.5) < 1e-6 for a, b in zip(w[:-2], w[1:-1]))
    short = plan_windows([Turn("SPEAKER_01", 10, 12.5)], 5.0, 2.5, 1.0)
    assert len(short) == 1 and short[0].duration == 2.5                      # turno corto: una ventana
    assert plan_windows([Turn("SPEAKER_01", 10, 10.4)], 5, 2.5, 1.0) == []


def test_windows_are_deterministic_for_resume():
    turns = [Turn("SPEAKER_00", i * 20.0, i * 20.0 + 14) for i in range(50)]
    a, b = plan_windows(turns), plan_windows(turns)
    assert [(x.index, x.start) for x in a] == [(x.index, x.start) for x in b]


def test_group_split_has_no_leakage_and_is_stratified():
    samples = [{"id": f"s{g}_{i}", "label": ["angry", "happy", "neutral"][(g + i) % 3], "group": f"person{g}"}
               for g in range(30) for i in range(8)]
    assign, info = group_stratified_split(samples, {"train": .7, "validation": .15, "test": .15}, 1)
    assert info["leakage_free"]
    by_group = {}
    for s in samples:
        by_group.setdefault(s["group"], set()).add(assign[s["id"]])
    assert all(len(v) == 1 for v in by_group.values())                       # ningún grupo en dos splits
    assert abs(sum(1 for v in assign.values() if v == "test") / len(samples) - .15) < .1
    assert set(assign.values()) == {"train", "validation", "test"}


def test_group_split_warns_when_too_few_groups():
    samples = [{"id": str(i), "label": "a" if i % 2 else "b", "group": "only_one"} for i in range(20)]
    _, info = group_stratified_split(samples, {"train": .7, "validation": .15, "test": .15})
    assert not info["leakage_free"] and info["warnings"]


def test_imbalance_alert_and_class_weights():
    dist = {"neutral": 80, "angry": 5, "happy": 5, "sad": 4, "fear": 3, "disgust": 2, "surprise": 1}
    rep = imbalance_report(dist, 0.5)
    assert rep["imbalanced"] and rep["message"]
    w = compute_class_weights(["neutral"] * 80 + ["angry"] * 5, ["neutral", "angry", "happy"])
    assert w[1] > w[0] and w[2] == 0.0


def test_metrics_confusion_matrix():
    m = metrics_from_predictions(["a", "a", "b", "b", "c"], ["a", "b", "b", "b", "a"], ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.6) and m["labels"] == ["a", "b", "c"]
    assert np.array(m["confusion_matrix"]).sum() == 5 and m["per_class"]["c"]["recall"] == 0


def test_pii_detection_and_redaction():
    t, spans = redact_text("Mi nombre es Juan Pérez, mi correo es juan@mail.com y mi teléfono 555-123-4567. Llevo tres días esperando.")
    assert "[PERSONA]" in t and "[CORREO]" in t and "[TELEFONO]" in t and "Juan" not in t
    assert "Llevo tres días" in t and len(spans) == 3
    assert all("text" not in s for s in spans)                               # no se guardan los datos sensibles
    assert redact_text("Estoy muy bien, soy feliz")[0] == "Estoy muy bien, soy feliz"


def test_text_context_does_not_treat_politeness_as_satisfaction():
    a = LexiconTextAnalyzer()
    assert abs(a.analyze("Gracias por llamar, ¿en qué puedo ayudarle?").score) < 0.15
    assert a.analyze("Ya estoy cansado de llamar y nadie me soluciona nada").score < -0.3
    assert a.analyze("Muchas gracias, quedó resuelto, excelente servicio").score > 0.3
    assert a.analyze("no me ayudó en nada").score <= 0


def test_word_to_speaker_assignment_splits_utterance_at_turn_change():
    words = [Word(0.0, 0.4, " Hola"), Word(0.4, 0.8, " buenas"), Word(1.0, 1.4, " sí"), Word(1.4, 1.9, " dígame")]
    turns = [Turn("SPEAKER_00", 0, 0.9), Turn("SPEAKER_01", 0.95, 2.0)]
    utts = group_utterances(assign_words_to_speakers(words, turns), "es")
    assert [(u.speaker, u.text) for u in utts] == [("SPEAKER_00", "Hola buenas"), ("SPEAKER_01", "sí dígame")]


def test_word_tokens_without_leading_space_attach_to_previous():
    words = [Word(0, .3, " Hola"), Word(.3, .35, ","), Word(.4, .8, " mundo"), Word(.8, .85, ".")]
    u = group_utterances([("SPEAKER_00", w) for w in words], "es")[0]
    assert u.text == "Hola, mundo."


def test_transcript_block_planning_cuts_in_silence():
    vad = [(0, 590), (596, 1190), (1195, 1800)]
    blocks = plan_blocks(1800, vad, 600)
    assert blocks[0][1] == pytest.approx(593, abs=1) and blocks[-1][1] == 1800
    assert all(b - a <= 700 for a, b in blocks) and sum(b - a for a, b in blocks) == pytest.approx(1800)


def test_diarization_reduces_to_two_and_warns():
    raw = {"A": [(0, 50), (100, 150)], "B": [(50, 100)], "C": [(160, 165), (170, 190)]}
    turns, warnings, info = reduce_to_two(raw, None, {"minor_speaker_ratio": 0.05})
    assert {t.speaker for t in turns} == {"SPEAKER_00", "SPEAKER_01"}
    assert any("más de dos" in w for w in warnings)
    assert merge_turns([Turn("SPEAKER_00", 0, 1), Turn("SPEAKER_00", 1.2, 2)], 0.5)[0].end == 2


def test_interruptions_and_silences():
    turns = [Turn("SPEAKER_00", 0, 10), Turn("SPEAKER_01", 8, 15), Turn("SPEAKER_00", 25, 30)]
    m, _ = interaction_metrics(turns, 30, get_config_store().defaults("audio"), {"SPEAKER_00": 30, "SPEAKER_01": 20})
    assert m["interruptions"]["SPEAKER_01"] == 1 and m["interruptions"]["SPEAKER_00"] == 0
    assert m["long_silences"] == 1 and m["overlap_seconds"] == pytest.approx(2.0)


def test_encryption_random_access(tmp_path):
    key = crypto.derive_key("k")
    data = np.random.default_rng(0).bytes(3 * crypto.CHUNK + 12345)
    p = tmp_path / "f.enc"
    with crypto.EncryptedWriter(p, key) as w:
        for i in range(0, len(data), 700_000):
            w.write(data[i:i + 700_000])
    assert crypto.is_encrypted(p) and data[:100] not in p.read_bytes()
    assert b"".join(crypto.read_range(p, key)) == data
    a, b = crypto.CHUNK - 10, 2 * crypto.CHUNK + 50
    assert b"".join(crypto.read_range(p, key, a, b)) == data[a:b + 1]
    assert crypto.plain_size(p) == len(data)


def test_embedding_decision_one_vs_two_speakers():
    """Con embeddings sintéticos: misma voz con variación -> 1 persona; dos direcciones muy distintas -> 2."""
    from app.ml.diarization.embedding import DEFAULTS, cluster_two, looks_like_two_speakers
    rng = np.random.default_rng(0)
    base = rng.normal(size=512); base /= np.linalg.norm(base)
    other = rng.normal(size=512); other -= other @ base * base; other /= np.linalg.norm(other)
    norm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    same = norm(base + 0.25 * rng.normal(size=(60, 512)) / np.sqrt(512) * 3)                 # una voz, variación moderada
    two = norm(np.vstack([base + 0.2 * rng.normal(size=(30, 512)) / np.sqrt(512) * 3,
                          other + 0.2 * rng.normal(size=(30, 512)) / np.sqrt(512) * 3]))     # dos voces ortogonales
    _, s1 = cluster_two(same)
    _, s2 = cluster_two(two)
    assert not looks_like_two_speakers(s1, DEFAULTS) and s1["centroid_similarity"] > 0.72
    assert looks_like_two_speakers(s2, DEFAULTS) and s2["gap"] > 0.25
    tiny = norm(np.vstack([base + 0.1 * rng.normal(size=(58, 512)) / np.sqrt(512), other[None] * np.ones((2, 1))]))
    assert not looks_like_two_speakers(cluster_two(tiny)[1], DEFAULTS)                          # 2 ventanas atípicas no son una persona


# ---- fusión audio+texto, temperatura, migración de columnas, repeticiones ---------------------------------------
def test_temperature_scaling_matches_softmax_of_scaled_logits():
    from app.ml.emotion.fusion import temper
    z = np.array([4.0, 1.0, -1.0, 0.5])
    p = np.exp(z) / np.exp(z).sum()
    for T in (0.5, 2.0, 7.5):
        want = np.exp(z / T) / np.exp(z / T).sum()
        got = temper({str(i): float(v) for i, v in enumerate(p)}, T)
        assert np.allclose([got[str(i)] for i in range(4)], want, atol=1e-9)
    hot = temper({"a": 0.99, "b": 0.01}, 7.5)
    assert hot["a"] < 0.7 and max(hot, key=hot.get) == "a"                      # aplana pero no cambia la emoción principal


def test_fusion_weights_agreement_and_language():
    from app.ml.emotion.fusion import TextTrack, fuse, weights_for
    from app.core.config import get_config_store
    cfg = get_config_store().defaults("emotion")
    assert weights_for(True, cfg)[0] > weights_for(True, cfg)[1] and weights_for(False, cfg)[1] > weights_for(False, cfg)[0]
    audio = {"angry": 0.1, "happy": 0.7, "neutral": 0.2}
    text = {"angry": 0.9, "happy": 0.0, "neutral": 0.1, "sad": 0.0}               # 'sad' no existe en este modelo de audio
    f, src = fuse(audio, text, 0.35, 0.65)
    assert abs(sum(f.values()) - 1) < 1e-9 and max(f, key=f.get) == "angry"       # con desajuste de idioma manda el texto
    assert src["weights"]["text"] > src["weights"]["audio"] and 0 <= src["agreement"] <= 1 and src["agreement"] < 0.5
    f2, src2 = fuse(audio, text, 0.7, 0.3)
    assert f2["angry"] < f["angry"]                                               # con el mismo idioma pesa menos el texto
    same = fuse({"angry": 0.8, "happy": 0.2}, {"angry": 0.9, "happy": 0.1}, 0.5, 0.5)[1]["agreement"]
    assert same > 0.9                                                             # coinciden -> concordancia alta
    f3, src3 = fuse(audio, None, 0.35, 0.65)                                      # sin texto: solo audio, sin inventar
    assert f3 == audio and src3["text"] is None and src3["agreement"] is None
    tr = TextTrack([(0.0, 5.0, {"angry": 1.0}), (5.0, 10.0, {"happy": 1.0})])
    assert tr.query(1, 4)["angry"] == 1.0 and abs(tr.query(3, 7)["angry"] - 0.5) < 1e-9 and tr.query(20, 25) is None


def test_ensure_columns_adds_missing_nullable_columns(tmp_path):
    from sqlalchemy import create_engine, inspect, text
    from app.database.base import ensure_columns
    import app.models  # noqa: F401
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as cx:                                                        # esquema antiguo: sin la columna `sources`
        cx.execute(text("CREATE TABLE emotion_predictions (id VARCHAR(32) PRIMARY KEY, org_id VARCHAR(32), call_id VARCHAR(32), "
                        "speaker_id VARCHAR(32), start FLOAT, \"end\" FLOAT, duration FLOAT, emotion VARCHAR(64), confidence FLOAT, "
                        "probabilities JSON, prosody JSON, segment_id VARCHAR(32), model_id VARCHAR(32), model_name VARCHAR(200), "
                        "model_version VARCHAR(100))"))
    added = ensure_columns(eng)
    assert "emotion_predictions.sources" in added
    assert "sources" in {c["name"] for c in inspect(eng).get_columns("emotion_predictions")}
    assert ensure_columns(eng) == [] or "emotion_predictions.sources" not in ensure_columns(eng)   # idempotente


def test_whisper_repeat_loops_are_collapsed_and_model_size_auto():
    from app.ml.transcription.base import collapse_repeats
    from app.ml.transcription.whisper import resolve_model_size
    assert collapse_repeats("Gracias por su llamada gracias por su llamada gracias por su llamada gracias por su llamada hasta luego") \
        == "Gracias por su llamada hasta luego"
    assert collapse_repeats("Sí, sí, estoy de acuerdo con usted.") == "Sí, sí, estoy de acuerdo con usted."     # repetición normal intacta
    assert resolve_model_size("auto", "cuda") == "large-v3" and resolve_model_size("auto", "cpu") == "small"
    assert resolve_model_size("medium", "cpu") == "medium"
