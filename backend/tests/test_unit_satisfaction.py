"""Satisfaction Engine + eventos con series controladas (datos de prueba explícitos, no simulación de la app)."""
import copy

import numpy as np

from app.core.config import get_config_store
from app.ml.satisfaction.engine import SatisfactionEngine
from app.ml.satisfaction.events import detect_events
from app.ml.satisfaction.features import CANONICAL, build_series

SAT = get_config_store().defaults("satisfaction")
EMO = get_config_store().defaults("emotion")


def series_from(profile, seconds=240, speaker="SPEAKER_00", conf=0.9):
    """profile(t) -> dict de probabilidades; una ventana de 5 s cada 2.5 s."""
    wins = []
    for t in np.arange(0, seconds - 5, 2.5):
        p = profile(t)
        s = sum(p.values())
        wins.append({"start": float(t), "end": float(t + 5), "probabilities": {k: v / s for k, v in p.items()}, "confidence": conf})
    return build_series(speaker, CANONICAL, wins, [], 1.0)


def prob(**kw):
    d = {e: 0.01 for e in CANONICAL}
    d.update(kw)
    return d


def engine(**over):
    cfg = copy.deepcopy(SAT)
    for k, v in over.items():
        cfg["weights"][k] = v
    return SatisfactionEngine(cfg, EMO)


def test_score_is_bounded_and_ordered():
    pos = engine().analyze_speaker(series_from(lambda t: prob(happy=0.9)), 240)
    neg = engine().analyze_speaker(series_from(lambda t: prob(angry=0.9)), 240)
    neu = engine().analyze_speaker(series_from(lambda t: prob(neutral=0.95)), 240)
    assert 0 <= neg.score < neu.score < pos.score <= 100
    assert neg.score < 25 and pos.score > 75
    assert neg.interpretation["key"] in ("very_low", "low")


def test_recovery_reflected_in_final_state_and_trend():
    """Empieza muy enojado y termina alegre: la evolución no se pierde promediando toda la llamada."""
    prof = lambda t: prob(angry=0.85) if t < 120 else prob(happy=0.85)
    a = engine().analyze_speaker(series_from(prof), 240)
    assert a.trend == "improving" and a.final_score > a.initial_score + 30
    assert a.metrics["initial_emotion"] == "angry" and a.metrics["final_emotion"] == "happy"


def test_weights_are_configurable_not_hardcoded():
    prof = lambda t: prob(sad=0.9)
    base = engine().analyze_speaker(series_from(prof), 240).score
    harsher = engine(sad_weight=-1.0).analyze_speaker(series_from(prof), 240).score
    milder = engine(sad_weight=0.0).analyze_speaker(series_from(prof), 240).score
    assert harsher < base < milder


def test_final_state_weight_matters():
    prof = lambda t: prob(angry=0.85) if t < 180 else prob(happy=0.85)
    low = engine(final_state_weight=0.0).analyze_speaker(series_from(prof), 240).score
    high = engine(final_state_weight=3.0).analyze_speaker(series_from(prof), 240).score
    assert high > low


def test_explainability_and_separate_confidence():
    a = engine().analyze_speaker(series_from(lambda t: prob(angry=0.8) if t < 60 else prob(happy=0.8)), 240)
    assert a.factors["positive"] and a.factors["negative"]
    assert 0 <= a.confidence <= 1 and "model" in a.details["confidence_parts"]
    short = engine().analyze_speaker(series_from(lambda t: prob(happy=0.9), seconds=30), 30)
    assert short.confidence < a.confidence          # menos evidencia => menos confianza


def test_language_mismatch_lowers_confidence():
    s = series_from(lambda t: prob(happy=0.9))
    a = engine().analyze_speaker(s, 240)
    b = engine().analyze_speaker(s, 240, language_mismatch=True)
    assert b.confidence < a.confidence


def test_timeline_reflects_evolution_and_events():
    prof = lambda t: prob(angry=0.9) if 60 <= t < 120 else prob(neutral=0.9) if t < 60 else prob(happy=0.9)
    s = series_from(prof, seconds=300)
    a = engine().analyze_speaker(s, 300)
    tl = {p["t"]: p["score"] for p in a.timeline}
    assert tl[100.0] < tl[30.0] and tl[280.0] > tl[100.0]
    evs = detect_events({"SPEAKER_00": s}, {"SPEAKER_00": a}, {"SPEAKER_00": []}, SAT, EMO, [])
    types = {e["event_type"] for e in evs}
    assert "anger_peak" in types or "frustration_peak" in types
    assert "recovery" in types or "positive_peak" in types
    assert all(0 <= e["confidence"] <= 1 for e in evs)


def test_possible_conflict_needs_both_speakers():
    hot = lambda t: prob(angry=0.9)
    s0, s1 = series_from(hot, speaker="SPEAKER_00"), series_from(hot, speaker="SPEAKER_01")
    e = engine()
    an = {"SPEAKER_00": e.analyze_speaker(s0, 240), "SPEAKER_01": e.analyze_speaker(s1, 240)}
    evs = detect_events({"SPEAKER_00": s0, "SPEAKER_01": s1}, an, {"SPEAKER_00": [], "SPEAKER_01": []}, SAT, EMO, [])
    assert any(x["event_type"] == "possible_conflict" for x in evs)


def test_text_context_is_a_factor_not_the_verdict():
    s = series_from(lambda t: prob(neutral=0.9))
    s.text_score[:] = -0.9
    with_text = engine().analyze_speaker(s, 240)
    assert "text" in with_text.components and with_text.components["text"] < 0
