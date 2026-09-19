"""Textos interpretativos en lenguaje prudente ("el modelo estima…"). Separa DATOS OBSERVADOS de INTERPRETACIÓN."""
from __future__ import annotations

EMOTION_ES = {"angry": "enojo", "disgust": "disgusto", "fear": "miedo", "happy": "alegría", "neutral": "neutralidad",
              "sad": "tristeza", "surprise": "sorpresa"}
TREND_ES = {"improving": "positiva", "declining": "negativa", "stable": "estable"}
EVENT_ES = {
    "frustration_peak": "Frustración elevada", "anger_peak": "Pico de enojo", "positive_peak": "Pico de emoción positiva",
    "emotional_shift": "Cambio emocional importante", "recovery": "Recuperación emocional",
    "deterioration": "Deterioro emocional", "satisfaction_jump": "Cambio brusco de satisfacción",
    "possible_conflict": "Posible conflicto", "positive_ending": "Final positivo", "negative_ending": "Final negativo",
}


def emo(e: str | None) -> str:
    return EMOTION_ES.get(e or "", e or "sin dato")


def confidence_word(c: float) -> str:
    return "alta" if c >= 0.75 else "moderada" if c >= 0.5 else "baja"


def person_name(label: str, role: str | None, roles_labels: dict | None = None) -> str:
    idx = 1 + int(label.split("_")[-1]) if label.split("_")[-1].isdigit() else 0
    base = f"Persona {idx}"
    if role and role != "other":
        rl = (roles_labels or {}).get(role, role.capitalize())
        return f"{base} ({rl})"
    return base


def executive_summary(speakers: dict, interaction: dict, events: list[dict], names: dict[str, str]) -> str:
    parts: list[str] = []
    for label, sp in speakers.items():
        m = sp["metrics"]
        nm = names.get(label, label)
        traj = (f"desde {emo(m['initial_emotion'])} como señal inicial predominante hacia {emo(m['final_emotion'])} "
                f"en el tramo final")
        parts.append(
            f"Durante la llamada, {nm} presentó una evolución emocional {traj}; el modelo estima una satisfacción de "
            f"{sp['score']:.0f}/100 (confianza {confidence_word(sp['confidence'])}) y una tendencia {TREND_ES[sp['trend']]}.")
    if interaction and interaction.get("initial") is not None and interaction.get("final") is not None:
        parts.append(
            f"A nivel de interacción, la satisfacción estimada pasó de {interaction['initial']:.0f} a {interaction['final']:.0f} "
            f"(variación {interaction['variation']:+.0f}).")
    if events:
        kinds = {}
        for e in events:
            kinds[e["event_type"]] = kinds.get(e["event_type"], 0) + 1
        top = ", ".join(f"{EVENT_ES.get(k, k).lower()} ({n})" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:4])
        parts.append(f"Se detectaron {len(events)} momentos destacados: {top}.")
    parts.append("Estos resultados son estimaciones de un modelo y no constituyen una afirmación definitiva sobre el estado "
                 "de los participantes; se recomienda revisión humana de los momentos críticos.")
    return " ".join(parts)


def interaction_message(variation: float | None, stable_delta: float = 6) -> str:
    if variation is None:
        return "No hay datos suficientes para describir la evolución."
    if variation > stable_delta:
        return "Tendencia positiva durante la interacción."
    if variation < -stable_delta:
        return "Tendencia negativa durante la interacción."
    return "Satisfacción estimada estable durante la interacción."
