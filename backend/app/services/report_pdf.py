"""Reporte PDF profesional (ReportLab + matplotlib). Lenguaje prudente: separa datos observados de interpretación."""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from app.services import narrative  # noqa: E402

NAVY, TEAL, GRAY = colors.HexColor("#1F3A5F"), colors.HexColor("#0E7C86"), colors.HexColor("#5B6770")
EMO_COLORS = {"angry": "#D64545", "disgust": "#8A6D3B", "fear": "#7D5BA6", "happy": "#2E9E5B", "neutral": "#8C99A6",
              "sad": "#3B6FB6", "surprise": "#E0A030"}
SPK_COLORS = {"SPEAKER_00": "#0E7C86", "SPEAKER_01": "#C2571A"}


def mmss(t: float | None) -> str:
    if t is None:
        return "-"
    t = int(round(t))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def _fig_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_satisfaction(data: dict, names: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for lab, s in data["satisfaction"]["speakers"].items():
        tl = s.get("timeline", [])
        if tl:
            ax.plot([p["t"] / 60 for p in tl], [p["score"] for p in tl], lw=2, color=SPK_COLORS.get(lab), label=names.get(lab, lab))
    for e in data["events"]:
        if e["event_type"] in ("frustration_peak", "recovery", "deterioration") and e["speaker"]:
            ax.axvline(e["timestamp"] / 60, color=SPK_COLORS.get(e["speaker"], "#999"), alpha=0.25, lw=1)
    ax.axhspan(0, 40, color="#D64545", alpha=0.06); ax.axhspan(60, 100, color="#2E9E5B", alpha=0.06)
    ax.set_ylim(0, 100); ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Satisfacción estimada (0-100)")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return _fig_png(fig)


def chart_emotions(data: dict, speaker: str) -> io.BytesIO | None:
    pts = [p for p in data["emotions"] if p["speaker"] == speaker]
    if not pts:
        return None
    t = np.array([(p["start"] + p["end"]) / 120 for p in pts])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    labels = list(pts[0]["probabilities"].keys())
    k = 3
    for l in labels:
        y = np.array([p["probabilities"].get(l, 0) for p in pts])
        if len(y) >= k:
            y = np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        ax.plot(t, y, lw=1.4, color=EMO_COLORS.get(l, "#444"), label=l)
    ax.set_ylim(0, 1.02); ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Probabilidad")
    ax.grid(alpha=0.25); ax.legend(ncol=7, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_png(fig)


def chart_distribution(summary_sp: dict, names: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    emos = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    x = np.arange(len(emos))
    w = 0.38
    for i, (lab, s) in enumerate(summary_sp.items()):
        mp = s["metrics"]["mean_probabilities"]
        ax.bar(x + (i - 0.5) * w, [mp.get(e, 0) * 100 for e in emos], w, color=SPK_COLORS.get(lab), label=names.get(lab, lab))
    ax.set_xticks(x); ax.set_xticklabels([narrative.emo(e) for e in emos], fontsize=8)
    ax.set_ylabel("% promedio"); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return _fig_png(fig)


def build_pdf(data: dict) -> bytes:
    call = data["call"]
    summ = call.get("summary") or {}
    sp = summ.get("speakers", {})
    names = {s["label"]: s["name"] for s in data["speakers"]}
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=18, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=6)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=13)
    SM = ParagraphStyle("SM", parent=B, fontSize=8, textColor=GRAY, leading=11)
    NOTE = ParagraphStyle("NOTE", parent=B, backColor=colors.HexColor("#F1F5F8"), borderPadding=6, fontSize=8.6)

    def tbl(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), 8.6), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
              ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DBE1")), ("TOPPADDING", (0, 0), (-1, -1), 3),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(st))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=f"Análisis de llamada {call['id'][:8]}", author="Analizador de Llamadas IA")
    W = A4[0] - 3.6 * cm
    el: list = []

    # ---- portada ----
    el += [Spacer(1, 5 * cm), Paragraph("ANÁLISIS DE LLAMADA", ParagraphStyle("t", parent=H1, fontSize=30, alignment=TA_CENTER)),
           Paragraph("Emociones y satisfacción estimadas", ParagraphStyle("s", parent=B, fontSize=14, alignment=TA_CENTER, textColor=TEAL)),
           Spacer(1, 1.2 * cm),
           tbl([["ID", call["id"]], ["Archivo", call["filename"]], ["Fecha de la llamada", (call.get("created_at") or "")[:19].replace("T", " ")],
                ["Duración", mmss(call.get("duration"))], ["Modelo emocional", call.get("model_version") or "-"],
                ["Generado", datetime.now().strftime("%Y-%m-%d %H:%M")]], [4.5 * cm, W - 4.5 * cm], header=False),
           Spacer(1, 1.2 * cm),
           Paragraph("Este informe presenta estimaciones generadas por modelos de aprendizaje automático. No constituye una "
                     "afirmación definitiva sobre el estado emocional ni la satisfacción de los participantes y debe interpretarse "
                     "junto con la revisión humana de los momentos destacados.", NOTE), PageBreak()]

    # ---- información ----
    aq, an = call.get("audio_quality_detail") or {}, call.get("analysis_quality_detail") or {}
    el += [Paragraph("Información de la llamada", H1),
           tbl([["ID", call["id"]], ["Fecha", (call.get("created_at") or "")[:19].replace("T", " ")], ["Duración", mmss(call.get("duration"))],
                ["Idioma", f"{call.get('language') or '-'} (confianza {round((call.get('language_confidence') or 0) * 100)} %)" if call.get("language") else "-"],
                ["Participantes", ", ".join(names.values()) or "-"], ["Diarización", call.get("diarization_mode") or "-"],
                ["Calidad de audio", f"{aq.get('score', '-')}/100"], ["Calidad del análisis", f"{an.get('score', '-')}/100"]],
               [4.5 * cm, W - 4.5 * cm], header=False)]
    if call.get("warnings"):
        el += [Spacer(1, 6), Paragraph("<b>Advertencias:</b> " + " ".join(dict.fromkeys(call["warnings"])), NOTE)]
    if an.get("components"):
        c = an["components"]
        lab = {"audio": "Audio", "speech": "Voz", "segments": "Segmentos válidos", "model": "Modelo emocional*",
               "diarization": "Diarización", "transcription": "Transcripción"}
        el += [Paragraph("Calidad del análisis por componente", H2),
               tbl([["Componente", "Puntaje (0-100)"]] + [[lab[k], f"{v:.0f}"] for k, v in c.items()], [8 * cm, 4 * cm]),
               Paragraph("* Estimada a partir de la confianza del modelo; NO equivale a su exactitud real en llamadas.", SM)]

    # ---- personas ----
    for lab in ("SPEAKER_00", "SPEAKER_01"):
        s = sp.get(lab)
        if not s:
            continue
        m = s["metrics"]
        el += [PageBreak(), Paragraph(names.get(lab, lab).upper(), H1)]
        rows = [["Satisfacción estimada", f"{s['score']:.0f}/100  ({s['interpretation']['label']})"],
                ["Confianza del análisis", f"{s['confidence'] * 100:.0f} %  ({narrative.confidence_word(s['confidence'])})"],
                ["Emoción predominante", narrative.emo(m["dominant_emotion"])], ["Emoción inicial", narrative.emo(m["initial_emotion"])],
                ["Emoción final", narrative.emo(m["final_emotion"])], ["Frustración", f"{m['frustration'] * 100:.0f} %"],
                ["Tensión", f"{m['tension'] * 100:.0f} %"], ["Emociones positivas", f"{m['positive'] * 100:.0f} %"],
                ["Emociones negativas", f"{m['negative'] * 100:.0f} %"], ["Estabilidad", f"{m['stability'] * 100:.0f} %"],
                ["Inicio → Final (satisfacción)", f"{s['initial_score']} → {s['final_score']}"],
                ["Tendencia", narrative.TREND_ES[s["trend"]].capitalize()]]
        el.append(tbl(rows, [6 * cm, W - 6 * cm], header=False))
        f = s.get("factors", {})
        if f.get("positive") or f.get("negative"):
            el.append(Paragraph("Señales utilizadas para la estimación (explicabilidad)", H2))
            fr = [["Factor", "Aporte (pts)"]] + [[x["label"], f"{x['points']:+.1f}"] for x in f.get("positive", []) + f.get("negative", [])]
            el.append(tbl(fr, [W - 3 * cm, 3 * cm]))
        img = chart_emotions(data, lab)
        if img:
            el += [Paragraph("Evolución de las probabilidades emocionales (observado)", H2), Image(img, width=W, height=W * 0.34)]

    # ---- interacción ----
    inter = summ.get("interaction") or {}
    el += [PageBreak(), Paragraph("Análisis de la interacción", H1)]
    if inter.get("initial") is not None:
        el.append(tbl([["Satisfacción inicial", f"{inter['initial']:.0f}"], ["Satisfacción final", f"{inter['final']:.0f}"],
                       ["Variación", f"{inter['variation']:+.0f}"], ["Lectura", inter.get("message", "")]], [6 * cm, W - 6 * cm], header=False))
    if sp:
        el += [Paragraph("Satisfacción durante la llamada", H2), Image(chart_satisfaction(data, names), width=W, height=W * 0.37),
               Paragraph("Distribución emocional promedio", H2), Image(chart_distribution(sp, names), width=W, height=W * 0.3)]
    it = call.get("interaction") or {}
    if it:
        rows = [["Métrica", *[names.get(k, k) for k in it.get("talk_time", {})]]]
        for lbl, key, fmt in (("Tiempo de habla (s)", "talk_time", "{:.0f}"), ("Interrupciones", "interruptions", "{}"),
                              ("Palabras por minuto", "speech_rate_wpm", "{}"), ("Turnos", "turn_count", "{}")):
            rows.append([lbl] + [fmt.format(it.get(key, {}).get(k, "-")) for k in it.get("talk_time", {})])
        el += [Paragraph("Métricas de interacción (señales adicionales, no valoradas como negativas por sí mismas)", H2),
               tbl(rows, [6 * cm] + [(W - 6 * cm) / max(len(rows[0]) - 1, 1)] * (len(rows[0]) - 1)),
               Paragraph(f"Solapamientos: {it.get('overlap_events', 0)} ({it.get('overlap_seconds', 0)} s) · "
                         f"Silencios largos: {it.get('long_silences', 0)} · Pausas: {it.get('pauses', 0)}", SM)]

    # ---- momentos críticos ----
    el += [PageBreak(), Paragraph("Momentos críticos", H1)]
    if data["events"]:
        rows = [["Tiempo", "Participante", "Evento", "Emoción", "Conf.", "Satisf."]]
        for e in data["events"][:60]:
            rows.append([mmss(e["timestamp"]), names.get(e["speaker"], "Interacción"), e["label"], narrative.emo(e["emotion"]) if e["emotion"] else "-",
                         f"{e['confidence'] * 100:.0f} %", "-" if e["satisfaction"] is None else f"{e['satisfaction']:.0f}"])
        el.append(tbl(rows, [2 * cm, 3.4 * cm, 5.6 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm]))
        factors = [(e, f) for e in data["events"] for f in (e.get("evidence") or {}).get("possible_factors", [])][:12]
        if factors:
            el.append(Paragraph("Posibles factores asociados (coincidencia temporal, no causalidad)", H2))
            for e, f in factors:
                el.append(Paragraph(f"<b>{mmss(e['timestamp'])} · {e['label']}</b> — «{f['text']}». Esta expresión coincide temporalmente "
                                    f"con un cambio en las señales asociadas; no se afirma que sea la causa.", B))
    else:
        el.append(Paragraph("No se detectaron momentos críticos con los umbrales configurados.", B))

    # ---- conclusiones ----
    el += [Paragraph("Conclusiones cuantitativas", H1), Paragraph("<b>Datos observados</b> (medidos): duración, distribución de "
                                                                  "probabilidades emocionales por segmento, tiempos de habla, interrupciones y transcripción.", B),
           Paragraph("<b>Interpretación del modelo</b> (estimada): satisfacción 0-100, tendencia, eventos y su confianza.", B), Spacer(1, 6)]
    if summ.get("executive_summary"):
        el.append(Paragraph(summ["executive_summary"], NOTE))
    el += [Spacer(1, 8), Paragraph(f"Modelo: {call.get('model_version') or '-'} · Motor de satisfacción: reglas configurables. "
                                   "Los pesos y umbrales utilizados quedan registrados en el análisis.", SM)]

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(1.8 * cm, 1.0 * cm, f"Llamada {call['id'][:8]} · Estimaciones de un modelo; requieren revisión humana")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {d.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=lambda c, d: None, onLaterPages=footer)
    return buf.getvalue()
