"use client";
import { useEffect, useMemo, useRef } from "react";
import { usePlayer } from "@/lib/player";
import { Badge, Card, Progress } from "../ui";
import { confWord, emoLabel, mmss, pct, satColor, SPK_COLOR, TREND_ES } from "@/lib/format";

export function SpeakerCard({ label, name, idx, s, onRole, roles }: { label: string; name: string; idx: number; s: any; onRole?: (role: string) => void; roles?: { key: string; label: string; cur: string } }) {
  if (!s) return <Card title={name}><p className="text-sm text-muted">Sin datos suficientes para este participante.</p></Card>;
  const m = s.metrics;
  const rows: [string, string][] = [
    ["Emoción predominante", emoLabel(m.dominant_emotion)], ["Emoción inicial", emoLabel(m.initial_emotion)], ["Emoción final", emoLabel(m.final_emotion)],
    ["Frustración", pct(m.frustration)], ["Tensión", pct(m.tension)], ["Emociones positivas", pct(m.positive)], ["Emociones negativas", pct(m.negative)],
    ["Estabilidad", pct(m.stability)], ["Confianza del análisis", `${pct(s.confidence)} · ${confWord(s.confidence)}`],
  ];
  return (
    <Card title={<span className="flex items-center gap-2"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: SPK_COLOR[idx] }} />{name.toUpperCase()}</span>}
      actions={onRole && roles && (
        <select value={roles.cur} onChange={(e) => onRole(e.target.value)} className="py-1 text-xs" aria-label="Rol">
          {["client", "agent", "user", "advisor", "other"].map((r) => <option key={r} value={r}>{{ client: "Cliente", agent: "Agente", user: "Usuario", advisor: "Asesor", other: "Sin rol" }[r]}</option>)}
        </select>)}>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs text-muted">Satisfacción estimada</p>
          <p className={`text-4xl font-semibold tabular-nums ${satColor(s.score)}`}>{s.score.toFixed(0)}<span className="text-lg text-muted">/100</span></p>
          <p className="text-xs text-muted">{s.interpretation.label}</p>
        </div>
        <div className="text-right text-xs">
          <Badge tone={s.trend === "improving" ? "good" : s.trend === "declining" ? "bad" : "neutral"}>Tendencia {TREND_ES[s.trend].toLowerCase()}</Badge>
          <p className="mt-2 text-muted">Inicio → Final</p>
          <p className="text-sm font-semibold tabular-nums">{s.initial_score ?? "—"} → {s.final_score ?? "—"}</p>
        </div>
      </div>
      <Progress value={s.score} className="mt-3" tone={s.score >= 61 ? "good" : s.score < 41 ? "bad" : "brand"} />
      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        {rows.map(([k, v]) => (<div key={k} className="flex flex-col"><dt className="text-[11px] text-muted">{k}</dt><dd className="font-medium">{v}</dd></div>))}
      </dl>
      <div className="mt-4 grid gap-3 border-t border-line pt-3 text-xs sm:grid-cols-2">
        <div><p className="mb-1 font-semibold text-good">Factores que suman</p>{s.factors.positive.length ? s.factors.positive.map((f: any) => <p key={f.key} className="text-muted">+ {f.label} <b className="text-ink">({f.points > 0 ? "+" : ""}{f.points})</b></p>) : <p className="text-muted">—</p>}</div>
        <div><p className="mb-1 font-semibold text-bad">Factores que restan</p>{s.factors.negative.length ? s.factors.negative.map((f: any) => <p key={f.key} className="text-muted">− {f.label} <b className="text-ink">({f.points})</b></p>) : <p className="text-muted">—</p>}</div>
      </div>
      <p className="mt-3 text-[11px] text-muted">Interpretación del modelo: «El modelo estima un nivel de satisfacción de {s.score.toFixed(0)}/100 con confianza {confWord(s.confidence).toLowerCase()}». La confianza del análisis es independiente de la probabilidad de cada emoción.</p>
    </Card>
  );
}

export function TranscriptView({ items, names, redact }: { items: any[]; names: Record<string, string>; redact: boolean }) {
  const p = usePlayer();
  const active = useMemo(() => items.findIndex((t) => p.time >= t.start && p.time < t.end + 0.25), [items, p.time]);
  const refs = useRef<Record<number, HTMLDivElement | null>>({});
  useEffect(() => { if (active >= 0 && p.playing) refs.current[active]?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [active, p.playing]);
  if (!items.length) return <p className="py-6 text-center text-sm text-muted">No hay transcripción disponible.</p>;
  return (
    <div className="max-h-[520px] space-y-2 overflow-auto pr-1">
      {items.map((t, i) => {
        const idx = t.speaker === "SPEAKER_00" ? 0 : 1;
        const sent = t.sentiment;
        return (
          <div key={t.id} ref={(el) => { refs.current[i] = el; }} onClick={() => p.seek(t.start)} role="button" tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && p.seek(t.start)}
            className={`cursor-pointer rounded-lg border px-3 py-2 transition ${i === active ? "border-brand bg-brand/10" : "border-line hover:bg-surface2"}`}>
            <div className="mb-0.5 flex items-center gap-2 text-[11px]">
              <span className="tabular-nums text-muted">[{mmss(t.start)}]</span>
              <span className="font-semibold" style={{ color: SPK_COLOR[idx] }}>{names[t.speaker] || t.speaker}</span>
              <span className="text-muted">conf. {Math.round((t.confidence ?? 0) * 100)}%</span>
              {sent?.cues?.some((c: any) => c.type === "frustration") && <Badge tone="warn">señal de queja</Badge>}
              {t.pii_count > 0 && !redact && <Badge tone="neutral">PII {t.pii_count}</Badge>}
            </div>
            <p className="text-sm leading-snug">“{t.text}”</p>
          </div>
        );
      })}
    </div>
  );
}

export function EventsTable({ events, names }: { events: any[]; names: Record<string, string> }) {
  const p = usePlayer();
  if (!events.length) return <p className="py-6 text-center text-sm text-muted">No se detectaron momentos críticos con los umbrales actuales.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead><tr className="border-b border-line text-left text-xs text-muted">
          {["Timestamp", "Speaker", "Evento", "Emoción", "Confianza", "Satisfacción", "Audio"].map((h) => <th key={h} className="px-2 py-2 font-medium">{h}</th>)}
        </tr></thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.id} className="border-b border-line/60 align-top hover:bg-surface2/60">
              <td className="px-2 py-2 tabular-nums">{mmss(e.timestamp)}</td>
              <td className="px-2 py-2">{e.speaker ? names[e.speaker] : "Interacción"}</td>
              <td className="px-2 py-2"><p className="font-medium">{e.label}</p><p className="max-w-md text-xs text-muted">{e.description}</p>
                {e.evidence?.possible_factors?.map((f: any, i: number) => (<p key={i} className="mt-1 text-xs"><span className="text-warn">Posible factor asociado:</span> «{f.text}» <span className="text-muted">(coincidencia temporal, no causalidad)</span></p>))}
              </td>
              <td className="px-2 py-2">{emoLabel(e.emotion)}</td>
              <td className="px-2 py-2 tabular-nums">{Math.round(e.confidence * 100)}%</td>
              <td className="px-2 py-2 tabular-nums">{e.satisfaction != null ? `${Math.round(e.satisfaction)}/100` : "—"}</td>
              <td className="px-2 py-2"><button className="rounded-md border border-line px-2 py-1 hover:bg-brand/10" onClick={() => e.end_timestamp ? p.playRange(Math.max(0, e.timestamp - 2), Math.min(e.end_timestamp, e.timestamp + 15)) : p.playRange(Math.max(0, e.timestamp - 3), e.timestamp + 8)} aria-label="Reproducir segmento">▶</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
