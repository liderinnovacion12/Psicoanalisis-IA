"use client";
import { use, useCallback, useEffect, useMemo, useState } from "react";
import { api, downloadUrl, audioUrl } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { PlayerProvider } from "@/lib/player";
import { Badge, Button, Card, Notice, Progress, Skeleton, Stat, StatusBadge, Tabs } from "@/components/ui";
import AudioPlayer from "@/components/call/AudioPlayer";
import { EmotionChart, SatisfactionChart, Timeline } from "@/components/call/Charts";
import { EventsTable, SpeakerCard, TranscriptView } from "@/components/call/Panels";
import { dateTime, isProcessing, mmss, num, shortId, STATUS_ES, TREND_ES } from "@/lib/format";

export default function CallDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { toast, can } = useApp();
  const [call, setCall] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);
  const [data, setData] = useState<{ emotions: any[]; sat: any; events: any[]; tr: any[] } | null>(null);
  const [redact, setRedact] = useState(false);
  const [tab, setTab] = useState<"transcript" | "events" | "quality">("transcript");
  const [err, setErr] = useState("");

  const loadCall = useCallback(async () => {
    try { setCall(await api(`/calls/${id}`)); } catch (e: any) { setErr(e.message); }
  }, [id]);

  const loadResults = useCallback(async () => {
    const [emo, sat, events, tr] = await Promise.all([
      api(`/calls/${id}/emotions`), api(`/calls/${id}/satisfaction`), api(`/calls/${id}/events`), api(`/calls/${id}/transcription`, { query: { redact } }),
    ]);
    setData({ emotions: emo.items, sat, events, tr: tr.items });
  }, [id, redact]);

  useEffect(() => { loadCall(); }, [loadCall]);

  // Sondeo del estado mientras se procesa
  useEffect(() => {
    if (!call || !isProcessing(call.status)) return;
    const t = setInterval(async () => {
      try {
        const s = await api(`/calls/${id}/status`);
        setStatus(s);
        if (!isProcessing(s.status)) loadCall();
      } catch {}
    }, 2000);
    return () => clearInterval(t);
  }, [call, id, loadCall]);

  useEffect(() => { if (call?.status === "COMPLETED") loadResults().catch((e) => setErr(e.message)); }, [call?.status, loadResults]);

  const names = useMemo<Record<string, string>>(() => Object.fromEntries((call?.speakers || []).map((s: any) => [s.label, s.name])), [call]);

  if (err && !call) return <Notice tone="bad">{err}</Notice>;
  if (!call) return <div className="space-y-4"><Skeleton className="h-16" /><Skeleton className="h-64" /><Skeleton className="h-96" /></div>;

  const processing = isProcessing(call.status);
  const stages = status?.stages || call.stages;

  async function setRole(sp: any, role: string) {
    try { await api(`/calls/${id}/speakers/${sp.id}`, { method: "PATCH", body: { role } }); toast("ok", "Rol actualizado."); loadCall(); } catch (e: any) { toast("err", e.message); }
  }
  async function resume() {
    try { await api(`/calls/${id}/resume`, { method: "POST" }); toast("ok", "Procesamiento reanudado desde el último checkpoint."); loadCall(); } catch (e: any) { toast("err", e.message); }
  }
  async function reanalyze() {
    try { await api(`/calls/${id}/reanalyze`, { method: "POST", query: { from_stage: "emotion_analysis" } }); toast("ok", "Re-análisis con el modelo activo en cola."); loadCall(); } catch (e: any) { toast("err", e.message); }
  }
  async function toggleTrain(v: boolean) {
    try { await api(`/calls/${id}`, { method: "PATCH", body: { allow_training: v } }); loadCall(); toast("ok", v ? "La llamada podrá usarse para mejorar el modelo." : "La llamada NO se usará para entrenamiento."); } catch (e: any) { toast("err", e.message); }
  }

  const Header = (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2"><h1 className="text-xl font-semibold">{call.display_name || call.filename}</h1><StatusBadge status={call.status} /></div>
        <p className="mt-1 text-xs text-muted">ID {shortId(call.id)} · {dateTime(call.created_at)} · Duración {mmss(call.duration)} · Idioma {call.language?.toUpperCase() || "—"} · Modelo {call.model_version || "—"}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        {call.status === "COMPLETED" && (["pdf", "csv", "xlsx", "json"] as const).map((f) => (
          <a key={f} href={downloadUrl(`/calls/${id}/export`, { format: f, ...(redact ? { redact: "true" } : {}) })}><Button size="sm">{f.toUpperCase()}</Button></a>))}
        {call.status === "COMPLETED" && can("ANALYST") && <Button size="sm" onClick={reanalyze} title="Vuelve a analizar emociones con el modelo actualmente activo">Re-analizar</Button>}
      </div>
    </div>
  );

  if (processing || call.status === "ERROR") {
    return (
      <div className="space-y-5">
        {Header}
        <Card title="Procesamiento" subtitle="Puede cerrar esta página: el análisis continúa en segundo plano y se reanuda desde el último checkpoint si falla.">
          <div className="mb-4 flex items-center gap-3"><Progress value={status?.progress ?? call.progress} className="flex-1" /><span className="w-12 text-right text-sm tabular-nums">{Math.round(status?.progress ?? call.progress)}%</span></div>
          <ul className="space-y-2.5">
            {stages.map((s: any) => (
              <li key={s.key} className="flex items-center gap-3 text-sm">
                <span className="w-5 text-center">{s.state === "COMPLETED" ? "✓" : s.state === "FAILED" ? "✗" : s.state === "RUNNING" ? "…" : "·"}</span>
                <span className="w-32 font-medium">{s.label}</span>
                {s.state === "RUNNING" ? (<><Progress value={s.progress} className="max-w-xs flex-1" /><span className="text-xs tabular-nums text-muted">{Math.round(s.progress)}% {s.message}</span></>)
                  : <span className={s.state === "FAILED" ? "text-bad" : "text-muted"}>{s.state === "COMPLETED" ? `Completado${s.seconds ? ` (${s.seconds}s)` : ""}` : s.state === "FAILED" ? "Falló" : "Pendiente"}</span>}
              </li>))}
          </ul>
          <p className="mt-4 text-xs text-muted">Estado: {STATUS_ES[call.status]}</p>
          {call.status === "ERROR" && (<div className="mt-4 space-y-3"><Notice tone="bad">{call.error || "No fue posible completar el análisis."}</Notice>{can("ANALYST") && <Button variant="primary" onClick={resume}>Reanudar desde el último checkpoint</Button>}</div>)}
        </Card>
      </div>
    );
  }

  const sp = call.summary?.speakers || {};
  const inter = call.summary?.interaction;
  const it = call.interaction;
  const q = call.analysis_quality_detail;
  const aq = call.audio_quality_detail;
  const s0 = call.speakers.find((s: any) => s.label === "SPEAKER_00");
  const s1 = call.speakers.find((s: any) => s.label === "SPEAKER_01");

  return (
    <PlayerProvider src={audioUrl(id)}>
      <div className="space-y-5">
        {Header}
        <div className="sticky top-[57px] z-20"><AudioPlayer duration={call.duration || 0} /></div>

        {call.warnings?.length > 0 && <Notice tone="warn"><p className="mb-1 font-medium">Advertencias del análisis</p><ul className="list-disc pl-5 text-xs">{call.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul></Notice>}

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <Stat label="Calidad del análisis" value={`${num(q?.score)}/100`} hint={q?.low ? "Puede haber menor precisión" : "Estimación heurística"} tone={q?.low ? "text-warn" : undefined} />
          <Stat label="Calidad del audio" value={`${num(aq?.score)}/100`} hint={`SNR ~${num(aq?.levels?.snr_db)} dB · voz ${Math.round((aq?.speech_ratio || 0) * 100)}%`} />
          <Stat label="Diarización" value={`${num(call.checkpoints?.diarization?.quality)}/100`} hint={`Motor: ${call.diarization_mode}`} />
          <Stat label="Transcripción" value={`${num((call.checkpoints?.transcription?.mean_confidence || 0) * 100)}/100`} hint={`${call.checkpoints?.transcription?.words ?? 0} palabras`} />
          <Stat label="Interacción" value={inter?.variation != null ? `${inter.variation > 0 ? "+" : ""}${inter.variation.toFixed(0)}` : "—"} hint={inter ? `${inter.initial} → ${inter.final}` : undefined} tone={inter?.variation > 6 ? "text-good" : inter?.variation < -6 ? "text-bad" : undefined} />
        </div>

        {call.diarization_mode === "spectral" && <Notice tone="warn">La diarización utilizada es el <b>método de respaldo</b> (sin pyannote), de menor precisión. Configure <code>HF_TOKEN</code> para usar pyannote.audio, o use audio estéreo con una persona por canal.</Notice>}

        {call.summary?.executive_summary && (
          <Card title="Resumen" subtitle="Separación entre datos observados e interpretación del modelo">
            <p className="text-sm leading-relaxed">{call.summary.executive_summary}</p>
            {inter && <p className="mt-2 text-sm font-medium text-brand">{inter.message}</p>}
          </Card>)}

        <div className="grid gap-5 xl:grid-cols-2">
          <SpeakerCard label="SPEAKER_00" name={names.SPEAKER_00} idx={0} s={sp.SPEAKER_00} onRole={can("ANALYST") && s0 ? (r) => setRole(s0, r) : undefined} roles={{ key: "r0", label: "", cur: s0?.role || "other" }} />
          <SpeakerCard label="SPEAKER_01" name={names.SPEAKER_01} idx={1} s={sp.SPEAKER_01} onRole={can("ANALYST") && s1 ? (r) => setRole(s1, r) : undefined} roles={{ key: "r1", label: "", cur: s1?.role || "other" }} />
        </div>

        {data ? (<>
          <Card title="Línea de tiempo" subtitle="Emoción dominante por segmento · clic para reproducir">
            <Timeline emotions={data.emotions} events={data.events} names={names} duration={call.duration} />
          </Card>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Emociones durante la llamada" subtitle="Probabilidades completas del modelo por ventana (dato observado)"><EmotionChart emotions={data.emotions} names={names} duration={call.duration} /></Card>
            <Card title="Satisfacción durante la llamada" subtitle="Estimación del Satisfaction Engine (interpretación del modelo)"><SatisfactionChart sat={data.sat} events={data.events} names={names} duration={call.duration} /></Card>
          </div>
        </>) : <Skeleton className="h-80" />}

        <Card pad={false}>
          <div className="px-5 pt-3"><Tabs tabs={[{ key: "transcript", label: "Transcripción" }, { key: "events", label: `Eventos (${data?.events.length ?? 0})` }, { key: "quality", label: "Detalle y privacidad" }]} value={tab} onChange={setTab} /></div>
          <div className="p-5">
            {tab === "transcript" && (data ? (<>
              <label className="mb-3 flex items-center gap-2 text-xs text-muted"><input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} />Ocultar datos personales (nombres, teléfonos, correos, documentos…)</label>
              <TranscriptView items={data.tr} names={names} redact={redact} />
            </>) : <Skeleton className="h-64" />)}
            {tab === "events" && (data ? <EventsTable events={data.events} names={names} /> : <Skeleton className="h-64" />)}
            {tab === "quality" && (
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="space-y-3 text-sm">
                  <h4 className="font-semibold">Interacción</h4>
                  {it && <table className="w-full text-sm"><tbody>
                    {[["Tiempo de habla", ...Object.values<number>(it.talk_time || {}).map((v) => `${v.toFixed(0)} s`)],
                      ["Interrupciones", ...Object.values<number>(it.interruptions || {}).map(String)],
                      ["Palabras/min", ...Object.values<number>(it.speech_rate_wpm || {}).map(String)]].map((r: any) => (
                      <tr key={r[0]} className="border-b border-line/60"><td className="py-1.5 text-muted">{r[0]}</td>{r.slice(1).map((v: string, i: number) => <td key={i} className="py-1.5 font-medium">{v}</td>)}</tr>))}
                  </tbody></table>}
                  {it && <p className="text-xs text-muted">Solapamientos: {it.overlap_events} ({it.overlap_seconds}s) · Silencios significativos: {it.long_silences} · Pausas: {it.pauses}. {it.note}</p>}
                  <h4 className="pt-2 font-semibold">Calidad del análisis por componente</h4>
                  {q?.components && Object.entries<number>(q.components).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-3 text-xs"><span className="w-28 capitalize text-muted">{k === "model" ? "modelo emocional*" : k}</span><Progress value={v} className="flex-1" /><span className="w-8 text-right tabular-nums">{v.toFixed(0)}</span></div>))}
                  <p className="text-[11px] text-muted">* Estimada a partir de la confianza del modelo; no equivale a su exactitud real.</p>
                </div>
                <div className="space-y-3 text-sm">
                  <h4 className="font-semibold">Privacidad y entrenamiento</h4>
                  <label className="flex items-start gap-2"><input type="checkbox" checked={call.allow_training} disabled={!can("ANALYST")} onChange={(e) => toggleTrain(e.target.checked)} className="mt-0.5" />
                    <span>Permitir utilizar esta llamada para mejorar el modelo<br /><span className="text-xs text-muted">Desactivado por defecto. Solo las llamadas habilitadas pueden agregarse a datasets de entrenamiento.</span></span></label>
                  <h4 className="pt-2 font-semibold">Datos técnicos</h4>
                  <p className="text-xs text-muted">Audio original: {call.sample_rate} Hz · {call.channels} canal(es) · {Math.round((call.bitrate || 0) / 1000)} kbps · Modo: {aq?.channel_layout?.is_stereo_split ? "estéreo con un hablante por canal" : "mono / mezcla (diarización)"} · VAD: {aq?.vad_engine}</p>
                  {call.retention_until && <p className="text-xs text-muted">Se eliminará automáticamente el {dateTime(call.retention_until)} (política de retención).</p>}
                </div>
              </div>)}
          </div>
        </Card>
        <p className="pb-4 text-center text-[11px] text-muted">Las estimaciones provienen de modelos de aprendizaje automático y no constituyen afirmaciones definitivas sobre las personas; revise los momentos críticos con criterio humano.</p>
      </div>
    </PlayerProvider>
  );
}
