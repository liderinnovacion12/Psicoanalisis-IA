"use client";
// Pantalla principal simplificada: subir la llamada → se analiza → resultado.
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api, audioUrl, downloadUrl, uploadWithProgress } from "@/lib/api";
import { DEMO } from "@/lib/demo";
import { useApp } from "@/lib/app-context";
import { PlayerProvider } from "@/lib/player";
import AudioPlayer from "@/components/call/AudioPlayer";
import { SatisfactionChart } from "@/components/call/Charts";
import { EventsTable } from "@/components/call/Panels";
import { Badge, Button, Card, Notice, Progress, Skeleton } from "@/components/ui";
import { bytes, confWord, emoLabel, isProcessing, mmss, pct, satColor, SPK_COLOR, TREND_ES } from "@/lib/format";

const EXT = ["mp3", "wav", "m4a", "aac", "flac", "ogg"];
type Phase = "idle" | "uploading" | "processing" | "result" | "error";

function Result({ id, onReset }: { id: string; onReset: () => void }) {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    (async () => {
      const [call, sat, events] = await Promise.all([api(`/calls/${id}`), api(`/calls/${id}/satisfaction`), api(`/calls/${id}/events`)]);
      setD({ call, sat, events });
    })().catch(() => setD({ error: true }));
  }, [id]);
  if (!d) return <div className="space-y-4"><Skeleton className="h-40" /><Skeleton className="h-72" /></div>;
  if (d.error) return <Notice tone="bad">No se pudo cargar el resultado.</Notice>;
  const { call, sat, events } = d;
  const names: Record<string, string> = Object.fromEntries(call.speakers.map((s: any) => [s.label, s.name]));
  const sp = call.summary?.speakers || {};
  const inter = call.summary?.interaction;
  return (
    <PlayerProvider src={audioUrl(id)}>
      <div className="space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h2 className="text-xl font-semibold">Resultado del análisis</h2><p className="text-xs text-muted">{call.display_name || call.filename} · {mmss(call.duration)} · idioma {call.language?.toUpperCase() || "—"}</p></div>
          <div className="flex flex-wrap gap-2">
            <a href={downloadUrl(`/reports/${id}`)}><Button variant="primary">Descargar PDF</Button></a>
            <Link href={`/calls/${id}`}><Button>Ver análisis completo</Button></Link>
            <Button variant="ghost" onClick={onReset}>Analizar otra llamada</Button>
          </div>
        </div>
        {call.warnings?.length > 0 && <Notice tone="warn"><ul className="list-disc pl-5 text-xs">{call.warnings.slice(0, 3).map((w: string, i: number) => <li key={i}>{w}</li>)}</ul></Notice>}

        <div className="grid gap-4 md:grid-cols-2">
          {(["SPEAKER_00", "SPEAKER_01"] as const).map((lab, i) => {
            const s = sp[lab];
            return (
              <Card key={lab}>
                <p className="mb-2 flex items-center gap-2 text-sm font-semibold"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: SPK_COLOR[i] }} />{names[lab] || lab}</p>
                {!s ? <p className="text-sm text-muted">Sin datos suficientes.</p> : (<>
                  <p className="text-xs text-muted">Satisfacción estimada</p>
                  <p className={`text-5xl font-semibold tabular-nums ${satColor(s.score)}`}>{s.score.toFixed(0)}<span className="text-xl text-muted">/100</span></p>
                  <p className="text-xs text-muted">{s.interpretation.label} · confianza {confWord(s.confidence).toLowerCase()}</p>
                  <Progress value={s.score} className="mt-3" tone={s.score >= 61 ? "good" : s.score < 41 ? "bad" : "brand"} />
                  <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
                    <div><p className="text-[11px] text-muted">Predominante</p><p className="font-medium">{emoLabel(s.metrics.dominant_emotion)}</p></div>
                    <div><p className="text-[11px] text-muted">Inicio → Final</p><p className="font-medium">{emoLabel(s.metrics.initial_emotion)} → {emoLabel(s.metrics.final_emotion)}</p></div>
                    <div><p className="text-[11px] text-muted">Frustración</p><p className="font-medium">{pct(s.metrics.frustration)}</p></div>
                  </div>
                  <div className="mt-3"><Badge tone={s.trend === "improving" ? "good" : s.trend === "declining" ? "bad" : "neutral"}>Tendencia {TREND_ES[s.trend].toLowerCase()}</Badge></div>
                </>)}
              </Card>);
          })}
        </div>

        {call.summary?.executive_summary && (
          <Card title="Resumen">
            <p className="text-sm leading-relaxed">{call.summary.executive_summary}</p>
            {inter && <p className="mt-2 text-sm font-medium text-brand">{inter.message}</p>}
          </Card>)}

        <AudioPlayer duration={call.duration || 0} />
        <Card title="Satisfacción durante la llamada" subtitle="Estimación del modelo; haga clic en la gráfica para escuchar ese momento">
          <SatisfactionChart sat={sat} events={events} names={names} duration={call.duration} />
        </Card>
        <Card title={`Momentos destacados (${events.length})`}><EventsTable events={events.slice(0, 8)} names={names} /></Card>
        <p className="pb-4 text-center text-[11px] text-muted">Estimaciones de modelos de aprendizaje automático; no son afirmaciones definitivas. Revise los momentos destacados con criterio humano.</p>
      </div>
    </PlayerProvider>
  );
}

export default function Analizar() {
  const { toast, can } = useApp();
  const input = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [pctUp, setPctUp] = useState(0);
  const [callId, setCallId] = useState("");
  const [status, setStatus] = useState<any>(null);
  const [err, setErr] = useState("");
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);

  const reset = () => { setPhase("idle"); setFile(null); setCallId(""); setStatus(null); setErr(""); setPctUp(0); if (typeof window !== "undefined") window.history.replaceState(null, "", "/"); };

  // recuperar un análisis en curso o terminado (?call=ID)
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("call");
    if (id) { setCallId(id); setPhase("processing"); }
  }, []);

  // sondeo del estado
  useEffect(() => {
    if (phase !== "processing" || !callId) return;
    let stop = false;
    const tick = async () => {
      try {
        const s = await api(`/calls/${callId}/status`);
        if (stop) return;
        setStatus(s);
        if (s.status === "COMPLETED") setPhase("result");
        else if (s.status === "ERROR") { setErr(s.error || "No fue posible completar el análisis."); setPhase("error"); }
      } catch (e: any) { if (!stop) { setErr(e.message); setPhase("error"); } }
    };
    tick();
    const t = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(t); };
  }, [phase, callId]);

  function pick(f?: File | null) {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase() || "";
    if (!EXT.includes(ext)) { setErr("Formato no soportado. Use MP3, WAV, M4A, AAC, FLAC u OGG."); return; }
    setErr(""); setFile(f);
  }
  async function analyze() {
    if (!file) return;
    setPhase("uploading"); setErr(""); setPctUp(0);
    const fd = new FormData();
    fd.append("file", file); fd.append("allow_training", "false");
    try {
      const r = await uploadWithProgress("/calls/upload", fd, setPctUp);
      setCallId(r.id); setPhase("processing");
      window.history.replaceState(null, "", `/?call=${r.id}`);
    } catch (e: any) { setErr(e.message); setPhase("idle"); }
  }
  async function loadExample() {
    try { const l = await api("/calls", { query: { status: "COMPLETED" } }); setCallId(l.items[0].id); setPhase("result"); } catch (e: any) { toast("err", e.message); }
  }

  if (phase === "result") return <Result id={callId} onReset={reset} />;

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div className="text-center">
        <h1 className="text-2xl font-semibold">Analiza una llamada</h1>
        <p className="mt-1 text-sm text-muted">Sube la grabación y obtén la satisfacción y las emociones de cada persona.</p>
      </div>

      {(phase === "idle" || phase === "uploading") && (<>
        <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
          className={`flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition ${drag ? "border-brand bg-brand/10" : "border-line bg-surface"}`}>
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-brand/15 text-2xl text-brand">⇪</div>
          {file ? (<><p className="font-medium">{file.name}</p><p className="text-xs text-muted">{bytes(file.size)}</p></>)
            : (<><p className="font-medium">Arrastra tu grabación aquí</p><p className="text-xs text-muted">MP3 • WAV • M4A • AAC • FLAC • OGG · llamadas cortas o de horas</p></>)}
          <div className="flex gap-2">
            <Button onClick={() => input.current?.click()} disabled={phase === "uploading"}>{file ? "Cambiar archivo" : "Elegir archivo"}</Button>
            {file && <Button variant="primary" onClick={analyze} disabled={phase === "uploading" || !can("ANALYST")}>{phase === "uploading" ? "Subiendo…" : "Analizar"}</Button>}
          </div>
          <input ref={input} type="file" hidden accept=".mp3,.wav,.m4a,.aac,.flac,.ogg,audio/*" onChange={(e) => pick(e.target.files?.[0])} />
        </div>
        {phase === "uploading" && <div><Progress value={pctUp} /><p className="mt-1 text-center text-xs text-muted">Subiendo… {Math.round(pctUp)}%</p></div>}
        {err && <Notice tone="bad">{err}</Notice>}
        {DEMO && (
          <Notice tone="info">
            <p className="text-sm"><b>Modo demostración (sin servidor):</b> aquí no se puede analizar audio nuevo. Puede ver el resultado real de una llamada de ejemplo <b>sintética</b>:</p>
            <Button className="mt-2" variant="primary" onClick={loadExample}>Ver resultado de ejemplo</Button>
          </Notice>)}
        <p className="text-center text-xs text-muted">Tu grabación se procesa en tu propio servidor y no se usa para entrenar modelos.</p>
      </>)}

      {phase === "processing" && (
        <Card title="Analizando la llamada…" subtitle="Puedes cerrar la página: el análisis continúa y podrás volver desde el Historial.">
          <div className="mb-4 flex items-center gap-3"><Progress value={status?.progress ?? 0} className="flex-1" /><span className="w-12 text-right text-sm tabular-nums">{Math.round(status?.progress ?? 0)}%</span></div>
          <ul className="space-y-2">
            {(status?.stages || []).map((s: any) => (
              <li key={s.key} className="flex items-center gap-3 text-sm">
                <span className="w-5 text-center">{s.state === "COMPLETED" ? "✓" : s.state === "FAILED" ? "✗" : s.state === "RUNNING" ? "…" : "·"}</span>
                <span className="w-32 font-medium">{s.label}</span>
                <span className="text-xs text-muted">{s.state === "RUNNING" ? `en curso${s.started_at ? ` · ${Math.max(0, Math.round((now - new Date(s.started_at).getTime()) / 1000))} s` : ""}${s.progress > 0 ? ` · ${Math.round(s.progress)}%` : ""}` : s.state === "COMPLETED" ? "Completado" : s.state === "FAILED" ? "Falló" : "Pendiente"}</span>
              </li>))}
          </ul>
        </Card>)}

      {phase === "error" && (
        <div className="space-y-3"><Notice tone="bad">{err}</Notice>
          <div className="flex gap-2"><Button variant="primary" onClick={reset}>Intentar de nuevo</Button>{callId && <Link href={`/calls/${callId}`}><Button>Ver detalle</Button></Link>}</div></div>)}
    </div>
  );
}
