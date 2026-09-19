"use client";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { PlayerProvider, usePlayer } from "@/lib/player";
import AudioPlayer from "@/components/call/AudioPlayer";
import { Badge, Button, Card, Empty, Field, Notice, Skeleton } from "@/components/ui";
import { EMO_ES, emoColor, mmss, SPK_COLOR } from "@/lib/format";

function Labeler({ call, datasets, labels, onLabels }: { call: any; datasets: any[]; labels: string[]; onLabels: (l: string[]) => void }) {
  const { toast, can } = useApp();
  const p = usePlayer();
  const [tr, setTr] = useState<any[]>([]);
  const [emo, setEmo] = useState<any[]>([]);
  const [ds, setDs] = useState(datasets.find((d) => !d.frozen)?.id || "");
  const [seg, setSeg] = useState({ speaker: "SPEAKER_00", start: 0, end: 5 });
  const [emotion, setEmotion] = useState("neutral");
  const [sat, setSat] = useState<number | "">("");
  const [custom, setCustom] = useState("");
  const [saved, setSaved] = useState<{ t: string }[]>([]);
  const names: Record<string, string> = Object.fromEntries(call.speakers.map((s: any) => [s.label, s.name]));

  useEffect(() => {
    api(`/calls/${call.id}/transcription`).then((r) => setTr(r.items));
    api(`/calls/${call.id}/emotions`).then((r) => setEmo(r.items));
  }, [call.id]);

  const suggestion = useMemo(() => {
    const m = emo.filter((e) => e.speaker === seg.speaker && e.end > seg.start && e.start < seg.end);
    if (!m.length) return null;
    const acc: Record<string, number> = {};
    m.forEach((e) => Object.entries<number>(e.probabilities).forEach(([k, v]) => (acc[k] = (acc[k] || 0) + v / m.length)));
    const top = Object.entries(acc).sort((a, b) => b[1] - a[1])[0];
    return { emotion: top[0], p: top[1] };
  }, [emo, seg]);

  async function save() {
    if (!ds) return toast("err", "Seleccione o cree un dataset.");
    try {
      await api(`/datasets/${ds}/samples`, { body: { call_id: call.id, speaker: seg.speaker, emotion, satisfaction: sat === "" ? null : sat, start: seg.start, end: seg.end } });
      setSaved((s) => [{ t: `${mmss(seg.start)}–${mmss(seg.end)} · ${names[seg.speaker]} · ${EMO_ES[emotion] || emotion}${sat !== "" ? ` · ${sat}/100` : ""}` }, ...s]);
      toast("ok", "Etiqueta guardada en el dataset.");
    } catch (e: any) { toast("err", e.message); }
  }
  async function enableTraining() {
    try { await api(`/calls/${call.id}`, { method: "PATCH", body: { allow_training: true } }); call.allow_training = true; toast("ok", "Llamada habilitada para entrenamiento."); setSaved((s) => [...s]); } catch (e: any) { toast("err", e.message); }
  }
  async function addLabel() {
    try { const r = await api("/labels", { body: { label: custom } }); onLabels(r.labels); setCustom(""); toast("ok", "Categoría agregada."); } catch (e: any) { toast("err", e.message); }
  }

  return (
    <div className="space-y-4">
      <AudioPlayer duration={call.duration} />
      {!call.allow_training && (
        <Notice tone="warn">Esta llamada no tiene habilitado «Permitir utilizar esta llamada para mejorar el modelo». Sin ese permiso no se pueden guardar segmentos en un dataset.
          {can("ANALYST") && <Button size="sm" className="ml-3" onClick={enableTraining}>Habilitar (queda registrado en auditoría)</Button>}</Notice>)}
      <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
        <Card title="Segmentos de la conversación" subtitle="Clic para reproducir y cargar el tramo en el formulario" pad={false}>
          <div className="max-h-[520px] space-y-1.5 overflow-auto p-3">
            {tr.map((t) => (
              <div key={t.id} onClick={() => { setSeg({ speaker: t.speaker, start: +t.start.toFixed(1), end: +t.end.toFixed(1) }); p.playRange(t.start, t.end); }}
                className={`cursor-pointer rounded-lg border px-3 py-2 text-sm hover:bg-surface2 ${seg.start === +t.start.toFixed(1) && seg.speaker === t.speaker ? "border-brand bg-brand/10" : "border-line"}`}>
                <span className="mr-2 text-[11px] tabular-nums text-muted">{mmss(t.start)}–{mmss(t.end)}</span>
                <span className="mr-2 text-[11px] font-semibold" style={{ color: SPK_COLOR[t.speaker === "SPEAKER_00" ? 0 : 1] }}>{names[t.speaker]}</span>{t.text}
              </div>))}
          </div>
        </Card>
        <Card title="Etiquetar segmento" subtitle="Etiquete lo que escucha; la sugerencia del modelo es solo una referencia">
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2">
              <Field label="Speaker"><select className="w-full" value={seg.speaker} onChange={(e) => setSeg({ ...seg, speaker: e.target.value })}>{call.speakers.map((s: any) => <option key={s.label} value={s.label}>{s.name}</option>)}</select></Field>
              <Field label="Inicio (s)"><input type="number" step="0.1" className="w-full" value={seg.start} onChange={(e) => setSeg({ ...seg, start: +e.target.value })} /></Field>
              <Field label="Fin (s)"><input type="number" step="0.1" className="w-full" value={seg.end} onChange={(e) => setSeg({ ...seg, end: +e.target.value })} /></Field>
            </div>
            <Button size="sm" onClick={() => p.playRange(seg.start, seg.end)}>▶ Reproducir segmento</Button>
            {suggestion && <p className="text-xs text-muted">Sugerencia del modelo (no vinculante): <b style={{ color: emoColor(suggestion.emotion) }}>{EMO_ES[suggestion.emotion] || suggestion.emotion}</b> ({Math.round(suggestion.p * 100)}%)</p>}
            <div>
              <p className="mb-1.5 text-xs font-medium text-muted">Emoción</p>
              <div className="flex flex-wrap gap-1.5">
                {labels.map((l) => <button key={l} onClick={() => setEmotion(l)} className={`rounded-full border px-3 py-1 text-xs font-medium transition ${emotion === l ? "border-transparent text-white" : "border-line hover:bg-surface2"}`} style={emotion === l ? { background: emoColor(l) } : undefined}>{EMO_ES[l] || l}</button>)}
              </div>
              <div className="mt-2 flex gap-2"><input className="flex-1 text-xs" placeholder="Nueva categoría personalizada" value={custom} onChange={(e) => setCustom(e.target.value)} /><Button size="sm" disabled={!custom} onClick={addLabel}>Agregar</Button></div>
            </div>
            <Field label={`Satisfacción del participante (0-100)${sat === "" ? " — opcional" : `: ${sat}`}`}>
              <div className="flex items-center gap-2"><input type="range" min={0} max={100} value={sat === "" ? 50 : sat} onChange={(e) => setSat(+e.target.value)} className="flex-1" /><Button size="sm" variant="ghost" onClick={() => setSat("")}>Sin dato</Button></div>
            </Field>
            <Field label="Dataset destino">
              <select className="w-full" value={ds} onChange={(e) => setDs(e.target.value)}>{datasets.filter((d) => !d.frozen).map((d) => <option key={d.id} value={d.id}>{d.code}</option>)}</select>
            </Field>
            <Button variant="primary" className="w-full" onClick={save} disabled={!can("ANALYST") || !call.allow_training}>Guardar etiqueta</Button>
            {saved.length > 0 && <div className="border-t border-line pt-2 text-xs text-muted"><p className="mb-1 font-medium">Guardadas en esta sesión</p>{saved.slice(0, 6).map((s, i) => <p key={i}>✓ {s.t}</p>)}</div>}
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function Labeling() {
  const [calls, setCalls] = useState<any[] | null>(null);
  const [sel, setSel] = useState("");
  const [call, setCall] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [labels, setLabels] = useState<string[]>([]);

  useEffect(() => {
    api("/calls", { query: { status: "COMPLETED", page_size: 100 } }).then((r) => { setCalls(r.items); if (r.items[0]) setSel(r.items[0].id); });
    api("/datasets").then(setDatasets);
    api("/labels").then((r) => setLabels(r.labels));
  }, []);
  useEffect(() => { if (sel) { setCall(null); api(`/calls/${sel}`).then(setCall); } }, [sel]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="text-2xl font-semibold">Etiquetado</h1><p className="text-sm text-muted">Cree datasets propios etiquetando segmentos de llamadas analizadas</p></div>
        <select value={sel} onChange={(e) => setSel(e.target.value)} aria-label="Llamada">{calls?.map((c) => <option key={c.id} value={c.id}>{c.display_name || c.filename} · {c.id.slice(0, 6)}</option>)}</select>
      </div>
      {calls && calls.length === 0 && <Card><Empty title="No hay llamadas analizadas" text="Analice una llamada primero para poder etiquetar sus segmentos." /></Card>}
      {datasets.length === 0 && <Notice tone="info">Aún no tiene datasets. Cree uno en la sección «Dataset».</Notice>}
      {sel && !call && <Skeleton className="h-96" />}
      {call && <PlayerProvider key={call.id} src={`/api/v1/calls/${call.id}/audio`}><Labeler call={call} datasets={datasets} labels={labels} onLabels={setLabels} /></PlayerProvider>}
    </div>
  );
}
