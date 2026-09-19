"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, downloadUrl } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { DEMO } from "@/lib/demo";
import { Badge, Button, Card, Empty, Field, Modal, Notice, Pagination, Skeleton, Stat } from "@/components/ui";
import { EMO_ES, emoColor, mmss } from "@/lib/format";

function Bars({ data, colorOf }: { data: Record<string, number>; colorOf?: (k: string) => string }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...Object.values(data), 1);
  return (
    <div className="space-y-1.5">
      {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="w-20 truncate text-muted">{EMO_ES[k] || k}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-surface2"><div className="h-full rounded" style={{ width: `${(v / max) * 100}%`, background: colorOf?.(k) || "rgb(var(--brand))" }} /></div>
          <span className="w-20 text-right tabular-nums">{v} ({((v / total) * 100).toFixed(0)}%)</span>
        </div>))}
    </div>
  );
}

export default function DatasetPage() {
  const { toast, can } = useApp();
  const [list, setList] = useState<any[] | null>(null);
  const [sel, setSel] = useState<string>("");
  const [ds, setDs] = useState<any>(null);
  const [samples, setSamples] = useState<any>(null);
  const [page, setPage] = useState(1);
  const [onlyIssues, setOnlyIssues] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [nf, setNf] = useState({ name: "", description: "", language: "es" });
  const [split, setSplit] = useState({ train: 70, validation: 15, test: 15 });
  const [busy, setBusy] = useState("");
  const csvRef = useRef<HTMLInputElement>(null);
  const zipRef = useRef<HTMLInputElement>(null);
  const upRef = useRef<HTMLInputElement>(null);
  const [up, setUp] = useState({ emotion: "neutral", satisfaction: "", speaker_group: "" });
  const [labels, setLabels] = useState<string[]>([]);

  const loadList = useCallback(async () => {
    const l = await api("/datasets"); setList(l);
    if (!sel && l.length) setSel(l[0].id);
  }, [sel]);
  const loadDs = useCallback(async () => {
    if (!sel) return;
    setDs(await api(`/datasets/${sel}`));
    setSamples(await api(`/datasets/${sel}/samples`, { query: { page, page_size: 15, issues: onlyIssues } }));
  }, [sel, page, onlyIssues]);
  useEffect(() => { loadList().catch((e) => toast("err", e.message)); api("/labels").then((r) => setLabels(r.labels)); }, []); // eslint-disable-line
  useEffect(() => { loadDs().catch((e) => toast("err", e.message)); }, [loadDs, toast]);

  const run = async (key: string, fn: () => Promise<any>, ok: string) => {
    setBusy(key);
    try { const r = await fn(); toast("ok", typeof ok === "string" ? ok : "Listo"); await loadDs(); await loadList(); return r; }
    catch (e: any) { toast("err", e.message); } finally { setBusy(""); }
  };
  const st = ds?.stats;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="text-2xl font-semibold">Dataset</h1><p className="text-sm text-muted">Datos etiquetados propios para entrenar y evaluar modelos · versionados</p></div>
        <div className="flex gap-2">
          <select value={sel} onChange={(e) => { setSel(e.target.value); setPage(1); }} aria-label="Dataset">
            {list?.map((d) => <option key={d.id} value={d.id}>{d.code} · {d.n_samples} muestras</option>)}
            {!list?.length && <option>Sin datasets</option>}
          </select>
          {can("ANALYST") && <Button variant="primary" onClick={() => setShowNew(true)}>＋ Nuevo dataset</Button>}
        </div>
      </div>

      {list && list.length === 0 && <Card><Empty title="Aún no hay datasets" text="Cree un dataset y agregue muestras desde la pantalla de Etiquetado, subiendo audios o importando un CSV." /></Card>}
      {!ds && list && list.length > 0 && <Skeleton className="h-64" />}

      {ds && (<>
        <Card title={<span>{ds.code} {ds.frozen && <Badge tone="brand">congelado</Badge>}</span>} subtitle={ds.description || `Idioma: ${ds.language || "—"} · Modelo utilizado: ${ds.model_used || "—"}`}
          actions={can("ANALYST") && (<>
            <Button size="sm" disabled={!!busy} onClick={() => run("val", () => api(`/datasets/${ds.id}/validate`, { method: "POST" }), "Validación completada")}>Validar</Button>
            <Button size="sm" disabled={!!busy || ds.frozen} onClick={() => run("ver", async () => { const n = await api(`/datasets/${ds.id}/version`, { method: "POST" }); setSel(n.id); }, "Nueva versión creada")}>Nueva versión</Button>
            <Button size="sm" disabled={ds.frozen} onClick={() => run("frz", () => api(`/datasets/${ds.id}/freeze`, { method: "POST" }), "Dataset congelado")}>Congelar</Button>
            <a href={downloadUrl(`/datasets/${ds.id}/export`)}><Button size="sm">Exportar CSV</Button></a></>)}>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <Stat label="Muestras" value={ds.n_samples.toLocaleString("es")} />
            <Stat label="Duración total" value={`${(ds.total_duration / 3600).toFixed(2)} h`} hint={mmss(ds.total_duration)} />
            <Stat label="Duración promedio" value={`${st?.avg_duration ?? 0}s`} />
            <Stat label="Personas/llamadas" value={st?.groups ?? 0} hint="grupos para evitar fuga de datos" />
            <Stat label="Sin etiqueta" value={st?.unlabeled ?? 0} tone={st?.unlabeled ? "text-warn" : undefined} />
          </div>
          {st?.imbalance?.imbalanced && <div className="mt-4"><Notice tone="warn"><b>Alerta de desbalance.</b> {st.imbalance.message} Estrategias disponibles en Entrenamiento: <i>class weights</i>, <i>oversampling</i>, <i>augmentation</i>.</Notice></div>}
        </Card>

        <div className="grid gap-5 xl:grid-cols-3">
          <Card title="Distribución por emoción">{Object.keys(st?.emotions || {}).length ? <Bars data={st.emotions} colorOf={emoColor} /> : <p className="text-sm text-muted">Sin muestras etiquetadas.</p>}</Card>
          <Card title="Distribución por speaker">{Object.keys(st?.speakers || {}).length ? <Bars data={st.speakers} /> : <p className="text-sm text-muted">—</p>}</Card>
          <Card title="Distribución por satisfacción">{Object.keys(st?.satisfaction || {}).length ? <Bars data={st.satisfaction} /> : <p className="text-sm text-muted">Sin etiquetas de satisfacción.</p>}</Card>
        </div>

        {can("ANALYST") && !ds.frozen && (
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Particiones train / validation / test" subtitle="Se reparten personas/llamadas completas (sin fuga de datos) y se estratifican las clases">
              <div className="grid grid-cols-3 gap-3">
                {(["train", "validation", "test"] as const).map((k) => <Field key={k} label={`${k.toUpperCase()} %`}><input type="number" className="w-full" min={0} max={100} value={split[k]} onChange={(e) => setSplit({ ...split, [k]: +e.target.value })} /></Field>)}
              </div>
              <div className="mt-3 flex items-center justify-between">
                <span className="text-xs text-muted">Actual: {Object.entries(st?.splits || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "sin asignar"}</span>
                <Button variant="primary" disabled={!!busy} onClick={async () => { const r = await run("split", () => api(`/datasets/${ds.id}/split`, { body: { train: split.train / 100, validation: split.validation / 100, test: split.test / 100 } }), "Particiones creadas"); r?.warnings?.forEach((w: string) => toast("info", w)); }}>Crear particiones</Button>
              </div>
            </Card>
            <Card title="Agregar datos" subtitle="También puede etiquetar segmentos de llamadas analizadas en «Etiquetado»">
              <div className="space-y-3 text-sm">
                <div className="flex flex-wrap items-end gap-2">
                  <input ref={csvRef} type="file" accept=".csv" className="max-w-[220px] text-xs" aria-label="Manifiesto CSV" />
                  <input ref={zipRef} type="file" accept=".zip" className="max-w-[220px] text-xs" aria-label="ZIP de audios" />
                  <Button disabled={!!busy} onClick={() => {
                    const c = csvRef.current?.files?.[0]; if (!c) return toast("err", "Seleccione el CSV de manifiesto.");
                    const fd = new FormData(); fd.append("manifest", c); const z = zipRef.current?.files?.[0]; if (z) fd.append("audio_zip", z);
                    run("imp", async () => { const r = await api(`/datasets/${ds.id}/import`, { form: fd }); if (r.error_count) toast("info", `${r.error_count} filas con error.`); return r; }, "Importación completada");
                  }}>Importar CSV + ZIP</Button>
                </div>
                <p className="text-xs text-muted">Columnas: <code>audio, speaker, emotion, satisfaction, start, end, language, speaker_group</code>. Tiempos en segundos o HH:MM:SS.</p>
                <div className="flex flex-wrap items-end gap-2 border-t border-line pt-3">
                  <input ref={upRef} type="file" accept="audio/*" className="max-w-[200px] text-xs" aria-label="Audio" />
                  <select value={up.emotion} onChange={(e) => setUp({ ...up, emotion: e.target.value })}>{labels.map((l) => <option key={l} value={l}>{EMO_ES[l] || l}</option>)}</select>
                  <input type="number" min={0} max={100} placeholder="Satisf." className="w-20" value={up.satisfaction} onChange={(e) => setUp({ ...up, satisfaction: e.target.value })} />
                  <input placeholder="Persona (grupo)" className="w-32" value={up.speaker_group} onChange={(e) => setUp({ ...up, speaker_group: e.target.value })} />
                  <Button disabled={!!busy} onClick={() => {
                    const f = upRef.current?.files?.[0]; if (!f) return toast("err", "Seleccione un audio.");
                    const fd = new FormData(); fd.append("file", f); fd.append("emotion", up.emotion); if (up.satisfaction) fd.append("satisfaction", up.satisfaction); if (up.speaker_group) fd.append("speaker_group", up.speaker_group);
                    run("up", () => api(`/datasets/${ds.id}/samples/upload`, { form: fd }), "Muestra agregada");
                  }}>Subir muestra</Button>
                </div>
                <Button variant="ghost" size="sm" disabled={!!busy} onClick={() => run("fb", async () => { const r = await api(`/datasets/${ds.id}/import-feedback`, { method: "POST" }); toast("info", `${r.imported} muestras creadas desde satisfacción real (CSAT/encuestas).`); }, "Listo")}>Importar satisfacción real de llamadas (CSAT/encuesta)</Button>
              </div>
            </Card>
          </div>)}

        <Card title="Muestras" pad={false} actions={<label className="flex items-center gap-1.5 text-xs text-muted"><input type="checkbox" checked={onlyIssues} onChange={(e) => { setOnlyIssues(e.target.checked); setPage(1); }} />Solo con problemas</label>}>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead><tr className="border-b border-line text-left text-xs text-muted">{["ID", "Origen", "Speaker", "Emoción", "Satisf.", "Duración", "Split", "Validación", ""].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
              <tbody>
                {samples?.items.map((s: any) => (
                  <tr key={s.id} className="border-b border-line/60">
                    <td className="px-3 py-2 text-xs text-muted">{s.id.slice(0, 8)}</td><td className="max-w-[160px] truncate px-3 py-2 text-xs" title={s.source}>{s.source || "—"}</td>
                    <td className="px-3 py-2">{s.speaker || "—"}</td>
                    <td className="px-3 py-2">{s.emotion ? <Badge>{EMO_ES[s.emotion] || s.emotion}</Badge> : <Badge tone="warn">sin etiqueta</Badge>}</td>
                    <td className="px-3 py-2 tabular-nums">{s.satisfaction ?? "—"}</td><td className="px-3 py-2 tabular-nums">{s.duration ?? "—"}s</td>
                    <td className="px-3 py-2">{s.split || "—"}</td>
                    <td className="px-3 py-2">{s.validation?.issues?.length ? s.validation.issues.map((i: string) => <Badge key={i} tone="warn" className="mr-1">{i}</Badge>) : <Badge tone="good">ok</Badge>}</td>
                    <td className="px-3 py-2 text-right">{!DEMO && <audio controls preload="none" className="hidden h-7 sm:inline-block" src={`/api/v1/datasets/${ds.id}/samples/${s.id}/audio`} />}{can("ANALYST") && !ds.frozen && <Button size="sm" variant="danger" onClick={() => run("del", () => api(`/datasets/${ds.id}/samples/${s.id}`, { method: "DELETE" }), "Muestra eliminada")}>✕</Button>}</td>
                  </tr>))}
              </tbody>
            </table>
          </div>
          {samples?.items.length === 0 && <Empty title="Sin muestras" />}
          <div className="px-4 pb-3"><Pagination page={page} pageSize={15} total={samples?.total ?? 0} onPage={setPage} /></div>
        </Card>
      </>)}

      <Modal open={showNew} onClose={() => setShowNew(false)} title="Nuevo dataset">
        <div className="space-y-3">
          <Field label="Nombre"><input className="w-full" value={nf.name} onChange={(e) => setNf({ ...nf, name: e.target.value })} placeholder="CALL_DATASET" /></Field>
          <Field label="Descripción"><input className="w-full" value={nf.description} onChange={(e) => setNf({ ...nf, description: e.target.value })} /></Field>
          <Field label="Idioma"><select className="w-full" value={nf.language} onChange={(e) => setNf({ ...nf, language: e.target.value })}><option value="es">Español</option><option value="en">Inglés</option></select></Field>
          <div className="flex justify-end gap-2"><Button onClick={() => setShowNew(false)}>Cancelar</Button><Button variant="primary" disabled={nf.name.length < 2} onClick={async () => { try { const d = await api("/datasets", { body: nf }); setShowNew(false); setSel(d.id); await loadList(); toast("ok", "Dataset creado."); } catch (e: any) { toast("err", e.message); } }}>Crear</Button></div>
        </div>
      </Modal>
    </div>
  );
}
