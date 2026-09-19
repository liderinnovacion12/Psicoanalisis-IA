"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import EChart, { axisStyle } from "@/components/EChart";
import { Badge, Button, Card, Empty, Modal, Notice, Skeleton, Tabs } from "@/components/ui";
import { dateTime, EMO_ES } from "@/lib/format";

const TONE: Record<string, any> = { PRODUCTION: "good", VALIDATION: "brand", TRAINING: "warn", ARCHIVED: "neutral" };
const f3 = (v?: number | null) => (v == null ? "—" : v.toFixed(3));

export default function Models() {
  const { toast, can } = useApp();
  const [models, setModels] = useState<any[] | null>(null);
  const [tab, setTab] = useState<"registry" | "compare">("registry");
  const [detail, setDetail] = useState<any>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [cmp, setCmp] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [cmpDs, setCmpDs] = useState("");
  const { theme } = useApp();

  const load = useCallback(() => api("/models").then(setModels).catch((e) => toast("err", e.message)), [toast]);
  useEffect(() => { load(); api("/datasets").then(setDatasets); }, [load]);
  useEffect(() => {
    if (!picked.length) return setCmp([]);
    api("/models/compare", { query: { ids: picked, dataset_id: cmpDs || undefined } }).then(setCmp).catch(() => {});
  }, [picked, cmpDs]);

  async function activate(m: any) {
    try { const r = await api(`/models/${m.id}/activate`, { method: "POST" }); toast("ok", r.message); load(); } catch (e: any) { toast("err", e.message); }
  }
  async function open(m: any) { try { setDetail(await api(`/models/${m.id}`)); } catch (e: any) { toast("err", e.message); } }
  async function evaluate(m: any) {
    if (!cmpDs) return toast("err", "Seleccione un dataset (con splits creados) en la pestaña Comparación.");
    try { const r = await api(`/models/${m.id}/evaluate`, { method: "POST", query: { dataset_id: cmpDs } }); toast("ok", r.message); } catch (e: any) { toast("err", e.message); }
  }
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-4)));

  const chart = useMemo(() => {
    const withM = cmp.filter((c) => c.metrics?.macro_f1 != null);
    if (!withM.length) return null;
    const dark = theme === "dark", ax = axisStyle(dark);
    const keys = [["accuracy", "Accuracy"], ["macro_f1", "Macro F1"], ["weighted_f1", "Weighted F1"], ["precision", "Precision"], ["recall", "Recall"]];
    return {
      legend: { top: 0, textStyle: { color: dark ? "#8B98A8" : "#64748B" } }, tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: keys.map((k) => k[1]), ...ax }, yAxis: { type: "value", min: 0, max: 1, splitLine: ax.splitLine, axisLabel: ax.axisLabel },
      series: withM.map((c) => ({ name: `${c.name}`, type: "bar", data: keys.map((k) => +(c.metrics[k[0]] ?? 0).toFixed(3)) })), grid: { left: 40, right: 12, top: 36, bottom: 28 },
    };
  }, [cmp, theme]);

  return (
    <div className="space-y-5">
      <div><h1 className="text-2xl font-semibold">Modelos</h1><p className="text-sm text-muted">Model Registry: versiones, estados, métricas y comparación</p></div>
      <Tabs tabs={[{ key: "registry", label: "Registro" }, { key: "compare", label: `Comparación (${picked.length})` }]} value={tab} onChange={setTab} />
      {!models ? <Skeleton className="h-64" /> : models.length === 0 ? <Card><Empty title="Sin modelos" /></Card> : (
        <Card pad={false}>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-sm">
              <thead><tr className="border-b border-line text-left text-xs text-muted">{["", "Nombre", "Versión", "Tipo", "Estado", "Base", "Idioma", "Macro F1", "Accuracy", "Creado", "Acciones"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className="border-b border-line/60 hover:bg-surface2/60">
                    <td className="px-3 py-2.5"><input type="checkbox" checked={picked.includes(m.id)} onChange={() => toggle(m.id)} aria-label="Comparar" /></td>
                    <td className="px-3 py-2.5 font-medium">{m.name} {m.is_active && <Badge tone="good">ACTIVO</Badge>} {m.is_builtin && <Badge>baseline</Badge>}</td>
                    <td className="px-3 py-2.5">{m.version}</td><td className="px-3 py-2.5">{m.kind}</td>
                    <td className="px-3 py-2.5"><Badge tone={TONE[m.status]}>{m.status}</Badge></td>
                    <td className="max-w-[180px] truncate px-3 py-2.5 text-xs text-muted" title={m.base_model || m.hf_id}>{m.base_model || m.hf_id || "—"}</td>
                    <td className="px-3 py-2.5">{m.language || "—"}</td>
                    <td className="px-3 py-2.5 tabular-nums">{f3(m.summary_metrics?.macro_f1)}</td><td className="px-3 py-2.5 tabular-nums">{f3(m.summary_metrics?.accuracy)}</td>
                    <td className="px-3 py-2.5 text-xs text-muted">{dateTime(m.created_at)}</td>
                    <td className="whitespace-nowrap px-3 py-2.5"><Button size="sm" onClick={() => open(m)}>Detalle</Button>{" "}
                      {can("ADMIN") && !m.is_active && m.status !== "TRAINING" && <Button size="sm" variant="primary" onClick={() => activate(m)}>Activar</Button>}</td>
                  </tr>))}
              </tbody>
            </table>
          </div>
        </Card>)}
      <Notice tone="info">El modelo <b>baseline</b> (<code>r-f/wav2vec-english-speech-emotion-recognition</code>) está entrenado en <b>inglés</b>. Su exactitud publicada no representa la precisión real en llamadas ni en español: evalúelo y compárelo con sus propios datos etiquetados.</Notice>

      {tab === "compare" && (
        <Card title="Comparación de modelos" subtitle="Para una comparación justa evalúe todos los modelos sobre el mismo conjunto TEST de un dataset"
          actions={<select value={cmpDs} onChange={(e) => setCmpDs(e.target.value)} aria-label="Dataset de comparación"><option value="">Métricas propias de cada modelo</option>{datasets.map((d) => <option key={d.id} value={d.id}>{d.code}</option>)}</select>}>
          {!picked.length ? <Empty title="Seleccione 2 a 4 modelos en el registro" /> : (<div className="space-y-5">
            {can("ANALYST") && cmpDs && <div className="flex flex-wrap items-center gap-2 text-xs"><span className="text-muted">Evaluar sobre el TEST de este dataset:</span>{cmp.map((c) => <Button key={c.id} size="sm" onClick={() => evaluate(c)}>{c.name}</Button>)}</div>}
            <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b border-line text-left text-xs text-muted"><th className="py-2">Métrica</th>{cmp.map((c) => <th key={c.id} className="py-2 font-medium">{c.name}<br /><span className="font-normal">{c.status}</span></th>)}</tr></thead>
              <tbody>{[["accuracy", "Accuracy"], ["macro_f1", "Macro F1"], ["weighted_f1", "Weighted F1"], ["precision", "Precision (macro)"], ["recall", "Recall (macro)"], ["n_samples", "Muestras"]].map(([k, l]) => (
                <tr key={k} className="border-b border-line/60"><td className="py-2 text-muted">{l}</td>{cmp.map((c) => <td key={c.id} className="py-2 font-medium tabular-nums">{k === "n_samples" ? (c.metrics?.n_samples ?? "—") : f3(c.metrics?.[k])}</td>)}</tr>))}</tbody></table></div>
            {chart && <EChart option={chart as any} height={260} />}
            <div className="grid gap-5 xl:grid-cols-2">{cmp.filter((c) => c.metrics?.confusion_matrix?.length).map((c) => <div key={c.id}><p className="mb-2 text-sm font-medium">{c.name}</p><ConfusionMatrix labels={c.metrics.labels} matrix={c.metrics.confusion_matrix} /></div>)}</div>
          </div>)}
        </Card>)}

      <Modal open={!!detail} onClose={() => setDetail(null)} title={detail ? `${detail.name} (${detail.version})` : ""} wide>
        {detail && (<div className="space-y-4 text-sm">
          <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[["Estado", detail.status], ["Tipo", detail.kind], ["Base", detail.base_model || detail.hf_id || "—"], ["Idioma", detail.language || "—"], ["Dataset", detail.parameters?.dataset || "—"], ["Etiquetas", (detail.labels || []).map((l: string) => EMO_ES[l] || l).join(", ") || "—"], ["Creado", dateTime(detail.created_at)], ["Loader", detail.loader]].map(([k, v]) => <div key={k}><dt className="text-[11px] text-muted">{k}</dt><dd className="break-words font-medium">{v}</dd></div>)}
          </dl>
          {detail.parameters && <details className="text-xs"><summary className="cursor-pointer text-muted">Parámetros de entrenamiento</summary><pre className="mt-2 overflow-auto rounded-lg bg-surface2 p-3">{JSON.stringify(detail.parameters, null, 2)}</pre></details>}
          {detail.metrics?.length ? detail.metrics.map((m: any) => (
            <div key={m.id} className="rounded-lg border border-line p-3">
              <p className="mb-2 font-medium">Split: {m.split}{m.extra?.dataset_id ? " (evaluación sobre dataset)" : ""} · {m.n_samples} muestras</p>
              {m.macro_f1 != null ? (<>
                <p className="mb-2 text-xs text-muted">Accuracy {f3(m.accuracy)} · Macro F1 {f3(m.macro_f1)} · Weighted F1 {f3(m.weighted_f1)} · Precision {f3(m.precision)} · Recall {f3(m.recall)}</p>
                <div className="grid gap-4 lg:grid-cols-2"><ConfusionMatrix labels={m.labels || []} matrix={m.confusion_matrix || []} />
                  <table className="w-full text-xs"><thead><tr className="text-left text-muted"><th>Clase</th><th>Prec.</th><th>Recall</th><th>F1</th><th>n</th></tr></thead><tbody>{Object.entries<any>(m.per_class || {}).map(([k, v]) => <tr key={k} className="border-t border-line/50 tabular-nums"><td>{EMO_ES[k] || k}</td><td>{f3(v.precision)}</td><td>{f3(v.recall)}</td><td>{f3(v.f1)}</td><td>{v.support}</td></tr>)}</tbody></table></div>
              </>) : <p className="text-xs text-muted">MAE {f3(m.extra?.mae)} · RMSE {f3(m.extra?.rmse)} · R² {f3(m.extra?.r2)} · Pearson {f3(m.extra?.pearson)}</p>}
            </div>)) : <p className="text-muted">Aún no hay métricas. {detail.is_builtin ? "Evalúe el baseline con sus propios datos desde la pestaña Comparación." : ""}</p>}
        </div>)}
      </Modal>
    </div>
  );
}
