"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import EChart, { axisStyle } from "@/components/EChart";
import { Badge, Button, Card, Empty, Field, Notice, Progress, Skeleton, Stat } from "@/components/ui";
import { dateTime } from "@/lib/format";

const DEFAULTS = { learning_rate: 3e-5, batch_size: 8, epochs: 10, validation_split: 0.15, test_split: 0.15, weight_decay: 0.01, warmup_steps: 50, gradient_accumulation: 2, early_stopping_patience: 3, imbalance_strategy: "class_weights", freeze_feature_encoder: true, augment: false };
const TONE: Record<string, any> = { RUNNING: "brand", PENDING: "neutral", STOPPING: "warn", STOPPED: "warn", COMPLETED: "good", FAILED: "bad" };

export default function Training() {
  const { toast, can, theme } = useApp();
  const [datasets, setDatasets] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [runs, setRuns] = useState<any[] | null>(null);
  const [f, setF] = useState<any>({ ...DEFAULTS, dataset_id: "", base_model_id: "", kind: "emotion", name: "", resplit: false });
  const [sel, setSel] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const loadRuns = useCallback(() => api("/training").then((r) => { setRuns(r); return r; }), []);
  useEffect(() => {
    api("/datasets").then((d) => { setDatasets(d); if (d[0]) setF((x: any) => ({ ...x, dataset_id: x.dataset_id || d[0].id })); });
    api("/models", { query: { kind: "emotion" } }).then((m) => { setModels(m); const a = m.find((x: any) => x.is_active); if (a) setF((x: any) => ({ ...x, base_model_id: a.id })); });
    loadRuns().then((r) => r[0] && setSel(r[0].id));
  }, [loadRuns]);
  // sondeo en vivo mientras haya entrenamiento activo
  useEffect(() => {
    if (!runs?.some((r) => ["RUNNING", "PENDING", "STOPPING"].includes(r.status))) return;
    const t = setInterval(loadRuns, 2500); return () => clearInterval(t);
  }, [runs, loadRuns]);

  const run = runs?.find((r) => r.id === sel);
  const dark = theme === "dark";
  const chart = useMemo(() => {
    if (!run?.metrics_history?.length) return null;
    const ax = axisStyle(dark); const h = run.metrics_history;
    return {
      legend: { top: 0, textStyle: { color: dark ? "#8B98A8" : "#64748B", fontSize: 11 } }, tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: h.map((x: any) => x.epoch), name: "Época", ...ax },
      yAxis: [{ type: "value", name: "Loss", splitLine: ax.splitLine, axisLabel: ax.axisLabel }, { type: "value", name: "Macro F1", min: 0, max: 1, splitLine: { show: false }, axisLabel: ax.axisLabel }],
      series: [
        { name: "Train loss", type: "line", data: h.map((x: any) => x.train_loss), itemStyle: { color: "#E0762A" } },
        { name: "Val loss", type: "line", data: h.map((x: any) => x.val_loss), itemStyle: { color: "#3B7DD8" } },
        { name: "Macro F1", type: "line", yAxisIndex: 1, data: h.map((x: any) => x.macro_f1), itemStyle: { color: "#2E9E5B" }, lineStyle: { width: 3 } },
      ],
      grid: { left: 48, right: 48, top: 36, bottom: 30 },
    };
  }, [run, dark]);

  async function start() {
    setBusy(true);
    try {
      const { dataset_id, base_model_id, kind, name, resplit, augment, ...params } = f;
      params.augmentation = { enabled: !!augment, noise_snr_db: [20, 35], gain_db: [-6, 6], time_mask_seconds: 0.2, speed_range: [1, 1], pitch_semitones: 0 };
      const r = await api("/training/start", { body: { dataset_id, kind, base_model_id: kind === "emotion" ? base_model_id : null, name: name || null, params, resplit } });
      toast("ok", "Entrenamiento iniciado."); await loadRuns(); setSel(r.id);
    } catch (e: any) { toast("err", e.message); } finally { setBusy(false); }
  }
  const act = async (id: string, what: "stop" | "resume") => { try { const r = await api(`/training/${id}/${what}`, { method: "POST" }); toast("ok", r.message); loadRuns(); } catch (e: any) { toast("err", e.message); } };
  const ds = datasets.find((d) => d.id === f.dataset_id);
  const num = (k: string, step = 1, min = 0, max?: number) => <input type="number" step={step} min={min} max={max} className="w-full" value={f[k]} onChange={(e) => setF({ ...f, [k]: parseFloat(e.target.value) })} />;

  return (
    <div className="space-y-5">
      <div><h1 className="text-2xl font-semibold">Entrenamiento</h1><p className="text-sm text-muted">Fine-tuning del modelo emocional y entrenamiento del modelo de satisfacción con datos propios</p></div>
      {can("ANALYST") ? (
        <Card title="Nueva ejecución">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Field label="Tipo"><select className="w-full" value={f.kind} onChange={(e) => setF({ ...f, kind: e.target.value })}><option value="emotion">Modelo emocional (fine-tuning)</option><option value="satisfaction">Modelo de satisfacción (regresor)</option></select></Field>
            <Field label="Dataset"><select className="w-full" value={f.dataset_id} onChange={(e) => setF({ ...f, dataset_id: e.target.value })}>{datasets.map((d) => <option key={d.id} value={d.id}>{d.code} · {d.n_samples} muestras</option>)}</select></Field>
            {f.kind === "emotion" && <Field label="Modelo base"><select className="w-full" value={f.base_model_id} onChange={(e) => setF({ ...f, base_model_id: e.target.value })}>{models.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.version}){m.is_active ? " · activo" : ""}</option>)}</select></Field>}
            <Field label="Nombre (opcional)"><input className="w-full" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></Field>
          </div>
          {f.kind === "emotion" && (<>
            <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4 xl:grid-cols-5">
              <Field label="Learning rate">{num("learning_rate", 1e-6, 0)}</Field><Field label="Batch size">{num("batch_size", 1, 1)}</Field><Field label="Epochs">{num("epochs", 1, 1)}</Field>
              <Field label="Validation split">{num("validation_split", 0.01, 0.05, 0.4)}</Field><Field label="Test split">{num("test_split", 0.01, 0, 0.4)}</Field>
              <Field label="Weight decay">{num("weight_decay", 0.001)}</Field><Field label="Warmup steps">{num("warmup_steps", 1)}</Field><Field label="Gradient accumulation">{num("gradient_accumulation", 1, 1)}</Field>
              <Field label="Early stopping (paciencia)">{num("early_stopping_patience", 1)}</Field>
              <Field label="Desbalance"><select className="w-full" value={f.imbalance_strategy} onChange={(e) => setF({ ...f, imbalance_strategy: e.target.value })}><option value="none">Ninguna</option><option value="class_weights">Class weights</option><option value="oversample">Oversampling</option></select></Field>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-5 text-sm">
              <label className="flex items-center gap-2"><input type="checkbox" checked={f.freeze_feature_encoder} onChange={(e) => setF({ ...f, freeze_feature_encoder: e.target.checked })} />Congelar extractor de características</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={f.augment} onChange={(e) => setF({ ...f, augment: e.target.checked })} />Data augmentation segura (ruido, volumen, time-mask)</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={f.resplit} onChange={(e) => setF({ ...f, resplit: e.target.checked })} />Recrear particiones (agrupadas por persona)</label>
            </div>
          </>)}
          {ds?.stats?.imbalance?.imbalanced && f.kind === "emotion" && <div className="mt-3"><Notice tone="warn">{ds.stats.imbalance.message}</Notice></div>}
          {f.kind === "emotion" && <p className="mt-3 text-xs text-muted">Requiere GPU para tiempos razonables con el modelo base (~300 M de parámetros). En CPU es viable solo con datasets pequeños. Al terminar se registra un nuevo modelo en estado VALIDATION; actívelo desde «Modelos» tras comparar métricas.</p>}
          <div className="mt-4 flex justify-end"><Button variant="primary" disabled={busy || !f.dataset_id} onClick={start}>{busy ? "Iniciando…" : "Iniciar entrenamiento"}</Button></div>
        </Card>) : <Notice tone="warn">Su rol solo permite consultar entrenamientos.</Notice>}

      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <Card title="Ejecuciones" pad={false}>
          {!runs ? <div className="p-4"><Skeleton className="h-24" /></div> : runs.length === 0 ? <Empty title="Sin entrenamientos" /> : (
            <ul className="max-h-[560px] divide-y divide-line overflow-auto">
              {runs.map((r) => (
                <li key={r.id}><button onClick={() => setSel(r.id)} className={`w-full px-4 py-3 text-left hover:bg-surface2 ${sel === r.id ? "bg-brand/10" : ""}`}>
                  <div className="flex items-center justify-between"><span className="truncate text-sm font-medium">{r.name}</span><Badge tone={TONE[r.status]}>{r.status}</Badge></div>
                  <p className="mt-0.5 text-xs text-muted">{r.kind} · {dateTime(r.created_at)}</p>
                  {["RUNNING", "PENDING"].includes(r.status) && <Progress value={r.progress} className="mt-2" />}
                </button></li>))}
            </ul>)}
        </Card>
        {run ? (
          <Card title={run.name} subtitle={`Dispositivo: ${run.device || "—"} · inicio ${dateTime(run.started_at)}`}
            actions={can("ANALYST") && (<>
              {["RUNNING", "PENDING"].includes(run.status) && <Button variant="danger" size="sm" onClick={() => act(run.id, "stop")}>Detener</Button>}
              {run.resumable && <Button variant="primary" size="sm" onClick={() => act(run.id, "resume")}>Continuar</Button>}</>)}>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <Stat label="Estado" value={run.status} />
              <Stat label="Época" value={run.kind === "emotion" ? `${run.current_epoch}/${run.total_epochs}` : "—"} />
              <Stat label="Mejor Macro F1" value={run.kind === "emotion" && run.best_metric != null ? run.best_metric.toFixed(3) : run.kind === "satisfaction" && run.best_metric != null ? `MAE ${(-run.best_metric).toFixed(1)}` : "—"} />
              <Stat label="Progreso" value={`${Math.round(run.progress)}%`} />
            </div>
            <Progress value={run.progress} className="mt-4" />
            {run.error && <div className="mt-3"><Notice tone="bad">{run.error}</Notice></div>}
            {run.status === "STOPPED" && <div className="mt-3"><Notice tone="warn">Entrenamiento detenido. Se guardó un checkpoint; puede continuar donde quedó.</Notice></div>}
            {chart && <div className="mt-4"><EChart option={chart as any} height={260} /></div>}
            {run.metrics_history?.length > 0 && run.kind === "emotion" && (
              <div className="mt-3 overflow-x-auto"><table className="w-full text-xs"><thead><tr className="border-b border-line text-left text-muted">{["Época", "Train loss", "Val loss", "Accuracy", "Macro F1", "Precision", "Recall"].map((h) => <th key={h} className="py-1.5 pr-3 font-medium">{h}</th>)}</tr></thead>
                <tbody>{run.metrics_history.map((h: any) => <tr key={h.epoch} className="border-b border-line/50 tabular-nums"><td className="py-1.5">{h.epoch}</td><td>{h.train_loss?.toFixed(3)}</td><td>{h.val_loss?.toFixed(3)}</td><td>{h.val_accuracy?.toFixed(3)}</td><td className="font-medium">{h.macro_f1?.toFixed(3)}</td><td>{h.precision?.toFixed(3)}</td><td>{h.recall?.toFixed(3)}</td></tr>)}</tbody></table></div>)}
            {run.model_id && <p className="mt-3 text-xs text-muted">Modelo registrado: <a className="text-brand" href="/models">ver en Modelos</a></p>}
          </Card>) : <Card><Empty title="Seleccione una ejecución" /></Card>}
      </div>
    </div>
  );
}
