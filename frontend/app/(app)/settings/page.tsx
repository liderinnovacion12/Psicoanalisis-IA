"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { Badge, Button, Card, Field, Modal, Notice, Skeleton, Tabs } from "@/components/ui";
import { dateTime } from "@/lib/format";

type Tab = "general" | "ml" | "audio" | "satisfaction" | "models" | "storage" | "privacy" | "users";

// Acceso por ruta "a.b.c" sobre el objeto de configuración
const get = (o: any, p: string) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
const setIn = (o: any, p: string, v: any) => { const c = JSON.parse(JSON.stringify(o)); const ks = p.split("."); let cur = c; ks.slice(0, -1).forEach((k) => (cur = cur[k] ??= {})); cur[ks.at(-1)!] = v; return c; };

function Num({ cfg, setCfg, path, label, hint, step = 1, min, max }: any) {
  return <Field label={label} hint={hint}><input type="number" className="w-full" step={step} min={min} max={max} value={get(cfg, path) ?? ""} onChange={(e) => setCfg(setIn(cfg, path, e.target.value === "" ? null : parseFloat(e.target.value)))} /></Field>;
}
function Sel({ cfg, setCfg, path, label, options, hint }: any) {
  return <Field label={label} hint={hint}><select className="w-full" value={get(cfg, path)} onChange={(e) => setCfg(setIn(cfg, path, e.target.value))}>{options.map((o: any) => <option key={o[0] ?? o} value={o[0] ?? o}>{o[1] ?? o}</option>)}</select></Field>;
}
function Chk({ cfg, setCfg, path, label, hint }: any) {
  return <label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-0.5" checked={!!get(cfg, path)} onChange={(e) => setCfg(setIn(cfg, path, e.target.checked))} /><span>{label}{hint && <span className="block text-xs text-muted">{hint}</span>}</span></label>;
}

export default function Settings() {
  const { toast, can } = useApp();
  const admin = can("ADMIN");
  const [tab, setTab] = useState<Tab>("general");
  const [all, setAll] = useState<any>(null);
  const [draft, setDraft] = useState<Record<string, any>>({});
  const [busy, setBusy] = useState(false);
  const [users, setUsers] = useState<any[]>([]);
  const [showUser, setShowUser] = useState(false);
  const [nu, setNu] = useState({ email: "", name: "", password: "", role: "VIEWER" });
  const [cal, setCal] = useState<any>(null);
  const [advanced, setAdvanced] = useState("");

  const load = useCallback(async () => { const r = await api("/settings"); setAll(r); setDraft(r.config); }, []);
  useEffect(() => { load().catch((e) => toast("err", e.message)); }, [load, toast]);
  useEffect(() => { if (tab === "users" && admin) api("/users").then(setUsers).catch(() => {}); if (tab === "satisfaction") api("/settings/calibration/status").then(setCal).catch(() => {}); }, [tab, admin]);

  const section = ({ general: "app", ml: "emotion", audio: "audio", satisfaction: "satisfaction", models: "emotion", storage: "app", privacy: "app" } as any)[tab];
  const cfg = draft[section];
  const setCfg = (v: any) => setDraft({ ...draft, [section]: v });
  const dirty = useMemo(() => all && section && JSON.stringify(all.config[section]) !== JSON.stringify(draft[section]), [all, draft, section]);

  async function save() {
    setBusy(true);
    try { await api(`/settings/${section}`, { method: "PUT", body: { value: draft[section] } }); toast("ok", "Configuración guardada. Se aplicará a los próximos análisis."); await load(); } catch (e: any) { toast("err", e.message); } finally { setBusy(false); }
  }
  async function reset() {
    if (!confirm("¿Restablecer esta sección a los valores por defecto?")) return;
    try { await api(`/settings/${section}`, { method: "DELETE" }); toast("ok", "Restablecido."); await load(); } catch (e: any) { toast("err", e.message); }
  }

  if (!all) return <Skeleton className="h-96" />;
  const weights = draft.satisfaction?.weights || {};
  const ranges: any[] = draft.satisfaction?.interpretation || [];

  const Actions = admin && section ? (
    <div className="mt-5 flex items-center justify-between border-t border-line pt-4">
      <Button variant="ghost" onClick={reset}>Restablecer valores por defecto</Button>
      <div className="flex items-center gap-3">{dirty && <span className="text-xs text-warn">Cambios sin guardar</span>}<Button variant="primary" disabled={!dirty || busy} onClick={save}>Guardar cambios</Button></div>
    </div>) : <p className="mt-4 text-xs text-muted">Solo un administrador puede modificar la configuración.</p>;

  return (
    <div className="space-y-5">
      <div><h1 className="text-2xl font-semibold">Configuración</h1><p className="text-sm text-muted">Los valores se guardan por organización y se registran junto a cada análisis (trazabilidad)</p></div>
      <Tabs tabs={[{ key: "general", label: "General" }, { key: "ml", label: "ML" }, { key: "audio", label: "Audio" }, { key: "satisfaction", label: "Satisfacción" }, { key: "models", label: "Modelos" }, { key: "storage", label: "Almacenamiento" }, { key: "privacy", label: "Privacidad" }, ...(admin ? [{ key: "users" as Tab, label: "Usuarios" }] : [])]} value={tab} onChange={setTab} />

      {tab === "general" && cfg && <Card title="Configuración general"><div className="grid gap-4 md:grid-cols-2">
        <Sel cfg={cfg} setCfg={setCfg} path="general.default_language" label="Idioma por defecto" options={cfg.general.supported_languages.map((l: string) => [l, l === "es" ? "Español" : l === "en" ? "Inglés" : l])} hint="Se detecta automáticamente por llamada; este idioma se prioriza si la detección es dudosa" />
        <Sel cfg={cfg} setCfg={setCfg} path="ui.default_theme" label="Tema por defecto" options={[["system", "Sistema"], ["light", "Claro"], ["dark", "Oscuro"]]} />
      </div>{Actions}</Card>}

      {tab === "ml" && cfg && <Card title="Configuración ML" subtitle="Modelo de emociones, dispositivo y fusión multimodal"><div className="grid gap-4 md:grid-cols-3">
        <Sel cfg={cfg} setCfg={setCfg} path="device" label="Dispositivo" options={["auto", "cuda", "cpu"]} hint="auto: usa la GPU NVIDIA (CUDA) si existe" />
        <Num cfg={cfg} setCfg={setCfg} path="temperature" label="Temperatura (calibración de confianza)" step={0.1} min={0.5} max={5} hint=">1 suaviza probabilidades sobreconfiadas" />
        <Num cfg={cfg} setCfg={setCfg} path="max_window_seconds" label="Ventana máxima (s)" />
        <Num cfg={cfg} setCfg={setCfg} path="confidence_levels.high" label="Umbral confianza «Alta»" step={0.05} min={0} max={1} />
        <Num cfg={cfg} setCfg={setCfg} path="confidence_levels.medium" label="Umbral confianza «Moderada»" step={0.05} min={0} max={1} />
        <Sel cfg={cfg} setCfg={setCfg} path="text_sentiment.engine" label="Análisis de texto" options={[["lexicon", "Léxico ES/EN (local)"], ["hf", "Modelo Hugging Face"]]} />
        <Num cfg={cfg} setCfg={setCfg} path="fusion.audio_weight" label="Peso audio (fusión)" step={0.05} /><Num cfg={cfg} setCfg={setCfg} path="fusion.prosody_weight" label="Peso prosodia" step={0.05} /><Num cfg={cfg} setCfg={setCfg} path="fusion.text_weight" label="Peso texto" step={0.05} />
      </div>
        <Notice tone="info"><span className="text-xs">Modelo base: <b>{cfg.default_model.hf_id}</b> (idioma: {cfg.default_model.language}). Para español, entrene o cargue un modelo propio y actívelo en «Modelos».</span></Notice>{Actions}</Card>}

      {tab === "audio" && cfg && <div className="space-y-5">
        <Card title="Segmentación por ventanas" subtitle="Permite detectar emociones breves en llamadas de cualquier duración"><div className="grid gap-4 md:grid-cols-4">
          <Num cfg={cfg} setCfg={setCfg} path="segmentation.window_size" label="Window size (s)" step={0.5} min={0.5} max={30} /><Num cfg={cfg} setCfg={setCfg} path="segmentation.hop_size" label="Hop size (s)" step={0.25} min={0.1} />
          <Num cfg={cfg} setCfg={setCfg} path="segmentation.batch_size" label="Batch size inferencia" /><Num cfg={cfg} setCfg={setCfg} path="segmentation.merge_gap" label="Unir turnos (gap s)" step={0.1} />
        </div></Card>
        <Card title="Procesamiento de audio"><div className="grid gap-4 md:grid-cols-4">
          <Sel cfg={cfg} setCfg={setCfg} path="normalization.sample_rate" label="Sample rate (Hz)" options={[16000, 22050, 24000]} hint="El baseline requiere 16000" />
          <Chk cfg={cfg} setCfg={setCfg} path="normalization.loudnorm" label="Normalizar volumen (loudnorm)" /><Sel cfg={cfg} setCfg={setCfg} path="vad.engine" label="VAD" options={[["silero", "Silero"], ["energy", "Energía"]]} />
          <Num cfg={cfg} setCfg={setCfg} path="vad.threshold" label="Umbral VAD" step={0.05} min={0} max={1} />
          <Sel cfg={cfg} setCfg={setCfg} path="diarization.engine" label="Diarización" options={[["auto", "Automática"], ["pyannote", "pyannote"], ["channels", "Por canal"], ["spectral", "Respaldo"]]} />
          <Num cfg={cfg} setCfg={setCfg} path="diarization.minor_speaker_ratio" label="Voz minoritaria (%)" step={0.01} />
          <Sel cfg={cfg} setCfg={setCfg} path="transcription.model_size" label="Modelo Whisper" options={["tiny", "base", "small", "medium", "large-v3"]} hint="Más grande = más preciso y más lento" />
          <Sel cfg={cfg} setCfg={setCfg} path="transcription.language" label="Idioma de transcripción" options={[["auto", "Detectar"], ["es", "Español"], ["en", "Inglés"]]} />
          <Num cfg={cfg} setCfg={setCfg} path="channels.max_correlation" label="Correlación máx. entre canales" step={0.05} /><Num cfg={cfg} setCfg={setCfg} path="interruptions.long_silence_seconds" label="Silencio significativo (s)" step={0.5} />
          <Num cfg={cfg} setCfg={setCfg} path="quality.low_quality_threshold" label="Umbral de calidad baja" /><Num cfg={cfg} setCfg={setCfg} path="max_duration_hours" label="Duración máx. (h)" />
        </div>{Actions}</Card></div>}

      {tab === "satisfaction" && cfg && <div className="space-y-5">
        <Card title="Pesos del Satisfaction Engine" subtitle="Ningún peso está en el código. Se pueden ajustar aquí o aprender con datos reales.">
          <div className="grid gap-4 md:grid-cols-4 xl:grid-cols-5">
            {Object.keys(weights).map((k) => <Num key={k} cfg={cfg} setCfg={setCfg} path={`weights.${k}`} label={k} step={0.05} />)}
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <Sel cfg={cfg} setCfg={setCfg} path="mode" label="Modo" options={[["rules", "Reglas configurables"], ["model", "Modelo entrenado"], ["hybrid", "Híbrido"]]} />
            <Num cfg={cfg} setCfg={setCfg} path="hybrid_alpha" label="Peso del modelo (híbrido)" step={0.05} min={0} max={1} />
            <Num cfg={cfg} setCfg={setCfg} path="score.sensitivity" label="Sensibilidad" step={0.1} /><Num cfg={cfg} setCfg={setCfg} path="final_segment.fraction" label="Tramo final (fracción)" step={0.05} />
          </div>
        </Card>
        <Card title="Interpretación por rangos" subtitle="Se muestra junto al puntaje"><div className="space-y-2">
          {ranges.map((r, i) => (
            <div key={i} className="grid grid-cols-[70px_70px_1fr] items-center gap-2"><input type="number" value={r.min} onChange={(e) => setCfg(setIn(cfg, `interpretation`, ranges.map((x, j) => (j === i ? { ...x, min: +e.target.value } : x))))} />
              <input type="number" value={r.max} onChange={(e) => setCfg(setIn(cfg, `interpretation`, ranges.map((x, j) => (j === i ? { ...x, max: +e.target.value } : x))))} />
              <input value={r.label} onChange={(e) => setCfg(setIn(cfg, `interpretation`, ranges.map((x, j) => (j === i ? { ...x, label: e.target.value } : x))))} /></div>))}
        </div>{Actions}</Card>
        <Card title="Calibración con satisfacción real (CSAT / NPS / encuesta)" subtitle="Compara lo estimado por el modelo con la medición real registrada en las llamadas">
          {!cal ? <Skeleton className="h-16" /> : (<div className="space-y-2 text-sm">
            <p>Pares (predicha, real): <b>{cal.n}</b> (mínimo {cal.min_samples}) · Calibración {cal.enabled ? <Badge tone="good">activa</Badge> : <Badge>inactiva</Badge>}</p>
            {cal.proposed && <p className="text-muted">Ajuste propuesto: real ≈ {cal.proposed.a.toFixed(2)}·pred + {cal.proposed.b.toFixed(1)} · MAE {cal.proposed.mae_before?.toFixed(1)} → {cal.proposed.mae_after?.toFixed(1)}</p>}
            {admin && <Button disabled={cal.n < cal.min_samples} onClick={async () => { try { await api("/settings/calibration/apply", { method: "POST" }); toast("ok", "Calibración aplicada."); load(); api("/settings/calibration/status").then(setCal); } catch (e: any) { toast("err", e.message); } }}>Aplicar calibración</Button>}
            {cal.n < cal.min_samples && <p className="text-xs text-muted">Registre la satisfacción real desde el detalle de cada llamada (API: <code>POST /calls/{"{id}"}/feedback</code>) para habilitar la calibración.</p>}
          </div>)}
        </Card></div>}

      {tab === "models" && cfg && <Card title="Modelos"><p className="text-sm">El modelo activo se gestiona en <a className="text-brand" href="/models">Modelos</a>. Cada análisis guarda el nombre y la versión del modelo utilizado.</p>
        <p className="mt-2 text-xs text-muted">Modelo base configurado: <code>{cfg.default_model.hf_id}</code>. Para usar un modelo propio: entrene desde «Entrenamiento» o copie un checkpoint Hugging Face a <code>data/models/</code> y regístrelo (README → «Modelo propio en español»).</p></Card>}

      {tab === "storage" && cfg && <Card title="Almacenamiento"><div className="grid gap-4 md:grid-cols-3">
        <Chk cfg={cfg} setCfg={setCfg} path="storage.keep_normalized" label="Conservar audio normalizado" hint="Necesario para re-análisis y etiquetado rápido" />
        <Chk cfg={cfg} setCfg={setCfg} path="storage.temp_cleanup" label="Limpiar temporales al terminar" />
      </div><p className="mt-3 text-xs text-muted">Backend de almacenamiento (local / S3 / MinIO) se define por variables de entorno (<code>STORAGE_BACKEND</code>, <code>S3_*</code>); ver README.</p>{Actions}</Card>}

      {tab === "privacy" && cfg && <Card title="Privacidad y retención"><div className="grid gap-4 md:grid-cols-2">
        <Num cfg={cfg} setCfg={setCfg} path="privacy.retention_days" label="Retención (días)" hint="0 = no eliminar automáticamente; las llamadas vencidas se borran (audio, transcripción y resultados)" />
        <div className="space-y-3">
          <Chk cfg={cfg} setCfg={setCfg} path="privacy.redact_exports_by_default" label="Exportar transcripciones anonimizadas por defecto" />
          <Chk cfg={cfg} setCfg={setCfg} path="privacy.encrypt_at_rest" label="Cifrar audio en reposo (AES-256-GCM)" hint="Aplica a nuevas subidas; requiere reiniciar el servicio para el almacenamiento local" />
          <Chk cfg={cfg} setCfg={setCfg} path="privacy.default_allow_training" label="Permitir entrenamiento por defecto" hint="Debe permanecer DESACTIVADO salvo decisión explícita" />
        </div></div>
        <Notice tone="info"><span className="text-xs">El procesamiento es 100 % local: ninguna grabación se envía a servicios externos. Cada llamada tiene su propio permiso «utilizar para mejorar el modelo» (desactivado por defecto).</span></Notice>{Actions}</Card>}

      {tab === "users" && admin && (<Card title="Usuarios de la organización" actions={<Button variant="primary" size="sm" onClick={() => setShowUser(true)}>＋ Nuevo usuario</Button>} pad={false}>
        <table className="w-full text-sm"><thead><tr className="border-b border-line text-left text-xs text-muted">{["Nombre", "Correo", "Rol", "Estado", "Creado", ""].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{users.map((u) => (<tr key={u.id} className="border-b border-line/60">
            <td className="px-4 py-2">{u.name || "—"}</td><td className="px-4 py-2">{u.email}</td>
            <td className="px-4 py-2"><select value={u.role} onChange={async (e) => { try { await api(`/users/${u.id}`, { method: "PATCH", body: { role: e.target.value } }); setUsers(await api("/users")); toast("ok", "Rol actualizado."); } catch (x: any) { toast("err", x.message); } }} className="py-1 text-xs"><option>ADMIN</option><option>ANALYST</option><option>VIEWER</option></select></td>
            <td className="px-4 py-2">{u.is_active ? <Badge tone="good">activo</Badge> : <Badge>inactivo</Badge>}</td><td className="px-4 py-2 text-xs text-muted">{dateTime(u.created_at)}</td>
            <td className="px-4 py-2"><Button size="sm" onClick={async () => { try { await api(`/users/${u.id}`, { method: "PATCH", body: { is_active: !u.is_active } }); setUsers(await api("/users")); } catch (x: any) { toast("err", x.message); } }}>{u.is_active ? "Desactivar" : "Activar"}</Button></td></tr>))}</tbody></table>
        <p className="px-4 py-3 text-xs text-muted">ADMIN: todo · ANALYST: llamadas, datasets y entrenamiento · VIEWER: solo visualización.</p>
      </Card>)}
      <Modal open={showUser} onClose={() => setShowUser(false)} title="Nuevo usuario"><div className="space-y-3">
        <Field label="Nombre"><input className="w-full" value={nu.name} onChange={(e) => setNu({ ...nu, name: e.target.value })} /></Field>
        <Field label="Correo"><input className="w-full" type="email" value={nu.email} onChange={(e) => setNu({ ...nu, email: e.target.value })} /></Field>
        <Field label="Contraseña inicial" hint="Mínimo 8 caracteres"><input className="w-full" type="password" value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} /></Field>
        <Field label="Rol"><select className="w-full" value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })}><option>VIEWER</option><option>ANALYST</option><option>ADMIN</option></select></Field>
        <div className="flex justify-end gap-2"><Button onClick={() => setShowUser(false)}>Cancelar</Button><Button variant="primary" onClick={async () => { try { await api("/users", { body: nu }); setShowUser(false); setUsers(await api("/users")); toast("ok", "Usuario creado."); } catch (x: any) { toast("err", x.message); } }}>Crear</Button></div>
      </div></Modal>
    </div>
  );
}
