"use client";
import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { uploadWithProgress } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { Button, Card, Notice, Progress } from "@/components/ui";
import { bytes, mmss } from "@/lib/format";

const EXT = ["mp3", "wav", "m4a", "aac", "flac", "ogg"];

export default function NewCall() {
  const router = useRouter();
  const { toast, can } = useApp();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [allow, setAllow] = useState(false);
  const [name, setName] = useState("");
  const [pct, setPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<any>(null);
  const [err, setErr] = useState("");

  function pick(f?: File | null) {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase() || "";
    if (!EXT.includes(ext)) { setErr("Formato no soportado. Use MP3, WAV, M4A, AAC, FLAC u OGG."); return; }
    setErr(""); setFile(f);
  }
  async function submit() {
    if (!file) return;
    setBusy(true); setErr(""); setPct(0);
    const fd = new FormData();
    fd.append("file", file); fd.append("allow_training", String(allow)); if (name) fd.append("display_name", name);
    try {
      const r = await uploadWithProgress("/calls/upload", fd, setPct);
      setInfo(r);
      toast("ok", "Llamada subida. El análisis continúa en segundo plano.");
      setTimeout(() => router.push(`/calls/${r.id}`), 900);
    } catch (e: any) { setErr(e.message); setBusy(false); }
  }
  if (!can("ANALYST")) return <Notice tone="warn">Su rol solo permite visualizar resultados.</Notice>;

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div className="text-center"><h1 className="text-2xl font-semibold">ANALIZADOR DE LLAMADAS IA</h1><p className="text-sm text-muted">Emociones y satisfacción estimadas · procesamiento local</p></div>
      <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
        className={`flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition ${drag ? "border-brand bg-brand/10" : "border-line bg-surface"}`}>
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-brand/15 text-2xl text-brand">⇪</div>
        <p className="font-medium">Arrastra tu grabación aquí</p><p className="text-xs text-muted">O</p>
        <Button variant="primary" onClick={() => input.current?.click()} disabled={busy}>SUBIR LLAMADA</Button>
        <p className="text-xs text-muted">MP3 • WAV • M4A • AAC • FLAC • OGG · Se admiten llamadas muy largas (horas)</p>
        <input ref={input} type="file" hidden accept=".mp3,.wav,.m4a,.aac,.flac,.ogg,audio/*" onChange={(e) => pick(e.target.files?.[0])} />
      </div>
      {err && <Notice tone="bad">{err}</Notice>}
      {file && (
        <Card title="Archivo seleccionado">
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <div><dt className="text-xs text-muted">Archivo</dt><dd className="truncate font-medium">{file.name}</dd></div>
            <div><dt className="text-xs text-muted">Tamaño</dt><dd className="font-medium">{bytes(file.size)}</dd></div>
            {info && <div><dt className="text-xs text-muted">Duración</dt><dd className="font-medium">{mmss(info.duration)}</dd></div>}
            <div><dt className="text-xs text-muted">Estado</dt><dd className="font-medium">{busy ? (pct < 100 ? "Subiendo…" : info ? "Procesando…" : "Validando…") : "Listo para subir"}</dd></div>
          </dl>
          <input className="mt-4 w-full" placeholder="Nombre para identificar la llamada (opcional)" value={name} onChange={(e) => setName(e.target.value)} disabled={busy} />
          <label className="mt-4 flex items-start gap-2 text-sm"><input type="checkbox" className="mt-0.5" checked={allow} onChange={(e) => setAllow(e.target.checked)} disabled={busy} />
            <span>Permitir utilizar esta llamada para mejorar el modelo<br /><span className="text-xs text-muted">Desactivado por defecto. Las grabaciones nunca se envían a servicios externos ni se usan para entrenar sin este permiso.</span></span></label>
          {busy && <div className="mt-4"><Progress value={pct} /><p className="mt-1 text-xs text-muted">{Math.round(pct)}%</p></div>}
          <div className="mt-4 flex justify-end gap-2"><Button onClick={() => { setFile(null); setInfo(null); }} disabled={busy}>Quitar</Button><Button variant="primary" onClick={submit} disabled={busy}>{busy ? "Procesando…" : "Analizar llamada"}</Button></div>
        </Card>)}
    </div>
  );
}
