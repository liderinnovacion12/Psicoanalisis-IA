"use client";
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, downloadUrl } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { Button, Card, Empty, Modal, Pagination, Skeleton, StatusBadge } from "@/components/ui";
import { dateTime, mmss, num, satColor, shortId } from "@/lib/format";

const PAGE = 15;
export default function Calls() {
  const { toast, can } = useApp();
  const [q, setQ] = useState("");
  const [f, setF] = useState<any>({ status: "", date_from: "", date_to: "", min_duration: "", max_duration: "", min_satisfaction: "", max_satisfaction: "", speaker: "any", model: "" });
  const [page, setPage] = useState(1);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [del, setDel] = useState<any>(null);
  const [sort, setSort] = useState({ sort: "created_at", order: "desc" });

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const md = f.min_duration ? parseFloat(f.min_duration) * 60 : "", xd = f.max_duration ? parseFloat(f.max_duration) * 60 : "";
      setData(await api("/calls", { query: { q, ...f, min_duration: md, max_duration: xd, ...sort, page, page_size: PAGE } }));
    } catch (e: any) { toast("err", e.message); } finally { setLoading(false); }
  }, [q, f, page, sort, toast]);
  useEffect(() => { const t = setTimeout(load, 250); return () => clearTimeout(t); }, [load]);
  // refresco automático si hay llamadas en proceso
  useEffect(() => {
    if (!data?.items?.some((c: any) => !["COMPLETED", "ERROR"].includes(c.status))) return;
    const t = setInterval(load, 4000); return () => clearInterval(t);
  }, [data, load]);

  const th = (label: string, key?: string) => (
    <th className="px-3 py-2 font-medium">{key ? <button className="hover:text-ink" onClick={() => setSort({ sort: key, order: sort.sort === key && sort.order === "desc" ? "asc" : "desc" })}>{label}{sort.sort === key ? (sort.order === "desc" ? " ↓" : " ↑") : ""}</button> : label}</th>);
  const set = (k: string, v: string) => { setF({ ...f, [k]: v }); setPage(1); };

  async function remove() {
    try { await api(`/calls/${del.id}`, { method: "DELETE" }); toast("ok", "Llamada eliminada."); setDel(null); load(); } catch (e: any) { toast("err", e.message); }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between"><div><h1 className="text-2xl font-semibold">Llamadas</h1><p className="text-sm text-muted">Historial de grabaciones analizadas</p></div>
        {can("ANALYST") && <Link href="/"><Button variant="primary">＋ Nueva llamada</Button></Link>}</div>
      <Card>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
          <input className="sm:col-span-2" placeholder="Buscar por ID, nombre, archivo o fecha…" value={q} onChange={(e) => { setQ(e.target.value); setPage(1); }} aria-label="Buscar" />
          <select value={f.status} onChange={(e) => set("status", e.target.value)} aria-label="Estado"><option value="">Estado: todos</option>{["COMPLETED", "ERROR", "QUEUED", "PROCESSING_AUDIO", "DIARIZING", "TRANSCRIBING", "ANALYZING_EMOTIONS", "CALCULATING_SATISFACTION"].map((s) => <option key={s} value={s}>{s}</option>)}</select>
          <input type="date" value={f.date_from} onChange={(e) => set("date_from", e.target.value)} aria-label="Desde" title="Desde" />
          <input type="date" value={f.date_to} onChange={(e) => set("date_to", e.target.value)} aria-label="Hasta" title="Hasta" />
          <select value={f.speaker} onChange={(e) => set("speaker", e.target.value)} aria-label="Speaker"><option value="any">Speaker: cualquiera</option><option value="SPEAKER_00">Persona 1</option><option value="SPEAKER_01">Persona 2</option></select>
          <input type="number" min={0} placeholder="Duración mín. (min)" value={f.min_duration} onChange={(e) => set("min_duration", e.target.value)} />
          <input type="number" min={0} placeholder="Duración máx. (min)" value={f.max_duration} onChange={(e) => set("max_duration", e.target.value)} />
          <input type="number" min={0} max={100} placeholder="Satisf. mín." value={f.min_satisfaction} onChange={(e) => set("min_satisfaction", e.target.value)} />
          <input type="number" min={0} max={100} placeholder="Satisf. máx." value={f.max_satisfaction} onChange={(e) => set("max_satisfaction", e.target.value)} />
          <input placeholder="Modelo (versión)" value={f.model} onChange={(e) => set("model", e.target.value)} />
          <Button variant="ghost" onClick={() => { setF({ status: "", date_from: "", date_to: "", min_duration: "", max_duration: "", min_satisfaction: "", max_satisfaction: "", speaker: "any", model: "" }); setQ(""); }}>Limpiar filtros</Button>
        </div>
      </Card>
      <Card pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead><tr className="border-b border-line text-left text-xs text-muted">{th("ID")}{th("Archivo", "filename")}{th("Fecha", "created_at")}{th("Duración", "duration")}{th("Estado", "status")}{th("Satisfacción P1", "satisfaction_p1")}{th("Satisfacción P2", "satisfaction_p2")}{th("Modelo")}{th("Acciones")}</tr></thead>
            <tbody>
              {loading && !data ? Array.from({ length: 6 }).map((_, i) => <tr key={i}><td colSpan={9} className="p-3"><Skeleton className="h-6" /></td></tr>) :
                data?.items.map((c: any) => (
                  <tr key={c.id} className="border-b border-line/60 hover:bg-surface2/60">
                    <td className="px-3 py-2.5"><Link className="text-brand" href={`/calls/${c.id}`}>{shortId(c.id)}</Link></td>
                    <td className="max-w-[240px] truncate px-3 py-2.5" title={c.filename}>{c.display_name || c.filename}</td>
                    <td className="px-3 py-2.5 text-muted">{dateTime(c.created_at)}</td>
                    <td className="px-3 py-2.5 tabular-nums">{mmss(c.duration)}</td>
                    <td className="px-3 py-2.5"><StatusBadge status={c.status} />{c.status !== "COMPLETED" && c.status !== "ERROR" && <span className="ml-1 text-xs text-muted">{Math.round(c.progress)}%</span>}</td>
                    <td className={`px-3 py-2.5 font-medium tabular-nums ${satColor(c.satisfaction_p1)}`}>{num(c.satisfaction_p1)}</td>
                    <td className={`px-3 py-2.5 font-medium tabular-nums ${satColor(c.satisfaction_p2)}`}>{num(c.satisfaction_p2)}</td>
                    <td className="max-w-[140px] truncate px-3 py-2.5 text-xs text-muted">{c.model_version || "—"}</td>
                    <td className="whitespace-nowrap px-3 py-2.5">
                      <Link href={`/calls/${c.id}`}><Button size="sm">Ver</Button></Link>{" "}
                      {c.status === "COMPLETED" && (<><a href={downloadUrl(`/reports/${c.id}`)}><Button size="sm">Reporte</Button></a>{" "}<a href={downloadUrl(`/calls/${c.id}/export`, { format: "csv" })}><Button size="sm">Exportar</Button></a>{" "}</>)}
                      {can("ANALYST") && <Button size="sm" variant="danger" onClick={() => setDel(c)}>Eliminar</Button>}
                    </td>
                  </tr>))}
            </tbody>
          </table>
        </div>
        {data && data.items.length === 0 && <Empty title="Sin resultados" text="Ajuste los filtros o suba una nueva llamada." />}
        <div className="px-4 pb-3"><Pagination page={page} pageSize={PAGE} total={data?.total ?? 0} onPage={setPage} /></div>
      </Card>
      <Modal open={!!del} onClose={() => setDel(null)} title="Eliminar llamada">
        <p className="text-sm">Se eliminarán de forma permanente el audio, la transcripción, los resultados y las muestras de dataset derivadas de <b>{del?.display_name || del?.filename}</b>. Esta acción no se puede deshacer.</p>
        <div className="mt-4 flex justify-end gap-2"><Button onClick={() => setDel(null)}>Cancelar</Button><Button variant="danger" onClick={remove}>Eliminar definitivamente</Button></div>
      </Modal>
    </div>
  );
}
