"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, downloadUrl } from "@/lib/api";
import { Button, Card, Empty, Notice, Pagination, Skeleton } from "@/components/ui";
import { dateTime, mmss, num, shortId } from "@/lib/format";

export default function Reports() {
  const [d, setD] = useState<any>(null);
  const [page, setPage] = useState(1);
  const [redact, setRedact] = useState(false);
  useEffect(() => { api("/calls", { query: { status: "COMPLETED", page, page_size: 15 } }).then(setD).catch(() => setD({ items: [], total: 0 })); }, [page]);
  const q = (format: string) => ({ format, ...(redact ? { redact: "true" } : {}) });
  return (
    <div className="space-y-5">
      <div><h1 className="text-2xl font-semibold">Reportes y exportación</h1><p className="text-sm text-muted">PDF profesional, CSV, Excel y JSON completo de cada llamada analizada</p></div>
      <Notice tone="info">
        <label className="flex items-center gap-2"><input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} />
          Anonimizar la transcripción exportada (nombres, teléfonos, correos, documentos, direcciones y cuentas → [PERSONA], [TELEFONO], …)</label>
      </Notice>
      <Card pad={false}>
        {!d ? <div className="p-5"><Skeleton className="h-40" /></div> : d.items.length === 0 ? <Empty title="Aún no hay llamadas analizadas" /> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead><tr className="border-b border-line text-left text-xs text-muted">{["ID", "Llamada", "Fecha", "Duración", "Calidad", "Descargar"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
              <tbody>
                {d.items.map((c: any) => (
                  <tr key={c.id} className="border-b border-line/60 hover:bg-surface2/60">
                    <td className="px-4 py-2.5"><Link href={`/calls/${c.id}`} className="text-brand">{shortId(c.id)}</Link></td>
                    <td className="max-w-[260px] truncate px-4 py-2.5">{c.display_name || c.filename}</td>
                    <td className="px-4 py-2.5 text-muted">{dateTime(c.created_at)}</td><td className="px-4 py-2.5 tabular-nums">{mmss(c.duration)}</td>
                    <td className="px-4 py-2.5 tabular-nums">{num(c.analysis_quality)}/100</td>
                    <td className="space-x-1.5 whitespace-nowrap px-4 py-2.5">
                      <a href={downloadUrl(`/reports/${c.id}`, redact ? { redact: "true" } : {})}><Button size="sm" variant="primary">PDF</Button></a>
                      {["csv", "xlsx", "json"].map((f) => <a key={f} href={downloadUrl(`/calls/${c.id}/export`, q(f))}><Button size="sm">{f.toUpperCase()}</Button></a>)}
                    </td>
                  </tr>))}
              </tbody>
            </table>
          </div>)}
        <div className="px-4 pb-3"><Pagination page={page} pageSize={15} total={d?.total ?? 0} onPage={setPage} /></div>
      </Card>
    </div>
  );
}
