"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, Empty, Skeleton, Button } from "@/components/ui";
import { dateTime, mmss, num, satColor, shortId } from "@/lib/format";

// Resultados: acceso rápido a las llamadas ya analizadas (ordenadas por fecha) con sus métricas clave.
export default function Results() {
  const [d, setD] = useState<any>(null);
  useEffect(() => { api("/calls", { query: { status: "COMPLETED", page_size: 30 } }).then(setD).catch(() => setD({ items: [] })); }, []);
  return (
    <div className="space-y-5">
      <div><h1 className="text-2xl font-semibold">Resultados</h1><p className="text-sm text-muted">Llamadas analizadas · abra una para ver diarización, transcripción, emociones, satisfacción y eventos</p></div>
      {!d ? <Skeleton className="h-64" /> : d.items.length === 0 ? <Card><Empty title="Aún no hay resultados" text="Cuando una llamada termine de procesarse aparecerá aquí." action={<Link href="/calls/new"><Button variant="primary">Subir llamada</Button></Link>} /></Card> : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {d.items.map((c: any) => (
            <Link key={c.id} href={`/calls/${c.id}`} className="rounded-xl border border-line bg-surface p-4 shadow-card transition hover:border-brand">
              <p className="truncate font-medium">{c.display_name || c.filename}</p>
              <p className="text-xs text-muted">{shortId(c.id)} · {dateTime(c.created_at)} · {mmss(c.duration)}</p>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                <div><p className="text-[11px] text-muted">Persona 1</p><p className={`text-xl font-semibold tabular-nums ${satColor(c.satisfaction_p1)}`}>{num(c.satisfaction_p1)}</p></div>
                <div><p className="text-[11px] text-muted">Persona 2</p><p className={`text-xl font-semibold tabular-nums ${satColor(c.satisfaction_p2)}`}>{num(c.satisfaction_p2)}</p></div>
                <div><p className="text-[11px] text-muted">Calidad</p><p className="text-xl font-semibold tabular-nums">{num(c.analysis_quality)}</p></div>
              </div>
            </Link>))}
        </div>)}
    </div>
  );
}
