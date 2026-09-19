"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import EChart, { axisStyle } from "@/components/EChart";
import { Card, Empty, Skeleton, Stat, StatusBadge, Button } from "@/components/ui";
import { EMO_ES, emoColor, mmss, num, satColor, shortId, dateTime } from "@/lib/format";

export default function Dashboard() {
  const { theme, can } = useApp();
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api("/dashboard/summary").then(setD).catch((e) => setErr(e.message)); }, []);
  const dark = theme === "dark";
  const ax = axisStyle(dark);

  const evolution = useMemo(() => d && ({
    grid: { left: 40, right: 12, top: 24, bottom: 28 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: d.satisfaction_evolution.map((x: any) => x.date), ...ax },
    yAxis: { type: "value", min: 0, max: 100, splitLine: ax.splitLine, axisLabel: ax.axisLabel },
    series: [{ type: "line", smooth: true, data: d.satisfaction_evolution.map((x: any) => x.avg), lineStyle: { width: 3, color: "#0E9AA7" }, areaStyle: { opacity: 0.1, color: "#0E9AA7" }, itemStyle: { color: "#0E9AA7" } }],
  }), [d, dark]);
  const emotions = useMemo(() => d && ({
    grid: { left: 40, right: 12, top: 16, bottom: 28 },
    tooltip: { trigger: "axis", valueFormatter: (v: number) => `${v.toFixed(1)}%` },
    xAxis: { type: "category", data: Object.keys(d.avg_emotions).map((k) => EMO_ES[k] || k), ...ax },
    yAxis: { type: "value", splitLine: ax.splitLine, axisLabel: { ...ax.axisLabel, formatter: "{value}%" } },
    series: [{ type: "bar", data: Object.entries(d.avg_emotions).map(([k, v]: any) => ({ value: +(v * 100).toFixed(1), itemStyle: { color: emoColor(k), borderRadius: [4, 4, 0, 0] } })) }],
  }), [d, dark]);
  const dist = useMemo(() => d && ({
    grid: { left: 40, right: 12, top: 16, bottom: 28 }, tooltip: {},
    xAxis: { type: "category", data: Object.keys(d.satisfaction_distribution), ...ax },
    yAxis: { type: "value", minInterval: 1, splitLine: ax.splitLine, axisLabel: ax.axisLabel },
    series: [{ type: "bar", data: Object.values(d.satisfaction_distribution).map((v, i) => ({ value: v, itemStyle: { color: ["#D64545", "#E0762A", "#E0A030", "#5FB878", "#2E9E5B"][i], borderRadius: [4, 4, 0, 0] } })) }],
  }), [d, dark]);
  const domin = useMemo(() => d && ({
    tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
    series: [{ type: "pie", radius: ["48%", "72%"], label: { color: dark ? "#8B98A8" : "#64748B", fontSize: 11 }, data: Object.entries(d.dominant_emotions).map(([k, v]) => ({ name: EMO_ES[k] || k, value: v, itemStyle: { color: emoColor(k) } })) }],
  }), [d, dark]);

  if (err) return <Card><Empty title="No se pudo cargar el dashboard" text={err} /></Card>;
  const t = d?.totals;
  const noData = d && t.analyzed_calls === 0;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div><h1 className="text-2xl font-semibold">Dashboard</h1><p className="text-sm text-muted">Resumen de los últimos {d?.days ?? 90} días · valores estimados por los modelos</p></div>
        {can("ANALYST") && <Link href="/calls/new"><Button variant="primary">＋ Nueva llamada</Button></Link>}
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <Stat label="Total de llamadas" value={t?.total_calls} loading={!d} />
        <Stat label="Llamadas analizadas" value={t?.analyzed_calls} hint={t?.processing ? `${t.processing} en proceso` : undefined} loading={!d} />
        <Stat label="Satisfacción promedio" value={t?.avg_satisfaction != null ? `${t.avg_satisfaction}/100` : "—"} tone={satColor(t?.avg_satisfaction)} loading={!d} />
        <Stat label="Frustración promedio" value={t?.avg_frustration != null ? `${t.avg_frustration}%` : "—"} loading={!d} />
        <Stat label="Duración promedio" value={t?.avg_duration != null ? mmss(t.avg_duration) : "—"} loading={!d} />
        <Stat label="Calidad promedio" value={t?.avg_quality != null ? `${t.avg_quality}/100` : "—"} loading={!d} />
      </div>
      {noData ? (
        <Card><Empty title="Aún no hay llamadas analizadas" text="Suba una grabación para ver aquí la satisfacción, las emociones y su evolución." action={<Link href="/calls/new"><Button variant="primary">Subir llamada</Button></Link>} /></Card>
      ) : (
        <>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Satisfacción promedio" subtitle="Evolución diaria (promedio de participantes)">{evolution ? <EChart option={evolution as any} height={250} /> : <Skeleton className="h-60" />}</Card>
            <Card title="Emociones" subtitle="Probabilidad media por emoción (todas las llamadas)">{emotions ? <EChart option={emotions as any} height={250} /> : <Skeleton className="h-60" />}</Card>
            <Card title="Distribución de satisfacción" subtitle="Nº de participantes por rango">{dist ? <EChart option={dist as any} height={250} /> : <Skeleton className="h-60" />}</Card>
            <Card title="Emoción predominante" subtitle="Por participante">{domin ? <EChart option={domin as any} height={250} /> : <Skeleton className="h-60" />}</Card>
          </div>
          <Card title="Llamadas recientes" actions={<Link href="/calls" className="text-xs text-brand">Ver todas →</Link>} pad={false}>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead><tr className="border-b border-line text-left text-xs text-muted">{["ID", "Archivo", "Fecha", "Duración", "Estado", "Satisf. P1", "Satisf. P2"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
                <tbody>
                  {(d?.recent || []).map((c: any) => (
                    <tr key={c.id} className="border-b border-line/60 hover:bg-surface2/60">
                      <td className="px-4 py-2"><Link href={`/calls/${c.id}`} className="text-brand">{shortId(c.id)}</Link></td>
                      <td className="max-w-[220px] truncate px-4 py-2">{c.display_name || c.filename}</td><td className="px-4 py-2 text-muted">{dateTime(c.created_at)}</td>
                      <td className="px-4 py-2 tabular-nums">{mmss(c.duration)}</td><td className="px-4 py-2"><StatusBadge status={c.status} /></td>
                      <td className={`px-4 py-2 tabular-nums font-medium ${satColor(c.satisfaction_p1)}`}>{num(c.satisfaction_p1)}</td>
                      <td className={`px-4 py-2 tabular-nums font-medium ${satColor(c.satisfaction_p2)}`}>{num(c.satisfaction_p2)}</td>
                    </tr>))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
