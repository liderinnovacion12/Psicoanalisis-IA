"use client";
import { useMemo, useState } from "react";
import EChart, { axisStyle } from "../EChart";
import { useApp } from "@/lib/app-context";
import { usePlayer } from "@/lib/player";
import { EMOTIONS, EMO_ES, emoColor, mmss, SPK_COLOR } from "@/lib/format";
import { Tabs } from "../ui";

type Who = "SPEAKER_00" | "SPEAKER_01" | "both";

export function EmotionChart({ emotions, names, duration }: { emotions: any[]; names: Record<string, string>; duration: number }) {
  const { theme } = useApp();
  const player = usePlayer();
  const [who, setWho] = useState<Who>("SPEAKER_00");
  const labels = useMemo(() => (emotions[0] ? Object.keys(emotions[0].probabilities) : [...EMOTIONS]), [emotions]);

  const option = useMemo(() => {
    const dark = theme === "dark";
    const ax = axisStyle(dark);
    const spks = who === "both" ? ["SPEAKER_00", "SPEAKER_01"] : [who];
    const series: any[] = [];
    spks.forEach((sp, si) => {
      const pts = emotions.filter((e) => e.speaker === sp);
      labels.forEach((l) => {
        series.push({
          name: who === "both" ? `${names[sp] || sp} · ${EMO_ES[l] || l}` : EMO_ES[l] || l,
          type: "line", showSymbol: false, smooth: 0.25, sampling: "lttb",
          lineStyle: { width: 1.8, type: si === 1 ? "dashed" : "solid" }, itemStyle: { color: emoColor(l) },
          data: pts.map((p) => [(p.start + p.end) / 2, +(p.probabilities[l] ?? 0).toFixed(4), p.start, p.end, p.emotion, p.confidence]),
        });
      });
    });
    series.push({ type: "line", markLine: { silent: true, symbol: "none", lineStyle: { color: "#0E9AA7", width: 1.5 }, label: { formatter: mmss(player.time), color: "#0E9AA7" }, data: [{ xAxis: player.time }] }, data: [] });
    return {
      legend: { top: 0, textStyle: { color: dark ? "#8B98A8" : "#64748B", fontSize: 11 }, type: "scroll" },
      tooltip: { trigger: "axis", axisPointer: { type: "cross" }, formatter: (ps: any) => {
        if (!Array.isArray(ps) || !ps.length) return "";
        const t = ps[0].value?.[0];
        const rows = ps.filter((p: any) => p.value).map((p: any) => `<div>${p.marker} ${p.seriesName}: <b>${(p.value[1] * 100).toFixed(0)}%</b></div>`).join("");
        return `<b>${mmss(t)}</b>${rows}<div style="color:#8B98A8;margin-top:4px">Clic: reproducir desde aquí</div>`;
      } },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 4, borderColor: "transparent" }],
      xAxis: { type: "value", min: 0, max: duration || undefined, ...ax, axisLabel: { formatter: (v: number) => mmss(v), color: dark ? "#8B98A8" : "#64748B", fontSize: 11 } },
      yAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: (v: number) => `${v * 100}%`, color: dark ? "#8B98A8" : "#64748B" }, splitLine: ax.splitLine },
      series,
      grid: { left: 48, right: 16, top: 44, bottom: 52 },
    } as any;
  }, [emotions, who, theme, labels, names, duration, player.time]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <Tabs tabs={[{ key: "SPEAKER_00" as Who, label: names.SPEAKER_00 || "Persona 1" }, { key: "SPEAKER_01" as Who, label: names.SPEAKER_01 || "Persona 2" }, { key: "both" as Who, label: "Ambas" }]} value={who} onChange={setWho} />
        <span className="text-xs text-muted">Zoom: rueda del ratón / barra inferior</span>
      </div>
      <EChart option={option} height={330} onClick={(p) => p?.value?.[0] != null && player.seek(p.value[2] ?? p.value[0])} />
    </div>
  );
}

export function SatisfactionChart({ sat, events, names, duration }: { sat: any; events: any[]; names: Record<string, string>; duration: number }) {
  const { theme } = useApp();
  const player = usePlayer();
  const option = useMemo(() => {
    const dark = theme === "dark";
    const ax = axisStyle(dark);
    const series: any[] = Object.entries(sat.speakers || {}).map(([sp, s]: any, i) => ({
      name: names[sp] || sp, type: "line", showSymbol: false, smooth: 0.3,
      lineStyle: { width: 2.6, color: SPK_COLOR[i] }, itemStyle: { color: SPK_COLOR[i] },
      areaStyle: { color: SPK_COLOR[i], opacity: 0.07 },
      data: (s.timeline || []).map((p: any) => [p.t, p.score]),
      markPoint: {
        symbolSize: 34, label: { fontSize: 10, color: "#fff" }, itemStyle: { color: SPK_COLOR[i] },
        data: [{ type: "max", name: "Pico" }, { type: "min", name: "Mínimo" }],
      },
    }));
    series.push({
      type: "scatter", name: "Eventos", symbol: "diamond", symbolSize: 12,
      itemStyle: { color: dark ? "#F0B046" : "#D6911E" },
      data: events.filter((e) => e.speaker && e.satisfaction != null).map((e) => ({ value: [e.timestamp, e.satisfaction], name: e.label })),
      tooltip: { formatter: (p: any) => `<b>${p.name}</b><br/>${mmss(p.value[0])}` },
    });
    series.push({ type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { color: "#0E9AA7" }, label: { formatter: mmss(player.time), color: "#0E9AA7" }, data: [{ xAxis: player.time }] } });
    return {
      legend: { top: 0, textStyle: { color: dark ? "#8B98A8" : "#64748B", fontSize: 11 } },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (typeof v === "number" ? v.toFixed(0) : v) },
      xAxis: { type: "value", min: 0, max: duration || undefined, ...ax, axisLabel: { formatter: (v: number) => mmss(v), color: dark ? "#8B98A8" : "#64748B", fontSize: 11 } },
      yAxis: { type: "value", min: 0, max: 100, splitLine: ax.splitLine, axisLabel: { color: dark ? "#8B98A8" : "#64748B" } },
      visualMap: undefined,
      dataZoom: [{ type: "inside" }],
      series: [
        { type: "line", data: [], markArea: { silent: true, data: [[{ yAxis: 0, itemStyle: { color: "rgba(214,69,69,0.06)" } }, { yAxis: 40 }], [{ yAxis: 60, itemStyle: { color: "rgba(46,158,91,0.06)" } }, { yAxis: 100 }]] } },
        ...series,
      ],
      grid: { left: 48, right: 16, top: 40, bottom: 34 },
    } as any;
  }, [sat, events, names, theme, duration, player.time]);
  return <EChart option={option} height={300} onClick={(p) => p?.value?.[0] != null && player.seek(p.value[0])} />;
}

/** Línea de tiempo: emoción dominante por segmento y hablante, eventos y satisfacción resumida. */
export function Timeline({ emotions, events, names, duration }: { emotions: any[]; events: any[]; names: Record<string, string>; duration: number }) {
  const player = usePlayer();
  const total = Math.max(duration, 1);
  const rows = ["SPEAKER_00", "SPEAKER_01"];
  return (
    <div className="space-y-2">
      {rows.map((sp) => (
        <div key={sp} className="flex items-center gap-3">
          <span className="w-24 shrink-0 truncate text-xs font-medium" style={{ color: SPK_COLOR[rows.indexOf(sp)] }}>{names[sp] || sp}</span>
          <div className="relative h-6 flex-1 overflow-hidden rounded bg-surface2">
            {emotions.filter((e) => e.speaker === sp).map((e) => (
              <button key={e.id} title={`${mmss(e.start)} · ${EMO_ES[e.emotion] || e.emotion} (${Math.round(e.confidence * 100)}%)`}
                onClick={() => player.seek(e.start)} className="absolute top-0 h-full opacity-90 hover:opacity-100"
                style={{ left: `${(e.start / total) * 100}%`, width: `${Math.max(((e.end - e.start) / total) * 100, 0.3)}%`, background: emoColor(e.emotion) }} />
            ))}
          </div>
        </div>
      ))}
      <div className="flex items-center gap-3">
        <span className="w-24 shrink-0 text-xs font-medium text-muted">Eventos</span>
        <div className="relative h-6 flex-1 rounded bg-surface2">
          {events.map((e) => (
            <button key={e.id} title={`${mmss(e.timestamp)} · ${e.label}`} onClick={() => player.seek(e.timestamp)} className="absolute -translate-x-1/2 text-[13px] leading-6" style={{ left: `${(e.timestamp / total) * 100}%` }}>
              {["recovery", "positive_peak", "positive_ending"].includes(e.event_type) ? "↑" : e.event_type === "possible_conflict" ? "⚡" : "⚠"}
            </button>
          ))}
        </div>
      </div>
      <div className="relative ml-[108px] h-1 rounded bg-line">
        <div className="absolute top-[-3px] h-2.5 w-0.5 bg-brand" style={{ left: `${Math.min(100, (player.time / total) * 100)}%` }} />
      </div>
      <div className="ml-[108px] flex flex-wrap gap-x-3 gap-y-1 pt-1 text-[11px] text-muted">
        {EMOTIONS.map((e) => <span key={e} className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-sm" style={{ background: emoColor(e) }} />{EMO_ES[e]}</span>)}
      </div>
    </div>
  );
}
