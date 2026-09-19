"use client";
import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { useApp } from "@/lib/app-context";

// Wrapper mínimo de ECharts: tema claro/oscuro, redimensionado y eventos click.
export default function EChart({ option, height = 280, onClick, onReady, className }: {
  option: echarts.EChartsOption; height?: number; onClick?: (p: any) => void; onReady?: (c: echarts.ECharts) => void; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const { theme } = useApp();

  useEffect(() => {
    if (!ref.current) return;
    chart.current?.dispose();
    const c = echarts.init(ref.current, undefined, { renderer: "canvas" });
    chart.current = c;
    onReady?.(c);
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); c.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const dark = theme === "dark";
    const txt = dark ? "#8B98A8" : "#64748B";
    const line = dark ? "#2A3441" : "#E2E8F0";
    c.setOption({
      backgroundColor: "transparent",
      textStyle: { color: txt, fontFamily: "Inter, system-ui, sans-serif" },
      grid: { left: 44, right: 16, top: 32, bottom: 44, containLabel: false },
      tooltip: { backgroundColor: dark ? "#141A22" : "#fff", borderColor: line, textStyle: { color: dark ? "#E8EDF3" : "#0F172A", fontSize: 12 } },
      ...option,
    } as any, true);
    c.off("click");
    if (onClick) c.on("click", onClick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [option, theme, onClick]);

  return <div ref={ref} style={{ height }} className={className} />;
}

export const axisStyle = (dark: boolean) => ({
  axisLine: { lineStyle: { color: dark ? "#2A3441" : "#CBD5E1" } },
  splitLine: { lineStyle: { color: dark ? "#222B37" : "#EEF2F6" } },
  axisLabel: { color: dark ? "#8B98A8" : "#64748B", fontSize: 11 },
});
