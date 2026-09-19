"use client";
import { usePlayer } from "@/lib/player";
import { mmss } from "@/lib/format";
import { Button } from "../ui";

const RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];

export default function AudioPlayer({ duration }: { duration: number }) {
  const p = usePlayer();
  const total = p.duration || duration || 0;
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface px-4 py-3 shadow-card">
      <Button variant="primary" onClick={p.toggle} className="w-20" aria-label={p.playing ? "Pausar" : "Reproducir"}>{p.playing ? "❚❚ Pausa" : "▶ Play"}</Button>
      <span className="w-12 text-right text-xs tabular-nums text-muted">{mmss(p.time)}</span>
      <input type="range" min={0} max={Math.max(total, 1)} step={0.1} value={Math.min(p.time, total)} onChange={(e) => p.seek(parseFloat(e.target.value), false)} className="min-w-[180px] flex-1" aria-label="Posición" />
      <span className="w-12 text-xs tabular-nums text-muted">{mmss(total)}</span>
      <label className="flex items-center gap-1.5 text-xs text-muted">Vol
        <input type="range" min={0} max={1} step={0.05} value={p.volume} onChange={(e) => p.setVolume(parseFloat(e.target.value))} className="w-20" aria-label="Volumen" />
      </label>
      <label className="flex items-center gap-1.5 text-xs text-muted">Velocidad
        <select value={p.rate} onChange={(e) => p.setRate(parseFloat(e.target.value))} className="py-1 text-xs" aria-label="Velocidad">
          {RATES.map((r) => <option key={r} value={r}>{r}x</option>)}
        </select>
      </label>
    </div>
  );
}
