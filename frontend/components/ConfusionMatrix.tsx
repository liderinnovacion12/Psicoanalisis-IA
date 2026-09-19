"use client";
import { EMO_ES } from "@/lib/format";

// Matriz de confusión interactiva: filas = etiqueta real, columnas = predicción. Hover muestra el detalle.
export default function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  if (!matrix?.length) return <p className="text-sm text-muted">Sin matriz de confusión disponible.</p>;
  const rowMax = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr><th className="px-2 text-right text-muted">Real ↓ / Pred →</th>{labels.map((l) => <th key={l} className="px-2 py-1 font-medium text-muted" title={EMO_ES[l] || l}>{(EMO_ES[l] || l).slice(0, 5)}</th>)}<th className="px-2 text-muted">n</th></tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="whitespace-nowrap px-2 text-right font-medium text-muted">{EMO_ES[labels[i]] || labels[i]}</th>
              {row.map((v, j) => {
                const p = rowMax[i] ? v / rowMax[i] : 0;
                const diag = i === j;
                return (
                  <td key={j} title={`Real: ${EMO_ES[labels[i]] || labels[i]} → Predicho: ${EMO_ES[labels[j]] || labels[j]}\n${v} muestras (${(p * 100).toFixed(0)}% de la fila)`}
                    className="h-9 w-12 rounded text-center font-medium tabular-nums transition hover:ring-2 hover:ring-brand"
                    style={{ background: `${diag ? "rgba(46,158,91," : "rgba(208,65,65,"}${Math.min(0.08 + p * 0.85, 0.95)})`, color: p > 0.5 ? "#fff" : undefined }}>
                    {v}
                  </td>);
              })}
              <td className="px-2 text-center text-muted">{rowMax[i]}</td>
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}
