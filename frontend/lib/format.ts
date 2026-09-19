export const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] as const;
export const EMO_ES: Record<string, string> = {
  angry: "Enojo", disgust: "Disgusto", fear: "Miedo", happy: "Alegría", neutral: "Neutral", sad: "Tristeza", surprise: "Sorpresa",
};
export const EMO_COLOR: Record<string, string> = {
  angry: "#D64545", disgust: "#9A7B3A", fear: "#8B5FBF", happy: "#2E9E5B", neutral: "#8C99A6", sad: "#3B7DD8", surprise: "#E0A030",
};
export const SPK_COLOR = ["#0E9AA7", "#E0762A"];
export const emoLabel = (e?: string | null) => (e ? EMO_ES[e] || e : "—");
export const emoColor = (e: string) => EMO_COLOR[e] || "#6B7A90";

export function mmss(t?: number | null): string {
  if (t == null || isNaN(t)) return "—";
  t = Math.max(0, Math.round(t));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${p(h)}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}
export const hms = (t?: number | null) => {
  if (t == null) return "—";
  t = Math.round(t);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(Math.floor(t / 3600))}:${p(Math.floor((t % 3600) / 60))}:${p(t % 60)}`;
};
export const pct = (v?: number | null, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v?: number | null, d = 0) => (v == null ? "—" : v.toFixed(d));
export const dateTime = (s?: string | null) => (s ? new Date(s).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" }) : "—");
export const bytes = (n?: number | null) => {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};
export const shortId = (id: string) => id.slice(0, 8);

export const STATUS_ES: Record<string, string> = {
  UPLOADED: "Subida", QUEUED: "En cola", PROCESSING_AUDIO: "Procesando audio", DIARIZING: "Diarizando",
  TRANSCRIBING: "Transcribiendo", ANALYZING_EMOTIONS: "Analizando emociones", CALCULATING_SATISFACTION: "Calculando satisfacción",
  GENERATING_REPORT: "Generando informe", COMPLETED: "Completada", ERROR: "Error",
};
export const isProcessing = (s: string) => !["COMPLETED", "ERROR"].includes(s);

export const satColor = (v?: number | null) => (v == null ? "text-muted" : v >= 61 ? "text-good" : v >= 41 ? "text-warn" : "text-bad");
export const confWord = (c?: number | null) => (c == null ? "—" : c >= 0.75 ? "Alta" : c >= 0.5 ? "Moderada" : "Baja");
export const TREND_ES: Record<string, string> = { improving: "Positiva", declining: "Negativa", stable: "Estable" };
export const ROLE_ES: Record<string, string> = { client: "Cliente", agent: "Agente", user: "Usuario", advisor: "Asesor", other: "Sin rol" };
