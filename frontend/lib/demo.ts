// MODO DEMOSTRACIÓN (sin backend). Sirve respuestas de la API que se guardaron al ejecutar la aplicación REAL sobre la
// llamada de ejemplo (scripts/export_demo_snapshot.py). No procesa audio ni escribe datos. Se activa con NEXT_PUBLIC_DEMO=1
// (en Vercel se activa solo si no hay BACKEND_URL; ver next.config.mjs).
export const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";
export const DEMO_MSG = "Modo demostración: esta acción está deshabilitada. Los datos son de un análisis real de la llamada de ejemplo; para procesar sus propias llamadas despliegue el backend.";

type Snap = { routes: Record<string, any>; models: Record<string, any>; ids: Record<string, string> };
let cache: Promise<Snap> | null = null;
const load = () => (cache ??= import("./demo-data.json").then((m) => (m as any).default as Snap));

export class DemoBlocked extends Error {}

export async function demoRequest(method: string, path: string, query: Record<string, any> = {}): Promise<any> {
  const S = await load();
  const R = S.routes;
  if (method !== "GET") {
    if (path === "/auth/login" || path === "/auth/register") return { access_token: "demo", token_type: "bearer", user: R["/auth/me"] };
    if (path === "/auth/logout") return { message: "ok" };
    throw new DemoBlocked(DEMO_MSG);
  }
  let m: RegExpMatchArray | null;
  if ((m = path.match(/^\/calls\/([^/]+)(\/[a-z]+)?$/))) {
    const [, id, sub = ""] = m;
    if (id === "upload" || (id !== S.ids.call && id !== "new")) throw new DemoBlocked("La llamada solicitada no existe en la demostración.");
    if (sub === "/transcription") return R[query.redact === true || query.redact === "true" ? "/calls/:id/transcription?redact" : "/calls/:id/transcription"];
    if (sub === "/audio") return null;
    return R[`/calls/:id${sub}`];
  }
  if ((m = path.match(/^\/models\/([^/]+)$/)) && m[1] !== "compare") {
    const d = S.models[m[1]];
    if (!d) throw new DemoBlocked("El modelo no existe.");
    return d;
  }
  if ((m = path.match(/^\/datasets\/([^/]+)(\/samples)?$/))) return R[m[2] ? "/datasets/:id/samples" : "/datasets/:id"];
  if ((m = path.match(/^\/training\/([^/]+)$/))) return (R["/training"] as any[]).find((r) => r.id === m![1]) ?? R["/training"][0];
  if (path === "/calls") {
    const q = String(query.q || "").toLowerCase();
    const items = (R["/calls"].items as any[]).filter((c) => (!query.status || c.status === query.status) && (!q || (c.filename + (c.display_name || "") + c.id).toLowerCase().includes(q)));
    return { ...R["/calls"], items, total: items.length };
  }
  if (path in R) return R[path];
  throw new DemoBlocked(DEMO_MSG);
}

export const demoAudioUrl = () => "/demo/audio.mp3";
export function demoDownload(path: string, query: Record<string, string>): string {
  if (path.startsWith("/reports/")) return "/demo/reporte.pdf";
  if (/^\/calls\/[^/]+\/export$/.test(path)) return query.format === "pdf" ? "/demo/reporte.pdf" : `/demo/exportacion.${query.format || "json"}`;
  return "#";
}
