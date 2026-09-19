// Cliente HTTP. Nunca muestra errores técnicos: solo el `message` seguro que devuelve la API.
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

const BASE = "/api/v1";
const GENERIC = "No fue posible completar la operación. Intente nuevamente.";

async function parse(res: Response) {
  if (res.ok) {
    if (res.status === 204) return null;
    const ct = res.headers.get("content-type") || "";
    return ct.includes("json") ? res.json() : res.blob();
  }
  let msg = GENERIC;
  try {
    const j = await res.json();
    if (j?.message && typeof j.message === "string") msg = j.message;
  } catch {}
  if (res.status === 401 && typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
    window.location.href = "/login";
  }
  throw new ApiError(msg, res.status);
}

export async function api<T = any>(path: string, opts: { method?: string; body?: any; form?: FormData; query?: Record<string, any> } = {}): Promise<T> {
  const q = opts.query
    ? "?" + Object.entries(opts.query).filter(([, v]) => v !== undefined && v !== null && v !== "").flatMap(([k, v]) => Array.isArray(v) ? v.map((x) => `${k}=${encodeURIComponent(x)}`) : [`${k}=${encodeURIComponent(String(v))}`]).join("&")
    : "";
  let res: Response;
  try {
    res = await fetch(BASE + path + q, {
      method: opts.method || (opts.body || opts.form ? "POST" : "GET"),
      credentials: "same-origin",
      headers: opts.body ? { "Content-Type": "application/json" } : undefined,
      body: opts.form ?? (opts.body ? JSON.stringify(opts.body) : undefined),
    });
  } catch {
    throw new ApiError("No hay conexión con el servidor. Verifique su red.", 0);
  }
  return parse(res) as Promise<T>;
}

/** Subida con progreso (XMLHttpRequest); el backend la recibe en streaming. */
export function uploadWithProgress(path: string, form: FormData, onProgress: (pct: number) => void): Promise<any> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", BASE + path);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress((e.loaded / e.total) * 100);
    xhr.onload = () => {
      let j: any = null;
      try { j = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(j);
      else reject(new ApiError(j?.message || GENERIC, xhr.status));
    };
    xhr.onerror = () => reject(new ApiError("No hay conexión con el servidor. Verifique su red.", 0));
    xhr.send(form);
  });
}

export const downloadUrl = (path: string, query: Record<string, string> = {}) =>
  BASE + path + (Object.keys(query).length ? "?" + new URLSearchParams(query).toString() : "");
