"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
import { useApp } from "@/lib/app-context";
import { api } from "@/lib/api";
import { Toasts, Skeleton, Badge } from "./ui";
import { DEMO } from "@/lib/demo";

const NAV = [
  { href: "/", label: "Analizar llamada", icon: "◉" },
  { href: "/calls", label: "Historial", icon: "☎" },
  { href: "/dashboard", label: "Dashboard", icon: "▦" },
  { href: "/dataset", label: "Dataset", icon: "▤", adv: true },
  { href: "/labeling", label: "Etiquetado", icon: "✎", adv: true },
  { href: "/training", label: "Entrenamiento", icon: "⚙", adv: true },
  { href: "/models", label: "Modelos", icon: "◈", adv: true },
  { href: "/reports", label: "Reportes", icon: "⇩", adv: true },
  { href: "/settings", label: "Configuración", icon: "☰", adv: true },
];
const CRUMBS: Record<string, string> = { dashboard: "Dashboard", calls: "Llamadas", new: "Nueva llamada", results: "Resultados", dataset: "Dataset", labeling: "Etiquetado", training: "Entrenamiento", models: "Modelos", reports: "Reportes", settings: "Configuración" };

export default function Shell({ children }: { children: ReactNode }) {
  const { user, loading, logout, theme, toggleTheme } = useApp();
  const path = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [dev, setDev] = useState<any>(null);

  useEffect(() => { if (!loading && !user) router.replace("/login"); }, [loading, user, router]);
  useEffect(() => { if (user) api("/system/info").then(setDev).catch(() => {}); }, [user]);
  useEffect(() => setOpen(false), [path]);

  if (loading || !user) {
    return <div className="flex h-screen items-center justify-center"><Skeleton className="h-10 w-48" /></div>;
  }
  const parts = path.split("/").filter(Boolean);
  const crumbs = [{ href: "/", label: "Inicio" }, ...parts.map((p, i) => ({ href: "/" + parts.slice(0, i + 1).join("/"), label: CRUMBS[p] || (p.length > 12 ? `Llamada ${p.slice(0, 8)}` : p) }))];
  const active = (h: string) => (h === "/" ? path === "/" : h === "/dashboard" ? path === "/dashboard" : h === "/calls" ? path === "/calls" || (path.startsWith("/calls/") && !path.startsWith("/calls/new")) : path.startsWith(h));

  const sidebar = (
    <nav className="flex h-full flex-col gap-1 p-3">
      <div className="mb-4 flex items-center gap-2.5 px-2 pt-1">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand text-lg font-bold text-white">◉</div>
        <div><p className="text-sm font-semibold leading-tight">Analizador de<br />Llamadas IA</p></div>
      </div>
      {NAV.map((n, i) => (
        <div key={n.href} className="contents">
        {n.adv && !NAV[i - 1]?.adv && <p className="mb-1 mt-3 px-3 text-[11px] font-semibold uppercase tracking-wide text-muted/70">Avanzado</p>}
        <Link href={n.href} className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${active(n.href) ? "bg-brand/12 text-brand" : "text-muted hover:bg-surface2 hover:text-ink"}`}>
          <span className="w-4 text-center opacity-80">{n.icon}</span>{n.label}
        </Link>
        </div>
      ))}
      <div className="mt-auto rounded-lg border border-line bg-surface2 p-3 text-xs">
        <p className="mb-1 font-medium text-muted">DEVICE</p>
        <p className="font-semibold">{dev ? dev.device.label : "…"}</p>
        {dev && <p className="mt-1 text-muted">Modo de tareas: {dev.task_mode}</p>}
      </div>
    </nav>
  );

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[236px_1fr]">
      <aside className="sticky top-0 hidden h-screen border-r border-line bg-surface lg:block">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-40 bg-black/40 lg:hidden" onClick={() => setOpen(false)}>
          <aside className="h-full w-64 bg-surface" onClick={(e) => e.stopPropagation()}>{sidebar}</aside>
        </div>
      )}
      <div className="min-w-0">
        {DEMO && (
          <div className="border-b border-warn/40 bg-warn/15 px-4 py-1.5 text-center text-xs lg:px-8">
            <b>MODO DEMOSTRACIÓN</b> — sin servidor: se muestran respuestas reales de la aplicación sobre una llamada de ejemplo <b>sintética (voz TTS)</b>;
            sus emociones no son representativas. Subir, etiquetar y entrenar están deshabilitados.
          </div>)}
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-line bg-surface/90 px-4 py-2.5 backdrop-blur lg:px-8">
          <div className="flex items-center gap-3">
            <button className="rounded-lg border border-line px-2.5 py-1.5 text-sm lg:hidden" onClick={() => setOpen(true)} aria-label="Menú">☰</button>
            <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-sm text-muted">
              {crumbs.map((c, i) => (
                <span key={c.href} className="flex items-center gap-1.5">
                  {i > 0 && <span>/</span>}
                  {i === crumbs.length - 1 ? <span className="font-medium text-ink">{c.label}</span> : <Link href={c.href} className="hover:text-ink">{c.label}</Link>}
                </span>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2">
            {!DEMO && dev && !dev.pyannote_available && <Badge tone="warn" className="hidden md:inline-flex">Diarización: respaldo (sin pyannote)</Badge>}
            <button onClick={toggleTheme} className="rounded-lg border border-line px-2.5 py-1.5 text-sm hover:bg-surface2" aria-label="Cambiar tema" title="Modo claro/oscuro">{theme === "dark" ? "☀" : "☾"}</button>
            <div className="hidden text-right text-xs leading-tight sm:block"><p className="font-medium">{user.name || user.email}</p><p className="text-muted">{user.role}</p></div>
            <button onClick={logout} className="rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface2">Salir</button>
          </div>
        </header>
        <main className="mx-auto max-w-[1500px] px-4 py-6 lg:px-8">{children}</main>
      </div>
      <Toasts />
    </div>
  );
}
