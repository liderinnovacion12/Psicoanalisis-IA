"use client";
import { ReactNode, ButtonHTMLAttributes, useEffect } from "react";
import { useApp } from "@/lib/app-context";

export const cx = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(" ");

export function Card({ title, subtitle, actions, children, className, pad = true }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children?: ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={cx("rounded-xl border border-line bg-surface shadow-card", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div>
            <h3 className="text-sm font-semibold">{title}</h3>
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={pad ? "p-5" : ""}>{children}</div>
    </section>
  );
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; size?: "sm" | "md" };
export function Button({ variant = "secondary", size = "md", className, ...p }: BtnProps) {
  const v = {
    primary: "bg-brand text-white hover:brightness-110 border-transparent",
    secondary: "bg-surface hover:bg-surface2 border-line text-ink",
    ghost: "bg-transparent hover:bg-surface2 border-transparent text-ink",
    danger: "bg-bad/10 text-bad hover:bg-bad/20 border-bad/30",
  }[variant];
  return <button {...p} className={cx("inline-flex items-center justify-center gap-2 rounded-lg border font-medium transition disabled:cursor-not-allowed disabled:opacity-50", size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm", v, className)} />;
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" | "brand"; className?: string }) {
  const t = { neutral: "bg-surface2 text-muted", good: "bg-good/15 text-good", warn: "bg-warn/15 text-warn", bad: "bg-bad/15 text-bad", brand: "bg-brand/15 text-brand" }[tone];
  return <span className={cx("inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium", t, className)}>{children}</span>;
}

export const Skeleton = ({ className }: { className?: string }) => <div className={cx("skeleton", className)} />;

export function Stat({ label, value, hint, tone, loading }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string; loading?: boolean }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4 shadow-card">
      <p className="text-xs font-medium text-muted">{label}</p>
      {loading ? <Skeleton className="mt-2 h-7 w-24" /> : <p className={cx("mt-1 text-2xl font-semibold tabular-nums", tone)}>{value}</p>}
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function Progress({ value, tone = "brand", className }: { value: number; tone?: "brand" | "good" | "bad"; className?: string }) {
  return (
    <div className={cx("h-2 w-full overflow-hidden rounded-full bg-surface2", className)}>
      <div className={cx("h-full rounded-full transition-all duration-500", tone === "good" ? "bg-good" : tone === "bad" ? "bg-bad" : "bg-brand")} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function Empty({ title, text, action }: { title: string; text?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <div className="h-10 w-10 rounded-full bg-surface2" />
      <p className="text-sm font-medium">{title}</p>
      {text && <p className="max-w-sm text-xs text-muted">{text}</p>}
      {action}
    </div>
  );
}

export function Notice({ tone = "warn", children }: { tone?: "warn" | "info" | "bad"; children: ReactNode }) {
  const t = { warn: "border-warn/40 bg-warn/10", info: "border-brand/30 bg-brand/10", bad: "border-bad/40 bg-bad/10" }[tone];
  return <div className={cx("rounded-lg border px-3.5 py-2.5 text-sm", t)}>{children}</div>;
}

export function Modal({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onMouseDown={onClose}>
      <div className={cx("max-h-[90vh] w-full overflow-auto rounded-xl border border-line bg-surface p-5 shadow-xl", wide ? "max-w-3xl" : "max-w-md")} onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label={title}>
        <div className="mb-3 flex items-center justify-between"><h3 className="font-semibold">{title}</h3><button onClick={onClose} className="text-muted hover:text-ink" aria-label="Cerrar">✕</button></div>
        {children}
      </div>
    </div>
  );
}

export function Toasts() {
  const { toasts, dismiss } = useApp();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div key={t.id} onClick={() => dismiss(t.id)} className={cx("pointer-events-auto cursor-pointer rounded-lg border px-3.5 py-2.5 text-sm shadow-lg bg-surface", t.kind === "ok" ? "border-good/50" : t.kind === "err" ? "border-bad/50" : "border-brand/40")}>
          {t.text}
        </div>
      ))}
    </div>
  );
}

export function Pagination({ page, pageSize, total, onPage }: { page: number; pageSize: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex items-center justify-between px-1 pt-3 text-xs text-muted">
      <span>{total === 0 ? "Sin resultados" : `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, total)} de ${total}`}</span>
      <div className="flex items-center gap-1">
        <Button size="sm" disabled={page <= 1} onClick={() => onPage(page - 1)}>Anterior</Button>
        <span className="px-2">Página {page} / {pages}</span>
        <Button size="sm" disabled={page >= pages} onClick={() => onPage(page + 1)}>Siguiente</Button>
      </div>
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-muted">{hint}</span>}
    </label>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { key: T; label: string }[]; value: T; onChange: (k: T) => void }) {
  return (
    <div className="flex gap-1 border-b border-line">
      {tabs.map((t) => (
        <button key={t.key} onClick={() => onChange(t.key)} className={cx("-mb-px border-b-2 px-3.5 py-2 text-sm font-medium transition", value === t.key ? "border-brand text-brand" : "border-transparent text-muted hover:text-ink")}>{t.label}</button>
      ))}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const tone = status === "COMPLETED" ? "good" : status === "ERROR" ? "bad" : status === "QUEUED" || status === "UPLOADED" ? "neutral" : "brand";
  const labels: Record<string, string> = { UPLOADED: "Subida", QUEUED: "En cola", PROCESSING_AUDIO: "Procesando audio", DIARIZING: "Diarizando", TRANSCRIBING: "Transcribiendo", ANALYZING_EMOTIONS: "Emociones", CALCULATING_SATISFACTION: "Satisfacción", GENERATING_REPORT: "Informe", COMPLETED: "Completada", ERROR: "Error" };
  return <Badge tone={tone as any}>{labels[status] || status}</Badge>;
}
