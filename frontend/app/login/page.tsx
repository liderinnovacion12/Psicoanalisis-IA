"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { Button, Field, Notice, Toasts } from "@/components/ui";

export default function LoginPage() {
  const router = useRouter();
  const { setUser, user } = useApp();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [setup, setSetup] = useState<{ needs_setup: boolean; signup_enabled: boolean } | null>(null);
  const [f, setF] = useState({ email: "", password: "", organization: "", name: "" });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (user) router.replace("/"); }, [user, router]);
  useEffect(() => {
    api("/auth/setup-status").then((s: any) => { setSetup(s); if (s.needs_setup) setMode("register"); }).catch(() => {});
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setErr("");
    try {
      const r: any = mode === "login"
        ? await api("/auth/login", { body: { email: f.email, password: f.password } })
        : await api("/auth/register", { body: { organization: f.organization, name: f.name, email: f.email, password: f.password } });
      setUser(r.user);
      router.replace("/");
    } catch (x: any) { setErr(x.message); } finally { setBusy(false); }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border border-line bg-surface p-7 shadow-card">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand text-xl font-bold text-white">◉</div>
          <div><h1 className="text-lg font-semibold leading-tight">Analizador de Llamadas IA</h1><p className="text-xs text-muted">{mode === "login" ? "Inicie sesión" : "Configuración inicial: cree su organización"}</p></div>
        </div>
        {mode === "register" && (<>
          <Field label="Organización"><input className="w-full" required minLength={2} value={f.organization} onChange={(e) => setF({ ...f, organization: e.target.value })} /></Field>
          <Field label="Su nombre"><input className="w-full" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></Field>
        </>)}
        <Field label="Correo"><input className="w-full" type="email" required autoComplete="username" value={f.email} onChange={(e) => setF({ ...f, email: e.target.value })} /></Field>
        <Field label="Contraseña" hint={mode === "register" ? "Mínimo 8 caracteres" : undefined}><input className="w-full" type="password" required minLength={mode === "register" ? 8 : 1} autoComplete={mode === "login" ? "current-password" : "new-password"} value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} /></Field>
        {err && <Notice tone="bad">{err}</Notice>}
        <Button variant="primary" className="w-full" disabled={busy}>{busy ? "Procesando…" : mode === "login" ? "Ingresar" : "Crear organización"}</Button>
        {setup?.signup_enabled && !setup.needs_setup && (
          <button type="button" className="w-full text-center text-xs text-muted hover:text-ink" onClick={() => setMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "Crear una organización nueva" : "Ya tengo cuenta"}
          </button>
        )}
      </form>
      <Toasts />
    </div>
  );
}
