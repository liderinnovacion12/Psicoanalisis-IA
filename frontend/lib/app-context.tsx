"use client";
import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from "react";
import { api } from "./api";

export type User = { id: string; org_id: string; email: string; name: string; role: "ADMIN" | "ANALYST" | "VIEWER" };
type Toast = { id: number; kind: "ok" | "err" | "info"; text: string };

type Ctx = {
  user: User | null;
  loading: boolean;
  logout: () => Promise<void>;
  toast: (kind: Toast["kind"], text: string) => void;
  toasts: Toast[];
  dismiss: (id: number) => void;
  theme: "light" | "dark";
  toggleTheme: () => void;
  can: (need: "ANALYST" | "ADMIN") => boolean;
  setUser: (u: User | null) => void;
};
const C = createContext<Ctx>(null as any);
export const useApp = () => useContext(C);

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    let t: "light" | "dark" = "light";
    try {
      const s = localStorage.getItem("theme");
      t = s === "dark" || s === "light" ? s : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    } catch {}
    setTheme(t);
    document.documentElement.classList.toggle("dark", t === "dark");
    api<User>("/auth/me").then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const n = prev === "dark" ? "light" : "dark";
      document.documentElement.classList.toggle("dark", n === "dark");
      try { localStorage.setItem("theme", n); } catch {}
      return n;
    });
  }, []);

  const toast = useCallback((kind: Toast["kind"], text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5500);
  }, []);
  const dismiss = (id: number) => setToasts((t) => t.filter((x) => x.id !== id));
  const logout = async () => { try { await api("/auth/logout", { method: "POST" }); } catch {} setUser(null); window.location.href = "/login"; };
  const rank = { VIEWER: 0, ANALYST: 1, ADMIN: 2 } as const;
  const can = (need: "ANALYST" | "ADMIN") => !!user && rank[user.role] >= rank[need];

  return <C.Provider value={{ user, loading, logout, toast, toasts, dismiss, theme, toggleTheme, can, setUser }}>{children}</C.Provider>;
}
