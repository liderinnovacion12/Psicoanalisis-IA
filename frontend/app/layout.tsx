import "./globals.css";
import type { Metadata } from "next";
import { AppProvider } from "@/lib/app-context";

export const metadata: Metadata = { title: "Analizador de Llamadas IA", description: "Análisis de emociones y satisfacción en llamadas telefónicas" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body className="min-h-screen font-sans">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
