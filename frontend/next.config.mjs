/** El navegador siempre habla con el mismo origen (/api/v1/*): Next lo reenvía al backend.
 *  Así la cookie de sesión httpOnly funciona y el <audio> reproduce con autenticación.
 *  En producción con archivos muy grandes, ponga un proxy inverso (nginx/traefik) delante: ver README. */
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";
// Modo demostración sin backend: explícito (NEXT_PUBLIC_DEMO=1|0) o automático en Vercel cuando no hay BACKEND_URL.
const DEMO = process.env.NEXT_PUBLIC_DEMO ?? (process.env.VERCEL && !process.env.BACKEND_URL ? "1" : "0");
export default {
  env: { NEXT_PUBLIC_DEMO: DEMO },
  output: DEMO === "1" ? undefined : "standalone",
  reactStrictMode: true,
  async rewrites() {
    if (DEMO === "1") return [];
    return [{ source: "/api/v1/:path*", destination: `${BACKEND}/api/v1/:path*` }];
  },
  // Sin esto el proxy de Next corta las subidas a 10 MB (las llamadas largas pesan cientos de MB).
  experimental: { proxyTimeout: 30 * 60 * 1000, middlewareClientMaxBodySize: "4gb" },
};
