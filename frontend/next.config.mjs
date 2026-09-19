/** El navegador siempre habla con el mismo origen (/api/v1/*): Next lo reenvía al backend.
 *  Así la cookie de sesión httpOnly funciona y el <audio> reproduce con autenticación.
 *  En producción con archivos muy grandes, ponga un proxy inverso (nginx/traefik) delante: ver README. */
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";
export default {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/v1/:path*", destination: `${BACKEND}/api/v1/:path*` }];
  },
  experimental: { proxyTimeout: 30 * 60 * 1000 },
};
