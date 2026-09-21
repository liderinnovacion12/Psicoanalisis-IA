#!/bin/bash
# Arranque del Space: backend (127.0.0.1:8000) + interfaz Next.js (0.0.0.0:7860).
if [ -z "$FIRST_ADMIN_EMAIL" ] || [ -z "$FIRST_ADMIN_PASSWORD" ] || [ -z "$SECRET_KEY" ]; then
  echo "ERROR: defina los secretos SECRET_KEY, FIRST_ADMIN_EMAIL y FIRST_ADMIN_PASSWORD en Settings > Variables and secrets del Space." >&2
  echo "Sin ellos cualquiera podria registrarse como administrador al abrir la URL." >&2
  sleep 5; exit 1
fi
mkdir -p "$DATA_DIR"
(cd /app/backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1) &
cd /app/frontend && PORT=7860 HOSTNAME=0.0.0.0 exec node server.js
