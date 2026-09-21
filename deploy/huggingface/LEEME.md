# Desplegar en Hugging Face Spaces (todo en un solo contenedor)

Un único Space corre la interfaz (Next.js) y el backend (FastAPI + Whisper + wav2vec) en el puerto 7860. Plan gratuito:
2 vCPU y 16 GB de RAM. **No verificado con Docker/HF real** (no había Docker en el entorno de desarrollo): el ensamblado
(Next.js *standalone* + backend, login, análisis y subida de 100 MB) se probó localmente, pero el `Dockerfile` no se construyó.

## Pasos

1. En https://huggingface.co/new-space: nombre (p. ej. `analizador-llamadas`), **SDK: Docker → Blank**, hardware **CPU basic (free)**.
2. En el Space → **Settings → Variables and secrets → New secret** (los tres son obligatorios; sin ellos el contenedor se niega a arrancar,
   para que nadie pueda registrarse como administrador al abrir la URL):
   * `SECRET_KEY` = una cadena aleatoria larga (`python -c "import secrets; print(secrets.token_hex(32))"`)
   * `FIRST_ADMIN_EMAIL` = su correo
   * `FIRST_ADMIN_PASSWORD` = una contraseña de al menos 8 caracteres
3. En **Files → Add file → Create a new file**:
   * `Dockerfile` → pegue el contenido de `deploy/huggingface/Dockerfile`
   * `README.md` (reemplace el existente) → pegue `deploy/huggingface/SPACE_README.md`
   (o clone el Space con git y copie esos dos archivos).
4. Espere la construcción (10-20 min la primera vez: instala dependencias y descarga ~2 GB de modelos). Cuando diga *Running*,
   **abra la URL directa** `https://<usuario>-<space>.hf.space` (no la página de huggingface.co, que la embebe en un iframe donde el navegador bloquea la cookie de sesión).
5. Inicie sesión con `FIRST_ADMIN_EMAIL` / `FIRST_ADMIN_PASSWORD`, suba una llamada.

Para actualizar a la última versión de `main` en GitHub: **Settings → Factory rebuild**.

## Qué esperar

* **Es lento en CPU gratuita**: con 2 vCPU calcule ~2-3 minutos de espera por minuto de audio con el modelo Whisper `small`. Para acelerar,
  entre como administrador a *Configuración → Audio → Modelo Whisper* y elija `base` o `tiny` (se descarga al primer uso).
* **Los datos no persisten**: el disco del Space es efímero. Al reiniciar (o al dormirse tras ~48 h sin uso) se pierden llamadas, resultados
  y datasets. Para conservarlos: almacenamiento persistente de pago de HF, o conectar una base de datos externa (`DATABASE_URL`) y S3 (`STORAGE_BACKEND=s3`).
* **Privacidad**: el procesamiento ocurre en los servidores de Hugging Face, no en su equipo. No suba grabaciones con datos sensibles sin autorización.
  El Space público solo se puede usar con el login del administrador (registro deshabilitado), pero considere hacerlo *Private*.
* Solo hay un trabajador de análisis: las llamadas se procesan de a una.
* La diarización en audio **mono** usa el método de respaldo (menos preciso) porque pyannote requiere un token de HF y no se instala aquí; con audio
  **estéreo con un hablante por canal** es fiable.
