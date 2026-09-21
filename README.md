# Analizador de Llamadas IA

Aplicación web de extremo a extremo para **estimar emociones y satisfacción en llamadas telefónicas** (dos participantes):
subida de la grabación → audio (FFmpeg) → VAD → diarización → transcripción → emociones por ventanas → Satisfaction Engine →
eventos críticos → dashboard, reproductor sincronizado, PDF/CSV/Excel/JSON, datasets, etiquetado, fine-tuning, registro y
comparación de modelos. Procesamiento **100 % local**.

> **Regla de lectura de los resultados.** Todo lo que muestra la aplicación son *estimaciones de modelos* («el modelo
> estima…»), no afirmaciones. La probabilidad de una emoción, la **confianza** de esa predicción y la **satisfacción** son tres
> cosas distintas. La exactitud publicada del modelo base **no** es la precisión real en llamadas ni en español.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Estado real de la entrega (qué está verificado y qué no)

Verificado en este entorno (Windows 11, CPU, sin GPU, sin Docker):

| Área | Evidencia |
|---|---|
| Modelo base `r-f/wav2vec-english-speech-emotion-recognition` | Descargado y probado (`backend/scripts/test_emotion_model.py`, salida en `docs/test_emotion_model_output.txt`). |
| Pipeline completo con modelos reales (Whisper `small` + wav2vec baseline + Silero VAD) | `tests/test_pipeline_ml.py` (5 pruebas) y `scripts/benchmark.py`: llamada de 84 s → RTF ≈ 1.0 en CPU de desarrollo. |
| Reanudación por checkpoints | Test con fallo forzado a mitad de la etapa de emociones: no se repite audio/transcripción y no hay predicciones duplicadas. |
| Fine-tuning real, splits sin fuga, métricas y matriz de confusión, registro versionado, activación, stop/continuar | `tests/test_training.py` (6 pruebas, wav2vec2 minúsculo) y `scripts/train.py`. |
| API (59 endpoints), roles, aislamiento multi-tenant, errores sin detalle técnico, Range de audio | `tests/test_api.py` (12 pruebas). |
| UI en navegador real (Edge): login, subida, progreso, resultados, clic en transcripción → salto de audio, modo oscuro, tablet | Prueba de humo con capturas. |
| Llamada de 60 min sin cargar el audio en memoria | `tests/test_audio.py::test_memory_does_not_grow_with_call_duration` |

**Total de pruebas:** 23 unitarias + 12 API + 10 audio/formatos + 5 pipeline ML + 6 entrenamiento.

**NO verificado / limitaciones (dicho claramente):**

* **pyannote.audio no se pudo ejecutar** (exige token de Hugging Face y aceptar condiciones del modelo). El código está escrito
  (chunks de 30 min, reconciliación de hablantes por embeddings, reducción a 2 voces, aviso de >2) pero **no probado con el modelo real**.
  Sin él se usa: (a) **audio estéreo con un hablante por canal** (fiable, probado), o (b) un **diarizador de respaldo con embeddings de voz**
  (`microsoft/wavlm-base-plus-sv`, sin token) que además decide si hay **1 o 2 personas**; su «calidad de diarización» se limita a ≤ 70 y la UI lo advierte.
  Calibración limitada: solo hay 1 ejemplo de dos voces (sintético) y 3 de una voz (2 reales). Si el automático se equivoca, el selector
  «Personas en la llamada» (auto / 1 / 2) lo fuerza. Un método anterior por estadísticas MFCC se descartó como principal: con datos reales dio
  los mismos números para una persona con entonación variable y para dos voces sintéticas (no las distingue); queda como último recurso.
* **WhisperX**: la integración está escrita, pero por defecto se usa `faster-whisper` (su mismo motor, con timestamps por palabra). No probada.
* **Docker/Compose, S3/MinIO, Celery+Redis, PostgreSQL, GPU/CUDA**: escritos y revisados, **no ejecutados aquí** (no hay Docker/GPU). En
  desarrollo se usa SQLite y `TASK_MODE=inline` con exactamente el mismo código de etapas. Pruebe `docker compose up` en su entorno.
* **No hay resultados emocionales «reales» que mostrar**: el audio de `examples/call/` es **voz sintética (TTS) de emoción plana**. Sirve para
  validar el flujo; sus emociones/satisfacción no significan nada. El modelo base es inglés y sobreconfiado (probabilidades ≈ 1.0).
  Evalúelo con **sus** grabaciones etiquetadas antes de confiar en él.
* El modelo de satisfacción entrenable y la calibración funcionan (probados con datos de prueba), pero sus pesos por defecto son
  **heurísticos configurables**, no aprendidos. Calíbrelos con satisfacción real (CSAT/encuestas).
* Anonimización: solo **texto** (PII por reglas ES/EN). No se anonimiza el audio. Cobertura de nombres heurística.
* Análisis multimodal: audio + prosodia + texto (léxico ES/EN) implementados con pesos configurables. Un modelo de texto Hugging Face o un LLM
  se enchufan implementando `TextAnalyzer` (hay un `HFTextAnalyzer` opcional sin probar).
* Subida: streaming a disco, sin memoria; **no** hay subida reanudable por fragmentos.
* Alembic no incluido: las tablas se crean con `create_all` al arrancar (añada migraciones antes de evolucionar el esquema en producción).

---------------------------------------------------------------------------------------------------------------------------------

## 2. Hallazgo importante sobre el modelo base

Al descargar el modelo se comprobó que **su checkpoint no es compatible con `AutoModelForAudioClassification`**: contiene la cabeza
personalizada `classifier.dense (1024×1024)` + `classifier.out_proj (7×1024)` con pooling medio (`Wav2Vec2ForSpeechClassification`,
`finetuning_task: wav2vec2_clf`). Cargarlo con la clase estándar **no falla**: inicializa la cabeza al azar y devuelve probabilidades de
apariencia válida pero sin significado. La aplicación implementa la arquitectura correcta
([architectures.py](backend/app/ml/emotion/architectures.py)) y **falla si falta alguna clave de la cabeza** al cargar cualquier modelo.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Arquitectura (resumen; detalle en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

```
Navegador ─► Next.js ──/api/v1/*──► FastAPI ─► PostgreSQL          Storage: local (cifrado opcional) | S3/MinIO
                                        │
                                      Redis ─► Celery: audio → diarization → transcription → emotion → satisfaction(+report)
                                                       (una cola/worker por etapa)                     training (GPU)
```

* **Modelo desacoplado**: `EmotionModel` (ABC) → `Wav2VecEmotionModel`, `FineTunedEmotionModel`; `register_loader()` para añadir otros.
  Cambiar de modelo = **activar otro registro** en *Modelos*; cada análisis guarda `model_id` y `model_version`.
* **Llamadas largas**: el audio normalizado es WAV PCM16 en disco; todo se lee por bloques/ventanas con `seek` (VAD 5 min, Whisper ~10 min
  cortando en silencios, emociones en lotes de ventanas de 5 s con hop 2.5 s, pyannote en chunks de 30 min). Nada carga la llamada entera.
* **Checkpoints**: `calls.checkpoints` por etapa; la de emociones persiste predicciones y avance en la misma transacción, y **reanuda por lote**.
* **Cache** por `(hash del audio, etapa, hash de configuración/modelo)` y por organización.
* **Multi-tenant**: `org_id` en todas las tablas y filtro en cada consulta; roles ADMIN / ANALYST / VIEWER; JWT en cookie httpOnly + Bearer.
* **Configuración centralizada**: `backend/config/{audio,emotion,satisfaction,app}_config.yaml` + overrides por organización (pantalla *Configuración*).

### Satisfaction Engine
No equipara emoción con satisfacción. Serie de valencia `v_t = Σ w_e·p_t(e)` y componentes en [-1,1]: señal emocional (duración·intensidad·confianza),
**estado final** (peso `final_state_weight`), **tendencia**, **persistencia** de estados negativos, **estabilidad**, texto y prosodia.
`score = 50 + 50·clip(sensibilidad · Σwᵢcᵢ/Σwᵢ)`. Devuelve factores con puntos aportados, una **confianza propia** (cobertura + confianza del modelo +
calidad de audio + texto; penaliza desajuste de idioma) y la evolución temporal. Modos `rules`, `model` (regresor sklearn entrenable), `hybrid`, y
calibración lineal con satisfacción real.

---------------------------------------------------------------------------------------------------------------------------------

## 3b. Modo demostración (ver la interfaz sin servidor, p. ej. en Vercel)

El frontend puede ejecutarse **sin backend**: con `NEXT_PUBLIC_DEMO=1` (o automáticamente en Vercel cuando no hay `BACKEND_URL`) sirve
respuestas de la API guardadas de una **ejecución real** de la aplicación (`frontend/lib/demo-data.json` + `frontend/public/demo/`):
llamada de ejemplo analizada con Whisper + wav2vec, su audio, PDF/CSV/XLSX/JSON reales y un dataset/entrenamiento de juguete.
* **Es una demostración, no producción**: la voz de la llamada es sintética (TTS) y el dataset son tonos, así que las emociones **no son representativas**.
  Un aviso lo indica en pantalla. Subir, etiquetar, entrenar y activar modelos están deshabilitados (no hay servidor que los ejecute).
* Regenerar los datos: `cd backend && python scripts/export_demo_snapshot.py` (ejecuta el pipeline real; ~3 min con los modelos descargados).
* Vercel: Root Directory `frontend`, preset Next.js, **sin** variables de entorno → modo demo. Para la app real, defina `BACKEND_URL` y despliegue el backend aparte.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Requisitos

Python 3.11/3.12 · Node 20+ (probado 22) · FFmpeg (o `imageio-ffmpeg`, ya incluido como respaldo) · opcional: Docker, GPU NVIDIA + drivers CUDA.
La primera ejecución descarga modelos (~1.3 GB wav2vec, ~0.5 GB Whisper `small`) a `MODEL_CACHE_DIR`; después no vuelve a descargar.

## 5. Instalación y ejecución (desarrollo, sin Docker)

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\install.ps1          # añada -Gpu para CUDA, -Pyannote para pyannote
cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000   # API  (TASK_MODE=inline)
cd frontend; npm run dev                                                      # UI en http://localhost:3000
```
```bash
# Linux/macOS
bash scripts/install.sh   # --gpu / --pyannote
cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000
cd frontend && npm run dev
```
La primera vez, la pantalla de login ofrece crear la organización y el administrador (o defina `FIRST_ADMIN_EMAIL/PASSWORD`,
o use `python scripts/create_admin.py --org "Mi Empresa" --email admin@empresa.com`). Documentación interactiva de la API: `http://localhost:8000/docs`.

### Docker (producción)
```bash
cp .env.example .env            # cambie SECRET_KEY, contraseñas, HF_TOKEN
docker compose up -d --build    # postgres, redis, backend, 6 workers, beat, frontend  → http://localhost:3000
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build    # GPU NVIDIA
docker compose up -d --scale worker-emotion=3                                    # escalar una etapa
docker compose --profile minio up -d        # MinIO (STORAGE_BACKEND=s3)
```
Servicios: `frontend, backend, postgres, redis, worker-*` (+ `minio` opcional). En varios servidores, los workers deben compartir `/data/work`
(NFS/EFS) y usar S3 para los originales. Con archivos muy grandes ponga un proxy inverso (nginx/traefik) delante del frontend con
`client_max_body_size` alto y `proxy_request_buffering off`.

## 6. Variables de entorno
Ver [.env.example](.env.example): `DATABASE_URL, REDIS_URL, TASK_MODE, MODEL_NAME, MODEL_CACHE_DIR, UPLOAD_DIR, REPORT_DIR, MODELS_DIR, MAX_FILE_SIZE,
DEVICE, HF_TOKEN, SECRET_KEY, ENCRYPTION_KEY, STORAGE_BACKEND, S3_*, CORS_ORIGINS, COOKIE_SECURE, ALLOW_SIGNUP, LOG_*`.

## 7. GPU
`DEVICE=auto` detecta CUDA. El encabezado de la app muestra **DEVICE: GPU: NVIDIA …** o **CPU MODE**. Con GPU se usa fp16. Benchmark:
`python scripts/benchmark.py --audio llamada.mp3 [--repeat 40]` → tiempo por etapa, **Real Time Factor**, RAM, CPU, VRAM/GPU.
Referencia medida aquí (CPU, 84 s de audio, modelos ya descargados): transcripción 29 s + emociones 51 s → **RTF ≈ 1.0**; con hop 2.5 s cada segundo se analiza
dos veces. Espere mucho mejor en GPU; en CPU una llamada de 1 h tardaría ≈ 1 h.

## 8. Diarización, audio y transcripción
* **Estéreo con un hablante por canal** (detección automática por correlación y actividad exclusiva) → cada canal es un hablante; se obtienen solapamientos reales.
* **Mono**: pyannote (defina `HF_TOKEN`, acepte las condiciones de `pyannote/speaker-diarization-3.1` y `pip install "pyannote.audio>=3.3,<4"`), o el respaldo local
  por embeddings de voz. **Una o dos personas**: se detecta automáticamente y se puede forzar en la subida o al re-analizar; con una sola persona la app muestra
  una única tarjeta «Persona» (sin resultado de interacción). Si hay más de dos voces relevantes: *«Se detectaron más de dos posibles hablantes. Revise la diarización.»*
* Formatos: MP3, WAV, M4A, AAC, FLAC, OGG (probados). Se normaliza a 16 kHz mono (loudnorm + filtro paso alto). Calidad de audio 0-100 (SNR estimada por percentiles,
  clipping, nivel, voz, ancho de banda); si es baja: *«El resultado puede presentar menor precisión debido a la calidad del audio.»*
* Transcripción: faster-whisper (`small` por defecto; cambie en *Configuración → Audio*), idioma detectado por votación y fijado para toda la llamada (prioridad español).
  Palabras asignadas al hablante por su punto medio → una frase que cruza un cambio de turno se parte correctamente.

## 9. Datasets, etiquetado y entrenamiento
1. **Etiquetado** (UI): elija una llamada analizada, seleccione un segmento (clic en la transcripción), hablante, emoción (+ categorías personalizadas) y satisfacción 0-100.
   Solo se admiten llamadas con **«Permitir utilizar esta llamada para mejorar el modelo»** (desactivado por defecto; cambios auditados).
2. **Dataset**: subida de audios sueltos, importación CSV+ZIP (`examples/dataset/`), versiones (`dataset_v1 → v2`), congelado, estadísticas (por emoción/speaker/satisfacción,
   duración total/promedio), **alerta de desbalance**, validación (corrupto, sin etiqueta, duplicado, muy corto, sin voz, inconsistencias).
3. **Splits** 70/15/15 configurables, agrupados por persona/llamada (**sin fuga de datos**) y estratificados; avisa si hay pocos grupos para evitarla.
4. **Entrenamiento**: AdamW + warmup, gradient accumulation, weight decay, early stopping por Macro-F1, class weights u oversampling, augmentation segura (pitch/velocidad
   desactivados por defecto y acotados: alteran la percepción emocional), checkpoint por época, **detener/continuar**, progreso en vivo. Solo un entrenamiento a la vez por organización.
   Si cambia el conjunto de etiquetas solo se reinicializa la capa de salida. En CPU solo es práctico con datasets pequeños; use GPU.
   CLI equivalente: `python scripts/train.py --manifest … --audio-dir … --output …`.
5. **Métricas**: accuracy, precision, recall, F1, **Macro-F1**, Weighted-F1, por clase, **matriz de confusión** interactiva.
6. **Registro**: `emotion_model_vN`, estados TRAINING → VALIDATION → PRODUCTION → ARCHIVED, **Comparación** de modelos y evaluación del baseline y del ajustado
   sobre el **mismo** TEST. Activar (solo ADMIN) hace que las nuevas llamadas usen ese modelo.
7. **Satisfacción**: entrenar regresor con etiquetas de satisfacción o con CSAT/NPS/encuesta reales (`POST /calls/{id}/feedback`, *Dataset → importar satisfacción real*);
   *Configuración → Satisfacción* muestra predicha vs real y aplica una calibración lineal.

### Modelo propio en español
Entrene con sus datos (paso 4) o copie un checkpoint Hugging Face (p. ej. `models/custom_emotion_model`) y regístrelo:
`python scripts/register_model.py --org "Mi Empresa" --path models/custom_emotion_model --name emotion_es_v1 --language es` → *Modelos → Activar*. Sin tocar backend ni frontend.
La app avisa cuando el idioma del modelo ≠ idioma de la llamada.

## 10. Seguridad y privacidad
JWT (cookie httpOnly SameSite=Lax + Bearer), bcrypt, roles, aislamiento por organización, auditoría (`audit_logs`), logs estructurados JSON separados en INFO/WARNING/ERROR
(`data/logs/`). Cifrado en reposo opcional (AES-256-GCM por bloques con lectura aleatoria, sirve `Range` sin descifrar todo). Retención configurable (Celery beat borra vencidas).
Eliminación granular (llamada completa, solo transcripción, solo resultados, solo audio; borra también muestras de dataset derivadas). PII en transcripción con versión redactada y
exportaciones con `redact=true`. Ninguna grabación sale a APIs externas; solo se descargan modelos al inicializar la caché. Los errores técnicos nunca se muestran al usuario.

## 11. API (prefijo `/api/v1`; OpenAPI en `/docs`)
`POST /calls/upload · GET /calls · GET|DELETE /calls/{id} · GET /calls/{id}/status|speakers|transcription|emotions|satisfaction|events|segments|audio · POST /calls/{id}/resume|reanalyze|feedback · GET /calls/{id}/export?format=pdf|csv|xlsx|json · GET /reports/{call_id} ·
POST|GET /datasets · POST /datasets/{id}/samples|samples/upload|split|validate|version|freeze|import · POST /training/start · GET /training[/{id}] · POST /training/{id}/stop|resume ·
GET /models[/{id}] · GET /models/compare · POST /models/{id}/activate|evaluate|archive · GET|PUT /settings[/{section}] · GET /dashboard/summary · GET /system/info · /auth/* · /users`

## 12. Base de datos
`organizations, users, calls, speakers, audio_segments, transcriptions, emotion_predictions, satisfaction_scores, critical_events, models, model_metrics, training_datasets,
training_samples, training_runs, settings` + `satisfaction_feedback` y `audit_logs`. Jerarquía: CALL → SPEAKERS → SEGMENTS → TRANSCRIPTIONS/EMOTIONS → SATISFACTION → EVENTS.
Las probabilidades completas se guardan en JSON por ventana; los archivos grandes van a storage, nunca a PostgreSQL.

## 13. Pruebas
```bash
cd backend
pytest -m "not slow"                      # unitarias + API + audio (rápidas, sin modelos)
pytest tests/test_training.py             # entrenamiento real con un modelo minúsculo (~1 min)
RUN_ML_TESTS=1 pytest tests/test_pipeline_ml.py     # pipeline con Whisper + wav2vec reales (~3 min; requiere modelos descargados)
pytest tests/test_audio.py -k memory_does_not_grow    # llamada de 60 min en streaming
```

## 14. Ejemplo de uso con la llamada incluida
`examples/call/` contiene una llamada **sintética** (cliente/agente): `llamada_ejemplo_estereo.wav` (un hablante por canal), `llamada_ejemplo_mono.wav`, `llamada_ejemplo.mp3`
y `llamada_ejemplo_verdad.json` (turnos reales). Se regenera con `python backend/scripts/make_sample_call.py` (Windows/SAPI).
1. *Nueva llamada* → arrastre `llamada_ejemplo_estereo.wav` → **Analizar**.  2. Vea el progreso por etapas (audio ✓, diarización ✓, transcripción, emociones %…).
3. Se abre el resultado: **diarización** (canal 0 = cliente, canal 1 = agente), **transcripción** clicable que salta el audio, **emociones**, **satisfacción** por persona y de la interacción,
   **eventos** y calidad del análisis.  4. *Reportes* o botones **PDF/CSV/XLSX/JSON** de la cabecera.  Recuerde: sus emociones son ruido (voz sintética).

## 15. Solución de problemas
* *«Diarización: respaldo»*: configure `HF_TOKEN` + pyannote, o use estéreo por canal.  * Lento en CPU: use `transcription.model_size: base/tiny`, hop mayor, o GPU.
* *Descarga de modelos bloqueada*: `HF_OFFLINE=true` con la caché ya poblada.  * FFmpeg no encontrado: `pip install imageio-ffmpeg` o instálelo en el PATH.
