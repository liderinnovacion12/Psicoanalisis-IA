# Arquitectura — Analizador de Llamadas IA

Documento de diseño previo a la implementación (sección 115 de la especificación).
Refleja las decisiones técnicas tomadas y las justificaciones.

## 1. Vista general

```
 Navegador ──► Next.js (React/TS) ──/api/v1/*──► FastAPI ──► PostgreSQL
                                                   │  └────► Storage (local | S3/MinIO | Azure*)
                                                   ▼
                                                 Redis ──► Celery
                                     ┌────────────┼──────────────┬───────────────┬────────────┐
                                     ▼            ▼              ▼               ▼            ▼
                              worker-audio  worker-diar   worker-transcribe  worker-emotion  worker-satisf/report
                              (FFmpeg,VAD)  (pyannote)    (faster-whisper)   (wav2vec2, GPU)  (CPU)
                                                                          worker-training (GPU)
```

* El frontend nunca procesa audio: sube el archivo (streaming a disco) y consulta el estado.
* Cada etapa del pipeline es **una tarea Celery independiente con su propia cola**. Se
  pueden escalar horizontalmente por etapa (p. ej. N workers GPU solo para emociones).
* `TASK_MODE=inline` ejecuta las mismas etapas en un hilo del proceso API (sin Redis) — solo
  para desarrollo/tests; el código de las etapas es idéntico.
* (*) Azure Blob: la interfaz `Storage` está preparada; implementados `local` y `s3` (S3/MinIO).

## 2. Tecnologías

| Capa | Elección | Motivo |
|---|---|---|
| API | FastAPI + Pydantic v2 | tipado, OpenAPI automática, async |
| ORM | SQLAlchemy 2.0 | PostgreSQL en producción; SQLite solo dev/tests |
| Cola | Celery + Redis | colas por etapa, reintentos, beat para retención |
| Audio | FFmpeg (+`imageio-ffmpeg` de respaldo), soundfile, librosa | lectura por bloques con `seek`, sin cargar todo |
| VAD | Silero VAD (fallback energía) | precisión + independencia de token |
| Diarización | pyannote.audio 3.x (principal), canales estéreo, fallback espectral | ver §5 |
| STT | faster-whisper (motor de WhisperX) con timestamps por palabra; WhisperX opcional | ver §6 |
| Emociones | `r-f/wav2vec-english-speech-emotion-recognition` tras interfaz `EmotionModel` | baseline reemplazable |
| Entrenamiento | PyTorch loop propio + transformers | control de stop/resume/progreso/checkpoints |
| Front | Next.js (App Router) + TypeScript + Tailwind + ECharts | zoom/hover/click nativos en gráficas |
| Auth | JWT (cookie httpOnly + Bearer), bcrypt | mismo origen vía proxy → `<audio>` funciona con cookie |
| PDF/Excel | ReportLab + matplotlib / openpyxl | sin servicios externos |

## 3. Estructura de carpetas

```
backend/
  app/
    api/         routers REST (auth, calls, datasets, training, models, reports, settings, ...)
    core/        config, seguridad, logging estructurado, dispositivo (GPU/CPU), errores
    database/    engine/sesión/Base
    models/      modelos ORM (tablas)
    schemas/     esquemas Pydantic
    services/    lógica de negocio (storage, llamadas, export, PDF, auditoría, cache, datasets…)
    ml/
      audio/         ffmpeg, calidad, VAD, análisis de canales, lectura por bloques
      diarization/   base, pyannote, channels, spectral (fallback), post-proceso
      transcription/ base, faster-whisper, whisperx, asignación palabra→hablante
      emotion/       base (EmotionModel), wav2vec, factory, inference (ventanas)
      prosody/       features de prosodia + interrupciones/pausas
      text/          sentimiento y señales textuales (léxico ES/EN, HF opcional)
      privacy/       detección/anonimización de PII
      satisfaction/  features, engine (reglas configurables), model (regresor entrenable), events
      training/      dataset, augmentation, splits, trainer, evaluator, versioning
    workers/     celery_app, tasks (una por etapa), pipeline (funciones de etapa)
    utils/       helpers
  config/        audio_config.yaml, emotion_config.yaml, satisfaction_config.yaml, app_config.yaml
  scripts/       test_emotion_model.py, benchmark.py, train.py, create_admin.py, make_sample_call.py
  tests/         unit, integración, API, pipeline ML
frontend/        Next.js
docker-compose.yml, docker/…
```

## 4. Flujo de datos y llamadas largas

```
upload (stream a disco, hash SHA-256) ─► UPLOADED ─► QUEUED
 1 PROCESSING_AUDIO : ffprobe/ffmpeg → PCM 16 kHz (por canal si hay separación) + calidad + VAD
 2 DIARIZING        : canales | pyannote (chunks + reconciliación de embeddings) | fallback
 3 TRANSCRIBING     : bloques de ~10 min cortados en silencios; idioma fijado 1 vez; palabra→hablante
 4 ANALYZING_EMOTIONS: ventanas window/hop sobre cada turno; lotes; checkpoint por lote; prosodia
 5 CALCULATING_SATISFACTION: análisis de texto + Satisfaction Engine + eventos + métricas
 6 GENERATING_REPORT: resumen ejecutivo + calidad del análisis (+ PDF bajo demanda)
 ─► COMPLETED (o ERROR con checkpoint conservado → POST /calls/{id}/resume)
```

**Reglas de memoria**: el audio normalizado es WAV PCM16 en disco; todas las lecturas usan
`soundfile.SoundFile.seek/read` de a bloques. Ningún paso carga la grabación completa
(VAD: bloques de 5 min; whisper: bloques de ~10 min; emociones: lotes de ventanas de 5 s;
pyannote: chunks de 30 min). Los temporales se eliminan al terminar (o al fallar de forma
definitiva) y se conservan durante un fallo para permitir la reanudación.

**Ventanas**: `window_size=5 s`, `hop=2.5 s` (configurable). Se generan **dentro de cada turno
de hablante**; el último tramo se ancla al final del turno para no perder emociones cortas.
La línea de tiempo por hablante promedia las ventanas solapadas sobre una rejilla regular.

**Checkpoints** (`calls.checkpoints`, JSON):
`{"audio_processing":{"status":"COMPLETED"}, "diarization":…, "transcription":…,
"emotion_analysis":{"status":"RUNNING","done":312,"total":447}, …}`. Cada etapa es idempotente:
si está `COMPLETED` se omite; la etapa de emociones reanuda desde el último lote persistido.

**Cache** (`services/cache.py`): resultados de etapa en disco indexados por
`hash(audio) + etapa + hash(configuración/modelo)`; mismo audio + mismo modelo/config ⇒ no se recalcula.

## 5. Estrategia de diarización

1. **Estéreo con un hablante por canal** (detección automática: correlación baja entre canales,
   actividad casi exclusiva): no se diariza; cada canal es un hablante con timestamps de VAD, se
   obtienen solapamientos reales y la transcripción/emoción se hacen sobre el canal limpio.
2. **Mono / mezcla**: pyannote (`speaker-diarization-3.1`) con `min_speakers=2`,
   `max_speakers=4`, en chunks de 30 min. Los centroides de embedding devueltos se reconcilian
   entre chunks (coseno) para mantener `SPEAKER_00/01` globales. Se conservan los 2 hablantes
   principales; voces minoritarias se reasignan al más cercano y se **avisa** ("más de dos hablantes").
3. **Fallback local** sin token: clustering de 2 hablantes sobre features MFCC/pitch por ventanas
   (`spectral`). Es real pero de menor precisión; el *diarization quality score* se limita y la
   UI lo advierte.
4. Post-proceso común: fusión de huecos < `merge_gap`, eliminación de turnos ínfimos.
Interfaz `Diarizer` ⇒ se puede sustituir por otro motor.

## 6. Estrategia de transcripción

faster-whisper (mismo motor de WhisperX) con timestamps por palabra. Idioma detectado una vez con
votos de varias muestras y fijado para toda la llamada (prioridad `es`). Las palabras se asignan
al hablante por punto medio de cada palabra contra los turnos ⇒ una frase que cruza un cambio de
turno se parte correctamente. Confianza = media de probabilidades de palabras. `WhisperXTranscriber`
opcional añade alineación forzada.

## 7. Modelo emocional y multimodal

`EmotionModel` (ABC): `load()`, `predict(waveforms, sr) → list[EmotionOutput]`, `labels`,
`sample_rate`, `info()`. `HFAudioEmotionModel` implementa transformers; `Wav2VecEmotionModel`
(baseline Hugging Face) y `FineTunedEmotionModel` (directorio local del registry) lo heredan.
`factory.py` resuelve el modelo activo desde la tabla `models` — cambiar de modelo = activar otro
registro, sin tocar API ni frontend. Siempre se guardan las probabilidades completas.

Fusión multimodal (`satisfaction/features.py`): audio (emociones) + prosodia (tensión) + texto
(sentimiento/cues) + contexto (eventos, evolución). Cada módulo entrega señales normalizadas con su
confianza; el motor las combina con pesos de configuración. Añadir un modelo de texto o un LLM = nueva
implementación de `TextAnalyzer`.

## 8. Satisfaction Engine

Independiente del modelo de emociones. Serie de valencia `v_t = Σ w_e · p_t(e)`; componentes:
señal emocional (ponderada por duración/intensidad/confianza), estabilidad, tendencia,
estado final (ventana final configurable, peso `final_state_weight`), persistencia negativa,
texto y tensión prosódica. `score = 50 + 50·clip(sensibilidad·Σ wᵢcᵢ/Σ wᵢ)`. Todos los pesos y rangos
de interpretación viven en `satisfaction_config.yaml` (editables desde la UI, con override por
organización). Devuelve factores explicables (puntos aportados) y una **confianza propia**
(cobertura + confianza del modelo + calidad de audio), distinta de la probabilidad emocional.
Modo `rules` (defecto), `model` (regresor entrenado) o `hybrid`; calibración lineal con
`satisfaction_feedback` (CSAT/NPS/encuesta).

## 9. Entrenamiento y versionado

* Dataset: muestras (audio, hablante, emoción, satisfacción, start/end, idioma, meta). Validación
  (corruptos, sin etiqueta, duplicados, cortos, sin voz, inconsistencias). Versionado clonando
  (`dataset_v1→v2`) con distribución de clases y modelo usado.
* Split 70/15/15 configurable **agrupado por persona/llamada** y estratificado (sin fuga de datos).
* Desbalance: alerta + `class weights` / oversampling / augmentation segura por etiqueta.
* Trainer: AdamW, warmup lineal, gradient accumulation, early stopping por Macro-F1,
  checkpoints por época (reanudable), stop cooperativo, progreso por época/paso en BD.
* Evaluación: accuracy, precision, recall, F1, macro/weighted F1, matriz de confusión, por clase.
* Registry: `models` (TRAINING → VALIDATION → PRODUCTION → ARCHIVED). Activar un modelo ⇒ nuevas
  llamadas lo usan; cada análisis guarda `model_id` y `model_version`.

## 10. Seguridad, privacidad y multi-tenant

* Todas las tablas de negocio llevan `org_id`; todas las consultas se filtran por la organización del token.
* Roles: ADMIN (todo), ANALYST (llamadas/datasets/entrenamiento), VIEWER (solo lectura).
* Cifrado en reposo opcional (AES-256-GCM por bloques con lectura aleatoria para servir audio con Range).
* Procesamiento 100 % local; ningún audio sale a APIs externas. Descarga de modelos desde Hugging Face
  solo al inicializar la caché (`MODEL_CACHE_DIR`).
* `allow_training` por llamada, **desactivado por defecto**; los datasets solo aceptan llamadas con ese permiso
  (o etiquetado explícito por un usuario autorizado).
* Retención configurable + purga (Celery beat), eliminación granular, auditoría (`audit_logs`).
* PII: detector por reglas ES/EN; transcripción original y redactada; exportaciones con opción `redact`.
  *No implementado*: anonimización del propio audio.

## 11. Esquema de base de datos

```
organizations 1─* users
organizations 1─* calls 1─* speakers 1─* audio_segments 1─* transcriptions
                    │                       └─* emotion_predictions (segment_id, model_id)
                    ├─* satisfaction_scores (por hablante + interacción)
                    ├─* critical_events
                    └─* satisfaction_feedback (real vs predicha)
models 1─* model_metrics ; training_datasets 1─* training_samples ; training_runs (dataset, model)
settings (org_id, key, value JSON) ; audit_logs
```

## 12. Diseño para escala futura

Colas por etapa, workers sin estado, storage abstracto (S3/MinIO), `org_id` en todo, configuración
por organización, modelos por idioma/tipo (`models.language`, `models.kind`), nuevos analizadores
(intención, cumplimiento, resolución, CSAT/NPS) como etapas adicionales del pipeline.
