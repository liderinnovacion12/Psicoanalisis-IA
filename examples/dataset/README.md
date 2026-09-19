# Ejemplo de dataset (FORMATO)

`manifest.example.csv` muestra el **formato** de un manifiesto. **No contiene audios ni etiquetas reales**: los
nombres de archivo son marcadores. Para usarlo con sus propios datos:

1. Coloque sus grabaciones en una carpeta (`audio/`) y edite el CSV con **sus** etiquetas (una fila = un segmento).
2. Importe desde la UI: *Dataset → Importar CSV + ZIP*, o por línea de comandos:

```bash
cd backend
python scripts/train.py --manifest ../examples/dataset/manifest.csv --audio-dir ../examples/dataset \
    --output data/models/mi_modelo_es --epochs 10
```

Columnas: `audio` (ruta relativa), `speaker`, `emotion` (angry, disgust, fear, happy, neutral, sad, surprise o una categoría
propia), `satisfaction` (0-100, opcional), `start`/`end` (segundos o `HH:MM:SS`, opcionales), `language`,
`speaker_group` (persona/llamada; **es lo que evita la fuga de datos entre TRAIN y TEST**).

> Nota: el audio sintético de `examples/call/` es voz TTS de emoción plana. NO sirve como dataset emocional; sirve solo para
> probar el flujo de la aplicación.
