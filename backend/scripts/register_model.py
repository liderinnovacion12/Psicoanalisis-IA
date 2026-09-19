"""Registra un checkpoint Hugging Face local (p. ej. un modelo entrenado en español) en el Model Registry.

    python scripts/register_model.py --org "Mi Empresa" --path models/custom_emotion_model --name emotion_es_v1 --language es
Queda en estado VALIDATION; actívelo desde la pantalla «Modelos». No requiere cambios en el backend ni en el frontend.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select  # noqa: E402

from app.database.base import init_db, session_scope  # noqa: E402
from app.ml.emotion.factory import ensure_builtin_model  # noqa: E402
from app.models import MLModel, Organization  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--org", required=True)
ap.add_argument("--path", required=True, help="directorio con config.json, pesos y preprocessor_config.json")
ap.add_argument("--name", required=True)
ap.add_argument("--version", default="v1")
ap.add_argument("--language", default=None)
a = ap.parse_args()
p = Path(a.path).resolve()
cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
labels = [cfg["id2label"][str(i)].lower() for i in range(len(cfg["id2label"]))]
init_db()
with session_scope() as db:
    ensure_builtin_model(db)
    org = db.scalar(select(Organization).where(Organization.name == a.org))
    if org is None:
        sys.exit("Organización no encontrada.")
    m = MLModel(org_id=org.id, name=a.name, version=a.version, kind="emotion", loader="finetuned", path=str(p), language=a.language,
                labels=labels, status="VALIDATION", parameters={"registered_manually": True})
    db.add(m)
    db.flush()
    print(f"Registrado {a.name} ({a.version}) con etiquetas {labels}. Estado: VALIDATION (actívelo en la UI).")
