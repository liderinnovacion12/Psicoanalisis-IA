"""Crea una organización y un usuario ADMIN.   python scripts/create_admin.py --org "Mi Empresa" --email admin@empresa.com"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import func, select  # noqa: E402

from app.core.security import hash_password  # noqa: E402
from app.database.base import init_db, session_scope  # noqa: E402
from app.ml.emotion.factory import ensure_builtin_model  # noqa: E402
from app.models import Organization, Role, User  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--org", required=True)
ap.add_argument("--email", required=True)
ap.add_argument("--name", default="Administrador")
ap.add_argument("--password")
a = ap.parse_args()
pw = a.password or getpass.getpass("Contraseña (mín. 8): ")
if len(pw) < 8:
    sys.exit("La contraseña debe tener al menos 8 caracteres.")
init_db()
with session_scope() as db:
    ensure_builtin_model(db)
    if db.scalar(select(User.id).where(func.lower(User.email) == a.email.lower())):
        sys.exit("Ya existe un usuario con ese correo.")
    org = db.scalar(select(Organization).where(Organization.name == a.org)) or Organization(name=a.org)
    db.add(org)
    db.flush()
    db.add(User(org_id=org.id, email=a.email.lower(), name=a.name, hashed_password=hash_password(pw), role=Role.ADMIN.value))
print(f"Administrador {a.email} creado en la organización «{a.org}».")
