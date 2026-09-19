"""Hash de contraseñas (bcrypt) y JWT (PyJWT)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.errors import Unauthorized

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode())
    except ValueError:
        return False


def _create(sub: str, kind: str, delta: timedelta, extra: dict | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "typ": kind, "iat": now, "exp": now + delta, "jti": uuid.uuid4().hex}
    payload.update(extra or {})
    return jwt.encode(payload, get_settings().secret_key, algorithm=ALGORITHM)


def create_access_token(user_id: str, org_id: str, role: str) -> str:
    s = get_settings()
    return _create(user_id, "access", timedelta(minutes=s.access_token_minutes),
                   {"org": org_id, "role": role})


def create_refresh_token(user_id: str) -> str:
    return _create(user_id, "refresh", timedelta(days=get_settings().refresh_token_days))


def decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, get_settings().secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:  # expirado, firma inválida, etc.
        raise Unauthorized() from e
    if payload.get("typ") != expected_type:
        raise Unauthorized()
    return payload
