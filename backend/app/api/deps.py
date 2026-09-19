"""Dependencias comunes: usuario actual (Bearer o cookie), roles y aislamiento multi-tenant."""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.errors import Forbidden, NotFound, Unauthorized
from app.core.security import decode_token
from app.database.base import get_db
from app.models import Call, Role, User

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    token = token or request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise Unauthorized()
    payload = decode_token(token, "access")
    user = db.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise Unauthorized()
    request.state.user = user
    return user


def require_roles(*roles: Role) -> Callable:
    allowed = {r.value for r in roles} | {Role.ADMIN.value}     # ADMIN puede todo

    def dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise Forbidden()
        return user
    return dep


# Atajos por permiso
AnyUser = Depends(get_current_user)
AnalystUser = Depends(require_roles(Role.ANALYST))
AdminUser = Depends(require_roles())            # solo ADMIN


def get_call_or_404(db: Session, user: User, call_id: str) -> Call:
    """Toda consulta de llamadas se filtra por organización: nunca se accede a datos de otra org."""
    call = db.get(Call, call_id)
    if call is None or call.org_id != user.org_id:
        raise NotFound("La llamada solicitada no existe.")
    return call


def client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None
