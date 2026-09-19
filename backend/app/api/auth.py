"""Autenticación (JWT en cookie httpOnly + Bearer), registro de organización y gestión de usuarios."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import ACCESS_COOKIE, REFRESH_COOKIE, AdminUser, client_ip, get_current_user
from app.core.config import get_settings
from app.core.errors import AppError, Conflict, Forbidden, NotFound, Unauthorized
from app.core.security import (create_access_token, create_refresh_token, decode_token, hash_password,
                               verify_password)
from app.database.base import get_db
from app.models import Organization, Role, User
from app.schemas.common import LoginIn, Msg, RegisterIn, TokenOut, UserCreate, UserOut, UserUpdate
from app.services.audit import audit

router = APIRouter(tags=["auth"])


def _set_cookies(resp: Response, user: User) -> str:
    s = get_settings()
    access = create_access_token(user.id, user.org_id, user.role)
    refresh = create_refresh_token(user.id)
    common = dict(httponly=True, samesite="lax", secure=s.cookie_secure, path="/")
    resp.set_cookie(ACCESS_COOKIE, access, max_age=s.access_token_minutes * 60, **common)
    resp.set_cookie(REFRESH_COOKIE, refresh, max_age=s.refresh_token_days * 86400, **common)
    return access


@router.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(func.lower(User.email) == body.email.lower()))
    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
        audit(db, org_id=None, user_id=None, action="auth.login_failed", details={"email": body.email}, ip=client_ip(request))
        raise Unauthorized("Correo o contraseña incorrectos.")
    token = _set_cookies(response, user)
    audit(db, org_id=user.org_id, user_id=user.id, action="auth.login", ip=client_ip(request))
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.post("/auth/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    tok = request.cookies.get(REFRESH_COOKIE)
    if not tok:
        raise Unauthorized()
    payload = decode_token(tok, "refresh")
    user = db.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise Unauthorized()
    return TokenOut(access_token=_set_cookies(response, user), user=UserOut.model_validate(user))


@router.post("/auth/logout", response_model=Msg)
def logout(response: Response):
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/")
    return Msg(message="Sesión cerrada.")


@router.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/auth/register", response_model=TokenOut)
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_db)):
    """Crea una organización nueva con su ADMIN. Solo si no hay usuarios (primer arranque) o ALLOW_SIGNUP=true."""
    s = get_settings()
    n_users = db.scalar(select(func.count()).select_from(User)) or 0
    if n_users > 0 and not s.allow_signup:
        raise Forbidden("El registro público está deshabilitado. Solicite acceso a un administrador.")
    if db.scalar(select(User.id).where(func.lower(User.email) == body.email.lower())):
        raise Conflict("Ya existe un usuario con ese correo.")
    if db.scalar(select(Organization.id).where(Organization.name == body.organization)):
        raise Conflict("Ya existe una organización con ese nombre.")
    org = Organization(name=body.organization)
    db.add(org)
    db.flush()
    user = User(org_id=org.id, email=body.email.lower(), name=body.name, hashed_password=hash_password(body.password),
                role=Role.ADMIN.value)
    db.add(user)
    db.commit()
    audit(db, org_id=org.id, user_id=user.id, action="org.create", entity="organization", entity_id=org.id, ip=client_ip(request))
    return TokenOut(access_token=_set_cookies(response, user), user=UserOut.model_validate(user))


@router.get("/auth/setup-status")
def setup_status(db: Session = Depends(get_db)):
    """Indica si aún no existen usuarios (para mostrar el asistente de primer registro)."""
    n = db.scalar(select(func.count()).select_from(User)) or 0
    return {"needs_setup": n == 0, "signup_enabled": n == 0 or get_settings().allow_signup}


# --- usuarios de la organización (ADMIN) --------------------------------------------------------------------
@router.get("/users", response_model=list[UserOut])
def list_users(user: User = AdminUser, db: Session = Depends(get_db)):
    return db.scalars(select(User).where(User.org_id == user.org_id).order_by(User.created_at)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, request: Request, user: User = AdminUser, db: Session = Depends(get_db)):
    if body.role not in {r.value for r in Role}:
        raise AppError("Rol inválido.", status_code=422)
    if db.scalar(select(User.id).where(func.lower(User.email) == body.email.lower())):
        raise Conflict("Ya existe un usuario con ese correo.")
    u = User(org_id=user.org_id, email=body.email.lower(), name=body.name, role=body.role,
             hashed_password=hash_password(body.password))
    db.add(u)
    db.commit()
    audit(db, org_id=user.org_id, user_id=user.id, action="user.create", entity="user", entity_id=u.id,
          details={"role": u.role}, ip=client_ip(request))
    return u


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: str, body: UserUpdate, request: Request, user: User = AdminUser, db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if u is None or u.org_id != user.org_id:
        raise NotFound("El usuario no existe.")
    if body.role is not None:
        if body.role not in {r.value for r in Role}:
            raise AppError("Rol inválido.", status_code=422)
        if u.id == user.id and body.role != Role.ADMIN.value:
            raise Conflict("No puede quitarse a sí mismo el rol de administrador.")
        u.role = body.role
    if body.name is not None:
        u.name = body.name
    if body.is_active is not None:
        if u.id == user.id and not body.is_active:
            raise Conflict("No puede desactivar su propia cuenta.")
        u.is_active = body.is_active
    if body.password:
        u.hashed_password = hash_password(body.password)
    db.commit()
    audit(db, org_id=user.org_id, user_id=user.id, action="user.update", entity="user", entity_id=u.id, ip=client_ip(request))
    return u
