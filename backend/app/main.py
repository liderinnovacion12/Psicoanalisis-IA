"""Aplicación FastAPI. Punto de entrada: `uvicorn app.main:app`."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger, setup_logging

settings = get_settings()
setup_logging(settings.log_level, settings.log_json, settings.data_dir / "logs")
log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.security import hash_password
    from app.database.base import init_db, session_scope
    from app.ml.emotion.factory import ensure_builtin_model
    from app.models import Organization, Role, User
    init_db()
    with session_scope() as db:
        ensure_builtin_model(db)
        s = get_settings()
        if s.first_admin_email and s.first_admin_password and not db.scalar(select(func.count()).select_from(User)):
            org = Organization(name="Organización principal")
            db.add(org)
            db.flush()
            db.add(User(org_id=org.id, email=s.first_admin_email.lower(), name="Administrador",
                        hashed_password=hash_password(s.first_admin_password), role=Role.ADMIN.value))
            log.info("Administrador inicial creado", extra={"email": s.first_admin_email})
    if settings.secret_key.startswith("CHANGE_ME"):
        log.warning("SECRET_KEY por defecto: cámbiela en producción")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Analizador de Llamadas IA", version="1.0.0", lifespan=lifespan,
                  description="Análisis de emociones y satisfacción en llamadas telefónicas (procesamiento local).")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        if exc.detail:
            log.warning("app_error", extra={"code": exc.code, "detail": exc.detail, "path": request.url.path})
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.user_message})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        log.info("validation_error", extra={"errors": exc.errors()[:5], "path": request.url.path})
        return JSONResponse(status_code=422, content={"error": "validation_error",
                                                       "message": "Los datos enviados no son válidos. Revise el formulario."})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        ref = uuid.uuid4().hex[:8]
        log.error("unhandled_exception", extra={"ref": ref, "path": request.url.path}, exc_info=exc)
        return JSONResponse(status_code=500, content={"error": "internal_error", "ref": ref,
                                                       "message": "Ocurrió un error inesperado. Intente nuevamente."})

    from app.api import auth, calls, datasets, models_api, reports, settings_api, system, training
    for r in (auth.router, calls.router, datasets.router, training.router, models_api.router, reports.router,
              settings_api.router, system.router):
        app.include_router(r, prefix="/api/v1")
    return app


app = create_app()
