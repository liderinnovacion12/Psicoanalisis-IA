"""Engine, sesión y Base declarativa. PostgreSQL en producción; SQLite solo en desarrollo/tests."""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import JSON, DateTime, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

JSONType = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None


def _make_engine(url: str):
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs.update(pool_size=10, max_overflow=20)
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragma(dbapi_con, _):  # noqa: ANN001
            cur = dbapi_con.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return eng


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _make_engine(get_settings().database_url)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, class_=Session)
    return _engine


def reset_engine() -> None:
    """Usado en tests para cambiar la BD."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def SessionLocal() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registra tablas)
    Base.metadata.create_all(get_engine())


__all__ = ["Base", "JSONType", "new_id", "utcnow", "DateTime", "get_db", "session_scope",
           "SessionLocal", "init_db", "get_engine", "reset_engine"]
