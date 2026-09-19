"""Política de retención: elimina llamadas vencidas (tarea periódica de Celery beat)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.core.logging import get_logger
from app.database.base import session_scope
from app.models import Call
from app.services.call_service import delete_call_data

log = get_logger(__name__)


def purge_expired() -> int:
    n = 0
    with session_scope() as db:
        now = datetime.now(timezone.utc)
        for c in db.scalars(select(Call).where(Call.retention_until.is_not(None), Call.retention_until < now)).all():
            delete_call_data(db, c, "all", None)
            n += 1
    log.info("retención: llamadas eliminadas", extra={"count": n})
    return n
