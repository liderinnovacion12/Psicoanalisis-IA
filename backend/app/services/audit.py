"""Auditoría: quién, qué, cuándo, con qué modelo/configuración."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models import AuditLog

log = get_logger("audit")


def audit(db: Session, *, org_id: str | None, user_id: str | None, action: str, entity: str | None = None,
          entity_id: str | None = None, details: dict | None = None, ip: str | None = None,
          commit: bool = True) -> None:
    db.add(AuditLog(org_id=org_id, user_id=user_id, action=action, entity=entity, entity_id=entity_id,
                    details=details or {}, ip=ip))
    if commit:
        db.commit()
    log.info("audit", extra={"org_id": org_id, "user_id": user_id, "action": action, "entity": entity,
                             "entity_id": entity_id})
