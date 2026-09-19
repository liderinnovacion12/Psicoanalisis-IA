"""Reportes y exportaciones: PDF, CSV, Excel, JSON."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import AnyUser, client_ip, get_call_or_404
from app.core.errors import Conflict
from app.database.base import get_db
from app.models import User
from app.services import export as exp
from app.services.audit import audit
from app.services.config_service import get_all
from app.services.report_pdf import build_pdf

router = APIRouter(tags=["reports"])

MEDIA = {"pdf": "application/pdf", "csv": "text/csv; charset=utf-8", "json": "application/json",
         "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def _render(db: Session, user: User, call_id: str, fmt: str, redact: bool | None, request: Request) -> Response:
    call = get_call_or_404(db, user, call_id)
    if call.status != "COMPLETED":
        raise Conflict("La llamada aún no tiene resultados disponibles.")
    cfg = get_all(db, user.org_id)["app"]
    if redact is None:
        redact = bool(cfg["privacy"]["redact_exports_by_default"])
    data = exp.gather(db, call, redact, cfg["general"]["roles_labels"])
    body = {"pdf": lambda: build_pdf(data), "csv": lambda: exp.to_csv(data), "xlsx": lambda: exp.to_xlsx(data),
            "json": lambda: exp.to_json(data)}[fmt]()
    audit(db, org_id=user.org_id, user_id=user.id, action=f"export.{fmt}", entity="call", entity_id=call.id,
          details={"redact": redact}, ip=client_ip(request))
    return Response(body, media_type=MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="llamada_{call.id[:8]}.{fmt}"'})


@router.get("/reports/{call_id}")
def report_pdf(call_id: str, request: Request, redact: bool | None = None, user: User = AnyUser, db: Session = Depends(get_db)):
    return _render(db, user, call_id, "pdf", redact, request)


@router.get("/calls/{call_id}/export")
def export_call(call_id: str, request: Request, format: str = Query("json", pattern="^(pdf|csv|xlsx|json)$"),
                redact: bool | None = None, user: User = AnyUser, db: Session = Depends(get_db)):
    return _render(db, user, call_id, format, redact, request)
