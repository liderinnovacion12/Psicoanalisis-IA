"""Cache de resultados de etapa en disco: mismo audio + misma configuración/modelo ⇒ no se recalcula."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def config_hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _path(org_id: str, file_hash: str, stage: str, cfg_hash: str) -> Path:
    # el cache es por organización: una organización nunca lee resultados de otra
    return get_settings().cache_dir / org_id / file_hash[:2] / f"{file_hash}_{stage}_{cfg_hash}.json"


def cache_get(org_id: str, file_hash: str | None, stage: str, cfg_hash: str) -> Any | None:
    if not file_hash:
        return None
    p = _path(org_id, file_hash, stage, cfg_hash)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
    return None


def cache_put(org_id: str, file_hash: str | None, stage: str, cfg_hash: str, value: Any) -> None:
    if not file_hash:
        return
    p = _path(org_id, file_hash, stage, cfg_hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def cache_purge(org_id: str, file_hash: str | None) -> None:
    if not file_hash:
        return
    d = get_settings().cache_dir / org_id / file_hash[:2]
    for p in d.glob(f"{file_hash}_*.json"):
        p.unlink(missing_ok=True)
