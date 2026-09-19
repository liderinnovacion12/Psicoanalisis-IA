"""Logging estructurado (JSON) con niveles INFO / WARNING / ERROR separados en archivos."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _Only(logging.Filter):
    def __init__(self, lo: int, hi: int):
        super().__init__()
        self.lo, self.hi = lo, hi

    def filter(self, record: logging.LogRecord) -> bool:
        return self.lo <= record.levelno <= self.hi


def setup_logging(level: str = "INFO", as_json: bool = True, log_dir: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt: logging.Formatter = JsonFormatter() if as_json else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for name, lo, hi in (("info", logging.INFO, logging.INFO),
                             ("warning", logging.WARNING, logging.WARNING),
                             ("error", logging.ERROR, logging.CRITICAL)):
            fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=10_000_000, backupCount=5,
                                     encoding="utf-8")
            fh.setFormatter(fmt)
            fh.addFilter(_Only(lo, hi))
            root.addHandler(fh)
    for noisy in ("urllib3", "botocore", "matplotlib", "numba", "PIL", "httpx", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
