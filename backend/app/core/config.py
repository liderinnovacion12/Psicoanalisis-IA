"""Configuración de entorno (variables de entorno / .env) y configuración YAML centralizada.

* `Settings`  -> infraestructura (BD, Redis, rutas, secretos). Viene del entorno.
* `ConfigStore` -> parámetros de negocio/ML (audio, emociones, satisfacción, app). Vienen de
  `config/*.yaml` y pueden sobrescribirse por organización (tabla `settings`).
"""
from __future__ import annotations

import copy
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR_DEFAULT = BACKEND_DIR / "config"

CONFIG_SECTIONS = {
    "audio": "audio_config.yaml",
    "emotion": "emotion_config.yaml",
    "satisfaction": "satisfaction_config.yaml",
    "app": "app_config.yaml",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    # Infraestructura
    database_url: str = "sqlite:///./data/app.db"
    redis_url: str = "redis://localhost:6379/0"
    task_mode: str = Field("inline", description="inline (hilo, sin Redis) | celery")
    inline_workers: int = 1
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    report_dir: Path = Path("./data/reports")
    model_cache_dir: Path = Path("./data/model_cache")
    models_dir: Path = Path("./data/models")          # modelos entrenados (registry)
    temp_dir: Path = Path("./data/tmp")
    cache_dir: Path = Path("./data/cache")
    config_dir: Path = CONFIG_DIR_DEFAULT

    # Modelo / dispositivo
    model_name: str = "r-f/wav2vec-english-speech-emotion-recognition"
    device: str = "auto"                              # auto | cuda | cpu
    max_file_size: int = 4 * 1024 ** 3                # bytes (por defecto 4 GB)
    hf_token: str | None = None                       # necesario para pyannote (modelo con acceso restringido)
    hf_offline: bool = False

    # Seguridad
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    encryption_key: str | None = None                 # si no se define, se deriva de secret_key
    access_token_minutes: int = 60 * 8
    refresh_token_days: int = 7
    allow_signup: bool = False                        # registro público (crea organización nueva)
    cookie_secure: bool = False
    cors_origins: str = "http://localhost:3000"
    first_admin_email: str | None = None
    first_admin_password: str | None = None

    # Almacenamiento
    storage_backend: str = "local"                    # local | s3
    s3_endpoint_url: str | None = None
    s3_bucket: str = "call-analyzer"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str | None = None

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.report_dir, self.model_cache_dir,
                  self.models_dir, self.temp_dir, self.cache_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigStore:
    """Lee los YAML una vez (con recarga manual) y combina con overrides de organización."""

    def __init__(self, config_dir: Path | None = None):
        self._dir = Path(config_dir or get_settings().config_dir)
        self._defaults: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            for section, fname in CONFIG_SECTIONS.items():
                path = self._dir / fname
                with open(path, "r", encoding="utf-8") as f:
                    self._defaults[section] = yaml.safe_load(f) or {}

    def defaults(self, section: str) -> dict:
        return copy.deepcopy(self._defaults[section])

    def resolve(self, section: str, overrides: dict | None = None) -> dict:
        return deep_merge(self._defaults[section], overrides or {})


_store: ConfigStore | None = None


def get_config_store() -> ConfigStore:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store


def reset_config_store() -> None:
    global _store
    _store = None
