"""Storage abstracto: local (con cifrado opcional) y S3/MinIO. Los archivos grandes NUNCA van a PostgreSQL."""
from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from app.core.config import Settings, get_settings
from app.utils import crypto

CHUNK_IO = 1024 * 1024


class Storage(ABC):
    @abstractmethod
    def put_stream(self, key: str, chunks: Iterator[bytes]) -> int: ...
    @abstractmethod
    def put_file(self, key: str, path: str | Path) -> int: ...
    @abstractmethod
    def size(self, key: str) -> int: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def open_range(self, key: str, start: int = 0, end: int | None = None) -> Iterator[bytes]: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...
    @abstractmethod
    def _materialize(self, key: str, dst: Path) -> None: ...

    @contextmanager
    def local_path(self, key: str, suffix: str | None = None) -> Iterator[Path]:
        """Entrega una copia en claro en disco local (temporal) para FFmpeg/soundfile."""
        tmp_dir = get_settings().temp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        suffix = suffix or Path(key).suffix
        fd, name = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
        os.close(fd)
        p = Path(name)
        try:
            self._materialize(key, p)
            yield p
        finally:
            p.unlink(missing_ok=True)


class LocalStorage(Storage):
    def __init__(self, root: Path, encrypt: bool = False, key: bytes | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encrypt = encrypt
        self.key = key

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError("key fuera del almacenamiento")   # anti path-traversal
        return p

    def put_stream(self, key: str, chunks: Iterator[bytes]) -> int:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        if self.encrypt:
            with crypto.EncryptedWriter(p, self.key) as w:  # type: ignore[arg-type]
                for c in chunks:
                    w.write(c)
                    total += len(c)
        else:
            with open(p, "wb") as f:
                for c in chunks:
                    f.write(c)
                    total += len(c)
        return total

    def put_file(self, key: str, path: str | Path) -> int:
        def gen() -> Iterator[bytes]:
            with open(path, "rb") as f:
                while b := f.read(CHUNK_IO):
                    yield b
        return self.put_stream(key, gen())

    def _encrypted(self, p: Path) -> bool:
        return self.encrypt or crypto.is_encrypted(p)

    def size(self, key: str) -> int:
        p = self._p(key)
        return crypto.plain_size(p) if crypto.is_encrypted(p) else p.stat().st_size

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def open_range(self, key: str, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        p = self._p(key)
        if crypto.is_encrypted(p):
            if not self.key:
                raise RuntimeError("archivo cifrado sin clave configurada")
            yield from crypto.read_range(p, self.key, start, end)
            return
        size = p.stat().st_size
        end = size - 1 if end is None or end >= size else end
        remaining = end - start + 1
        with open(p, "rb") as f:
            f.seek(start)
            while remaining > 0:
                b = f.read(min(CHUNK_IO, remaining))
                if not b:
                    break
                remaining -= len(b)
                yield b

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        p = self._p(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)

    def _materialize(self, key: str, dst: Path) -> None:
        p = self._p(key)
        if crypto.is_encrypted(p):
            crypto.decrypt_to_file(p, dst, self.key)  # type: ignore[arg-type]
        else:
            shutil.copyfile(p, dst)


class S3Storage(Storage):
    """S3 / MinIO / cualquier servicio compatible. Cifrado: usar SSE del servicio (o disco cifrado).
    Nota: implementado y revisado, pero no ejecutado contra un bucket real en esta entrega."""

    def __init__(self, s: Settings):
        import boto3  # import diferido
        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3", endpoint_url=s.s3_endpoint_url, aws_access_key_id=s.s3_access_key,
            aws_secret_access_key=s.s3_secret_key, region_name=s.s3_region)
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put_stream(self, key: str, chunks: Iterator[bytes]) -> int:
        # Se escribe en un temporal para usar multipart automático de upload_fileobj
        fd, name = tempfile.mkstemp(dir=get_settings().temp_dir)
        total = 0
        try:
            with os.fdopen(fd, "wb") as f:
                for c in chunks:
                    f.write(c)
                    total += len(c)
            self.client.upload_file(name, self.bucket, key)
        finally:
            Path(name).unlink(missing_ok=True)
        return total

    def put_file(self, key: str, path: str | Path) -> int:
        self.client.upload_file(str(path), self.bucket, key)
        return Path(path).stat().st_size

    def size(self, key: str) -> int:
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def open_range(self, key: str, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        rng = f"bytes={start}-{'' if end is None else end}"
        body = self.client.get_object(Bucket=self.bucket, Key=key, Range=rng)["Body"]
        for b in body.iter_chunks(CHUNK_IO):
            yield b

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        pag = self.client.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})

    def _materialize(self, key: str, dst: Path) -> None:
        self.client.download_file(self.bucket, key, str(dst))


_storage: Storage | None = None


def get_storage(encrypt: bool | None = None) -> Storage:
    global _storage
    if _storage is None:
        s = get_settings()
        if s.storage_backend == "s3":
            _storage = S3Storage(s)
        else:
            from app.core.config import get_config_store
            enc = get_config_store().defaults("app")["privacy"]["encrypt_at_rest"] if encrypt is None else encrypt
            key = crypto.derive_key(s.encryption_key or s.secret_key)
            _storage = LocalStorage(s.upload_dir, encrypt=bool(enc), key=key)
    return _storage


def reset_storage() -> None:
    global _storage
    _storage = None


def call_key(org_id: str, call_id: str, name: str) -> str:
    return f"orgs/{org_id}/calls/{call_id}/{name}"
