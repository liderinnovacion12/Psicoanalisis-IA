"""Cifrado en reposo por bloques (AES-256-GCM) con lectura aleatoria.

Formato de archivo:  MAGIC(6) | nonce_prefix(8) | plain_size(8, BE) | [ chunk_i ]...
chunk_i = AESGCM(key).encrypt(nonce_prefix || i(4, BE), plaintext_i)   (CHUNK bytes de texto plano; +16 de tag)
Permite servir peticiones HTTP Range sin descifrar el archivo completo.
"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"CAENC1"
HEADER = len(MAGIC) + 8 + 8
CHUNK = 1024 * 1024
TAG = 16


def derive_key(secret: str) -> bytes:
    return hashlib.sha256(("call-analyzer|" + secret).encode()).digest()


def is_encrypted(path: str | Path) -> bool:
    with open(path, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC


def _nonce(prefix: bytes, i: int) -> bytes:
    return prefix + struct.pack(">I", i)


class EncryptedWriter:
    """Escribe texto plano y lo cifra por bloques. Uso: `with EncryptedWriter(path, key) as w: w.write(b)`."""

    def __init__(self, path: str | Path, key: bytes):
        self._f: BinaryIO = open(path, "wb")
        self._aes = AESGCM(key)
        self._prefix = os.urandom(8)
        self._buf = bytearray()
        self._i = 0
        self._size = 0
        self._f.write(MAGIC + self._prefix + struct.pack(">Q", 0))

    def write(self, data: bytes) -> None:
        self._buf.extend(data)
        self._size += len(data)
        while len(self._buf) >= CHUNK:
            self._flush(bytes(self._buf[:CHUNK]))
            del self._buf[:CHUNK]

    def _flush(self, block: bytes) -> None:
        self._f.write(self._aes.encrypt(_nonce(self._prefix, self._i), block, None))
        self._i += 1

    def close(self) -> int:
        if self._buf:
            self._flush(bytes(self._buf))
            self._buf.clear()
        self._f.seek(len(MAGIC) + 8)
        self._f.write(struct.pack(">Q", self._size))
        self._f.close()
        return self._size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def plain_size(path: str | Path) -> int:
    with open(path, "rb") as f:
        head = f.read(HEADER)
    return struct.unpack(">Q", head[len(MAGIC) + 8:HEADER])[0]


def read_range(path: str | Path, key: bytes, start: int = 0, end: int | None = None,
               block: int = CHUNK) -> Iterator[bytes]:
    """Genera texto plano [start, end] (inclusive) leyendo solo los bloques necesarios."""
    aes = AESGCM(key)
    with open(path, "rb") as f:
        head = f.read(HEADER)
        prefix = head[len(MAGIC):len(MAGIC) + 8]
        size = struct.unpack(">Q", head[len(MAGIC) + 8:HEADER])[0]
        if end is None or end >= size:
            end = size - 1
        if size == 0 or start > end:
            return
        first, last = start // CHUNK, end // CHUNK
        for i in range(first, last + 1):
            f.seek(HEADER + i * (CHUNK + TAG))
            ct = f.read(CHUNK + TAG)
            pt = aes.decrypt(_nonce(prefix, i), ct, None)
            lo = start - i * CHUNK if i == first else 0
            hi = end - i * CHUNK + 1 if i == last else len(pt)
            yield pt[lo:hi]


def decrypt_to_file(src: str | Path, dst: str | Path, key: bytes) -> None:
    with open(dst, "wb") as out:
        for part in read_range(src, key):
            out.write(part)
