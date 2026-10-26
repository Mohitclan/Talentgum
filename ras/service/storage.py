"""Object storage interface. LocalStore keeps files on disk; an S3 store can replace it later."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get_path(self, key: str) -> Path: ...
    def exists(self, key: str) -> bool: ...


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError("key escapes the store root")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic: a reader never sees half a file

    def get_path(self, key: str) -> Path:
        return self._p(key)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
