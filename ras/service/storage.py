"""Object storage: LocalStore keeps files on disk, S3Store keeps them in an S3 bucket.

Keys are content-addressed (they contain the SHA-256), so an object never changes once written and a
local copy downloaded from S3 stays valid for good.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get_path(self, key: str) -> Path: ...  # a local file with the object's bytes
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...  # drop any local copy of the object


def _safe_join(root: Path, key: str) -> Path:
    p = (root / key).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError("key escapes the store root")
    return p


def _atomic_write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(p)  # atomic: a reader never sees half a file


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return _safe_join(self.root, key)

    def put(self, key: str, data: bytes) -> None:
        _atomic_write(self._p(key), data)

    def get_path(self, key: str) -> Path:
        return self._p(key)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def release(self, key: str) -> None:
        pass  # the file is the stored object itself


class S3Store:
    """S3 (or any S3-compatible store, e.g. SeaweedFS locally).

    Endpoint and credentials come from the standard AWS environment (`AWS_ENDPOINT_URL_S3`, the task
    role on ECS). Audio objects are tagged `kind=audio` so a bucket lifecycle rule can expire them as a
    backstop to the application's own retention purge.
    """

    def __init__(self, bucket: str, cache_dir: str | Path, prefix: str = "", client=None, create_bucket=False):
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.s3, self.bucket, self.prefix = client, bucket, prefix.strip("/")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        if create_bucket:  # local stand-ins only; on AWS the bucket comes from Terraform
            try:
                self.s3.head_bucket(Bucket=bucket)
            except Exception:
                try:
                    self.s3.create_bucket(Bucket=bucket)
                except Exception:  # another process created it first
                    self.s3.head_bucket(Bucket=bucket)

    def _k(self, key: str) -> str:
        if ".." in key.split("/"):
            raise ValueError("key escapes the store root")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        tag = "kind=audio" if "/audio/" in f"/{key}" else "kind=other"
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, Tagging=tag)

    def get_path(self, key: str) -> Path:
        p = _safe_join(self.cache, key)
        if not p.exists():
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._k(key))
            _atomic_write(p, obj["Body"].read())
        return p

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self._k(key))
        self.release(key)

    def release(self, key: str) -> None:
        """Drop the local copy (recordings are read once; keeps a long-running worker's disk small)."""
        _safe_join(self.cache, key).unlink(missing_ok=True)


def make_store(settings) -> ObjectStore:
    if settings.storage == "s3":
        if not settings.s3_bucket:
            raise ValueError("RAS_STORAGE=s3 needs RAS_S3_BUCKET")
        return S3Store(settings.s3_bucket, settings.data_dir / "s3_cache", settings.s3_prefix,
                       create_bucket=settings.create_resources)
    return LocalStore(settings.data_dir / "objects")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
