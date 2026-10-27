"""Settings from the environment. Defaults run everything locally (SQLite, files on disk, DB-polled queue);
on AWS the same code runs on PostgreSQL, S3 and SQS (see infra/terraform)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

DEV_WEBHOOK_SECRET = "dev-secret"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    env: str = field(default_factory=lambda: _env("RAS_ENV", "local"))  # local | staging | production
    data_dir: Path = field(default_factory=lambda: Path(_env("RAS_DATA_DIR", "data/service")))
    db_url: str | None = field(default_factory=lambda: _env("RAS_DB_URL"))
    transcriber: str = field(default_factory=lambda: _env("RAS_TRANSCRIBER", "transkun"))
    transcription_cache: bool = True
    api_key: str | None = field(default_factory=lambda: _env("RAS_API_KEY"))
    webhook_secret: str = field(default_factory=lambda: _env("RAS_WEBHOOK_SECRET", DEV_WEBHOOK_SECRET))
    # webhooks to private, loopback and link-local addresses (e.g. the cloud metadata service) are refused
    # unless this is on; it is on only for local development
    allow_private_webhooks: bool = field(default_factory=lambda: _flag("RAS_ALLOW_PRIVATE_WEBHOOKS",
                                                                       _env("RAS_ENV", "local") == "local"))
    max_attempts: int = 3
    visibility_timeout_s: int = 900  # a job 'processing' longer than this is considered stuck
    max_upload_mb: int = 50
    # cloud backends
    storage: str = field(default_factory=lambda: _env("RAS_STORAGE", "local"))  # local | s3
    s3_bucket: str | None = field(default_factory=lambda: _env("RAS_S3_BUCKET"))
    s3_prefix: str = field(default_factory=lambda: _env("RAS_S3_PREFIX", ""))
    queue: str = field(default_factory=lambda: _env("RAS_QUEUE", "db"))  # db | sqs
    sqs_queue_url: str | None = field(default_factory=lambda: _env("RAS_SQS_QUEUE_URL"))
    sqs_queue_name: str = field(default_factory=lambda: _env("RAS_SQS_QUEUE_NAME", "ras-jobs"))
    create_resources: bool = field(default_factory=lambda: _flag("RAS_CREATE_RESOURCES", False))  # stand-ins only
    log_format: str = field(default_factory=lambda: _env("RAS_LOG_FORMAT", "text"))  # text | json

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        host = _env("RAS_DB_HOST")
        if host:  # on ECS the password arrives on its own from Secrets Manager
            user, pw = quote_plus(_env("RAS_DB_USER", "ras")), quote_plus(_env("RAS_DB_PASSWORD", ""))
            name, port = _env("RAS_DB_NAME", "ras"), _env("RAS_DB_PORT", "5432")
            ssl = _env("RAS_DB_SSLMODE", "require")
            return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{name}?sslmode={ssl}"
        return f"sqlite:///{self.data_dir.resolve() / 'ras.db'}"

    def check(self) -> None:
        """Refuse to start a non-local deployment with development defaults."""
        if self.env == "local":
            return
        problems = []
        if self.webhook_secret == DEV_WEBHOOK_SECRET:
            problems.append("RAS_WEBHOOK_SECRET is the development default")
        if self.database_url.startswith("sqlite"):
            problems.append("SQLite is for local use; set RAS_DB_URL or RAS_DB_HOST")
        if self.storage != "s3":
            problems.append("RAS_STORAGE must be s3 outside local development")
        if problems:
            raise RuntimeError(f"unsafe settings for RAS_ENV={self.env}: " + "; ".join(problems))
