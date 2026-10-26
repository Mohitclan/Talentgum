from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("RAS_DATA_DIR", "data/service")))
    db_url: str | None = field(default_factory=lambda: os.environ.get("RAS_DB_URL"))
    transcriber: str = field(default_factory=lambda: os.environ.get("RAS_TRANSCRIBER", "transkun"))
    transcription_cache: bool = True
    api_key: str | None = field(default_factory=lambda: os.environ.get("RAS_API_KEY"))
    webhook_secret: str = field(default_factory=lambda: os.environ.get("RAS_WEBHOOK_SECRET", "dev-secret"))
    max_attempts: int = 3
    visibility_timeout_s: int = 900  # a job 'processing' longer than this is considered stuck
    max_upload_mb: int = 50

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{self.data_dir.resolve() / 'ras.db'}"
