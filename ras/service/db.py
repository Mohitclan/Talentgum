"""Database models (SQLAlchemy 2.0). SQLite locally; the same models run on PostgreSQL.

Every client organisation is a tenant. Every row that holds client data carries `tenant_id`, and the
service layer filters every query by it, so one tenant can never read another's songs, recordings,
results or reviews.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DEFAULT_TENANT = "local"
SCHEMA_VERSION = 2


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class ApiKey(Base):
    """Only a SHA-256 hash of the key is stored; the key itself is shown once, at creation."""

    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    prefix: Mapped[str] = mapped_column(String(12))  # first characters, to recognise a key in logs
    label: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[dt.datetime] = mapped_column(default=now)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)


class FileRow(Base):
    __tablename__ = "files"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16))  # audio | reference
    size: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(255))  # object-store key, always under the tenant's prefix
    created_at: Mapped[dt.datetime] = mapped_column(default=now)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)  # bytes removed (retention/erasure)


class Song(Base):
    __tablename__ = "songs"
    __table_args__ = (UniqueConstraint("tenant_id", "song_id"),)
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    song_id: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class Reference(Base):
    """Immutable: a change is a new version."""

    __tablename__ = "references"
    __table_args__ = (UniqueConstraint("tenant_id", "song_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    song_id: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id"))
    sha256: Mapped[str] = mapped_column(String(64))
    n_notes: Mapped[int] = mapped_column(Integer)
    n_measures: Mapped[int] = mapped_column(Integer)
    bpm: Mapped[float] = mapped_column(Float)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class RubricRow(Base):
    """Immutable: a change is a new version."""

    __tablename__ = "rubrics"
    __table_args__ = (UniqueConstraint("tenant_id", "rubric_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    rubric_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class TeacherProfile(Base):
    """How teachers play one reference version (ras.core.profile); the newest one is used for students."""

    __tablename__ = "teacher_profiles"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    song_id: Mapped[str] = mapped_column(String(64), index=True)
    reference_version: Mapped[int] = mapped_column(Integer)
    assessment_ids: Mapped[list] = mapped_column(JSON, default=list)
    profile: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class Assessment(Base):
    __tablename__ = "assessments"
    __table_args__ = (UniqueConstraint("tenant_id", "idempotency_key"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    song_id: Mapped[str] = mapped_column(String(64), index=True)
    reference_version: Mapped[int] = mapped_column(Integer)
    rubric_id: Mapped[str] = mapped_column(String(64))
    rubric_version: Mapped[int] = mapped_column(Integer)
    recording_file_id: Mapped[str] = mapped_column(ForeignKey("files.id"))
    student_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    external_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    start_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    callback_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # queued | processing | completed | review_required | reviewed | rejected | failed
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    locked_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=now, index=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)


class Review(Base):
    """A teacher's judgement of an assessment: ground truth for evaluation and calibration."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    corrections: Mapped[list] = mapped_column(JSON, default=list)
    teacher_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class AuditLog(Base):
    """Who touched recordings and personal data, and when."""

    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(64))  # API key prefix, "local" or "system"
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(128))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    at: Mapped[dt.datetime] = mapped_column(default=now, index=True)


class SchemaError(RuntimeError):
    pass


def make_session_factory(url: str):
    kwargs = {}
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False, "timeout": 30}}
    else:
        kwargs = {"pool_pre_ping": True}  # survive database failovers and idle-connection reaping
    engine = create_engine(url, **kwargs)
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":  # API and workers may start together: one creates the schema
            conn.execute(text("SELECT pg_advisory_xact_lock(7242001)"))
        insp = inspect(conn)
        if insp.has_table("assessments") and "tenant_id" not in {c["name"] for c in insp.get_columns("assessments")}:
            raise SchemaError(f"the database at {engine.url!r} uses an older schema (before tenants). It holds only "
                              "local development data: move it aside and start again, or migrate it.")
        Base.metadata.create_all(conn)
        if conn.execute(text("SELECT 1 FROM tenants WHERE id = :t"), {"t": DEFAULT_TENANT}).first() is None:
            conn.execute(Tenant.__table__.insert().values(id=DEFAULT_TENANT, name="Local development",
                                                          created_at=now()))
    return sessionmaker(engine, expire_on_commit=False)


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
