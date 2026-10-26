"""Database models (SQLAlchemy 2.0). SQLite locally; the same models run on PostgreSQL."""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class FileRow(Base):
    __tablename__ = "files"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16))  # audio | reference
    size: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(255))  # object-store key
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class Song(Base):
    __tablename__ = "songs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class Reference(Base):
    """Immutable: a change is a new version."""

    __tablename__ = "references"
    __table_args__ = (UniqueConstraint("song_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    song_id: Mapped[str] = mapped_column(ForeignKey("songs.id"))
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
    __table_args__ = (UniqueConstraint("rubric_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rubric_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    song_id: Mapped[str] = mapped_column(ForeignKey("songs.id"), index=True)
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
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    locked_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=now, index=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)


class Review(Base):
    """A teacher's judgement of an assessment: ground truth for evaluation and calibration."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    corrections: Mapped[list] = mapped_column(JSON, default=list)
    teacher_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=now)


def make_session_factory(url: str):
    kwargs = {}
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False, "timeout": 30}}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
