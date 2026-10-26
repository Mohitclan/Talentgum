"""Business logic shared by the API and the CLI. No HTTP here."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ras.core.rubric import Rubric
from ras.core.score.reference import load_reference

from .db import Assessment, FileRow, Reference, Review, RubricRow, Song, now
from .storage import ObjectStore, sha256_bytes

AUDIO_EXT = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".ogg", ".webm", ".opus"}
REFERENCE_EXT = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}
SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ServiceError(Exception):
    def __init__(self, code: str, detail: str, status: int = 422):
        super().__init__(detail)
        self.code, self.detail, self.status = code, detail, status


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# ---- files ------------------------------------------------------------------------------------


def save_file(db: Session, store: ObjectStore, filename: str, data: bytes, kind: str,
              max_mb: int = 50) -> FileRow:
    ext = Path(filename).suffix.lower()
    allowed = AUDIO_EXT if kind == "audio" else REFERENCE_EXT
    if ext not in allowed:
        raise ServiceError("unsupported_file_type", f"{ext or 'no extension'} is not accepted for {kind}; "
                           f"use one of {sorted(allowed)}", 415)
    if not data:
        raise ServiceError("empty_file", "the file is empty", 422)
    if len(data) > max_mb * 1024 * 1024:
        raise ServiceError("file_too_large", f"limit is {max_mb} MB", 413)
    digest = sha256_bytes(data)
    fid = _new_id("file")
    key = f"{kind}/{digest[:2]}/{digest}{ext}"  # content-addressed: identical uploads share bytes
    if not store.exists(key):
        store.put(key, data)
    row = FileRow(id=fid, sha256=digest, filename=Path(filename).name, kind=kind, size=len(data), key=key)
    db.add(row)
    db.commit()
    return row


def get_file(db: Session, file_id: str, kind: str | None = None) -> FileRow:
    row = db.get(FileRow, file_id)
    if row is None or (kind and row.kind != kind):
        raise ServiceError("file_not_found", f"no {kind or ''} file {file_id}".replace("  ", " "), 404)
    return row


# ---- songs and references --------------------------------------------------------------------


def _parse_reference(store: ObjectStore, f: FileRow, bpm: float | None):
    try:
        return load_reference(store.get_path(f.key), bpm=bpm)
    except Exception as exc:
        raise ServiceError("reference_unreadable", f"could not read the reference: {exc}", 422) from exc


def add_reference(db: Session, store: ObjectStore, song_id: str, file_id: str, bpm: float | None = None) -> Reference:
    song = db.get(Song, song_id)
    if song is None:
        raise ServiceError("song_not_found", f"no song {song_id}", 404)
    f = get_file(db, file_id, "reference")
    ref = _parse_reference(store, f, bpm)
    if ref.n_notes == 0:
        raise ServiceError("reference_empty", "the reference has no notes", 422)
    version = (db.execute(select(func.max(Reference.version)).where(Reference.song_id == song_id)).scalar() or 0) + 1
    row = Reference(song_id=song_id, version=version, file_id=file_id, sha256=ref.sha256, n_notes=ref.n_notes,
                    n_measures=ref.n_measures, bpm=ref.bpm, warnings=ref.warnings)
    db.add(row)
    db.commit()
    return row


def create_song(db: Session, store: ObjectStore, song_id: str, title: str, reference_file_id: str,
                bpm: float | None = None) -> tuple[Song, Reference]:
    if not SLUG.match(song_id):
        raise ServiceError("invalid_song_id", "use letters, digits, '_', '.', '-' (max 64)", 422)
    if db.get(Song, song_id):
        raise ServiceError("song_exists", f"song {song_id} already exists; add a new reference version", 409)
    db.add(Song(id=song_id, title=title))
    db.commit()
    try:
        ref = add_reference(db, store, song_id, reference_file_id, bpm)
    except ServiceError:
        db.delete(db.get(Song, song_id))
        db.commit()
        raise
    return db.get(Song, song_id), ref


def latest_reference(db: Session, song_id: str, version: int | None = None) -> Reference:
    q = select(Reference).where(Reference.song_id == song_id)
    q = q.where(Reference.version == version) if version else q.order_by(Reference.version.desc())
    row = db.execute(q.limit(1)).scalar()
    if row is None:
        raise ServiceError("reference_not_found", f"no reference for song {song_id}", 404)
    return row


# ---- rubrics ---------------------------------------------------------------------------------


def create_rubric(db: Session, body: dict) -> RubricRow:
    try:
        rubric = Rubric.model_validate(body)
    except Exception as exc:
        raise ServiceError("invalid_rubric", str(exc), 422) from exc
    last = db.execute(select(func.max(RubricRow.version)).where(RubricRow.rubric_id == rubric.rubric_id)).scalar() or 0
    stored = rubric.model_copy(update={"version": last + 1})  # versions are assigned, never edited
    row = RubricRow(rubric_id=rubric.rubric_id, version=last + 1,
                    body=json.loads(stored.model_dump_json(by_alias=True)))
    db.add(row)
    db.commit()
    return row


def get_rubric(db: Session, rubric_id: str, version: int | None = None) -> RubricRow:
    q = select(RubricRow).where(RubricRow.rubric_id == rubric_id)
    q = q.where(RubricRow.version == version) if version else q.order_by(RubricRow.version.desc())
    row = db.execute(q.limit(1)).scalar()
    if row is None:
        raise ServiceError("rubric_not_found", f"no rubric {rubric_id}" + (f" v{version}" if version else ""), 404)
    return row


# ---- assessments -----------------------------------------------------------------------------


def create_assessment(db: Session, req: dict, idempotency_key: str | None = None) -> tuple[Assessment, bool]:
    """Returns (assessment, created). The same Idempotency-Key and body returns the original job."""
    req_hash = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
    if idempotency_key:
        prior = db.execute(select(Assessment).where(Assessment.idempotency_key == idempotency_key)).scalar()
        if prior is not None:
            if prior.request_hash != req_hash:
                raise ServiceError("idempotency_key_reused",
                                   "this Idempotency-Key was used with a different request", 409)
            return prior, False
    ref = latest_reference(db, req["song_id"], req.get("reference_version"))
    get_file(db, req["recording_file_id"], "audio")
    rub = get_rubric(db, req["rubric_id"], req.get("rubric_version"))
    a = Assessment(
        id=_new_id("asm"), song_id=req["song_id"], reference_version=ref.version, rubric_id=rub.rubric_id,
        rubric_version=rub.version, recording_file_id=req["recording_file_id"], student_id=req.get("student_id"),
        external_ref=req.get("external_ref"), start_s=req.get("start_s"), callback_url=req.get("callback_url"),
        idempotency_key=idempotency_key, request_hash=req_hash)
    db.add(a)
    db.commit()
    return a, True


def get_assessment(db: Session, asm_id: str) -> Assessment:
    a = db.get(Assessment, asm_id)
    if a is None:
        raise ServiceError("assessment_not_found", f"no assessment {asm_id}", 404)
    return a


def assessment_view(a: Assessment, include_notes: bool = False) -> dict:
    res = a.result or {}
    out = {
        "assessment_id": a.id, "status": a.status, "song_id": a.song_id, "reference_version": a.reference_version,
        "rubric": {"id": a.rubric_id, "version": a.rubric_version}, "student_id": a.student_id,
        "external_ref": a.external_ref, "attempts": a.attempts, "error_code": a.error_code,
        "error_detail": a.error_detail, "needs_review": a.needs_review,
        "total_score": a.total_score, "confidence": a.confidence,
        "created_at": a.created_at.isoformat(), "finished_at": a.finished_at.isoformat() if a.finished_at else None,
        "recording_file_id": a.recording_file_id,
    }
    if res:
        out.update(summary=res.get("summary"), scoring=res.get("scoring"), confidence_detail=res.get("confidence"),
                   feedback=res.get("feedback"), warnings=res.get("warnings"), versions=res.get("versions"),
                   quality=res.get("quality"), transposition=res.get("transposition"),
                   start_timestamp=res.get("start_timestamp"), timings_s=res.get("timings_s"))
        if include_notes:
            out["notes"] = res.get("notes")
    return out


# ---- review ----------------------------------------------------------------------------------

NOTE_STATUSES = {"correct", "wrong_pitch", "wrong_octave", "missing", "uncertain"}


def record_review(db: Session, asm_id: str, reviewer: str, corrections: list[dict], teacher_scores: dict,
                  overall_score: float | None, comment: str | None) -> Review:
    a = get_assessment(db, asm_id)
    if a.status not in ("completed", "review_required", "reviewed"):
        raise ServiceError("not_reviewable",
                           f"assessment is {a.status}; only finished assessments can be reviewed", 409)
    notes = (a.result or {}).get("notes") or []
    for c in corrections:
        if c.get("kind") == "note":
            if c.get("status") not in NOTE_STATUSES or not any(n["ref_index"] == c.get("ref_index") for n in notes):
                raise ServiceError("invalid_correction", f"bad note correction: {c}", 422)
        elif c.get("kind") == "extra":
            if not any(n["status"] == "extra" and n["perf_index"] == c.get("perf_index") for n in notes):
                raise ServiceError("invalid_correction", f"bad extra-note correction: {c}", 422)
        else:
            raise ServiceError("invalid_correction", f"unknown correction kind: {c}", 422)
    rv = Review(assessment_id=asm_id, reviewer=reviewer, corrections=corrections, teacher_scores=teacher_scores,
                overall_score=overall_score, comment=comment)
    db.add(rv)
    a.status, a.needs_review = "reviewed", False
    db.commit()
    return rv


def review_queue(db: Session, limit: int = 50) -> list[Assessment]:
    return list(db.execute(select(Assessment).where(Assessment.status == "review_required")
                           .order_by(Assessment.created_at).limit(limit)).scalars())


def touch_finished(a: Assessment) -> None:
    a.finished_at = now()
