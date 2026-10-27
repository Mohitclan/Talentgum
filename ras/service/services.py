"""Business logic shared by the API and the CLI. No HTTP here.

Every function takes the caller's tenant and touches only that tenant's rows and storage prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ras.core.errors.analyze import Analysis
from ras.core.rubric import Rubric, score_analysis
from ras.core.score.reference import load_reference

from .db import (
    ApiKey,
    Assessment,
    AuditLog,
    FileRow,
    Reference,
    Review,
    RubricRow,
    Song,
    TeacherProfile,
    Tenant,
    now,
)
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


def audit(db: Session, tenant: str, actor: str, action: str, resource: str, **detail) -> None:
    db.add(AuditLog(tenant_id=tenant, actor=actor, action=action, resource=resource, detail=detail))
    db.commit()


# ---- tenants and API keys ---------------------------------------------------------------------


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(db: Session, tenant_id: str, name: str) -> Tenant:
    if not SLUG.match(tenant_id):
        raise ServiceError("invalid_tenant_id", "use letters, digits, '_', '.', '-' (max 64)")
    if db.get(Tenant, tenant_id):
        raise ServiceError("tenant_exists", f"tenant {tenant_id} already exists", 409)
    t = Tenant(id=tenant_id, name=name)
    db.add(t)
    db.commit()
    return t


def create_api_key(db: Session, tenant_id: str, label: str = "") -> str:
    """Returns the new key. Only its hash is stored, so it cannot be shown again."""
    if db.get(Tenant, tenant_id) is None:
        raise ServiceError("tenant_not_found", f"no tenant {tenant_id}", 404)
    key = "ras_" + secrets.token_urlsafe(32)
    db.add(ApiKey(tenant_id=tenant_id, key_hash=_hash_key(key), prefix=key[:12], label=label))
    db.commit()
    return key


def revoke_api_key(db: Session, prefix: str) -> int:
    rows = db.execute(select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))).scalars().all()
    for r in rows:
        r.revoked_at = now()
    db.commit()
    return len(rows)


def any_api_keys(db: Session) -> bool:
    return db.execute(select(ApiKey.id).where(ApiKey.revoked_at.is_(None)).limit(1)).scalar() is not None


def resolve_api_key(db: Session, key: str) -> tuple[str, str] | None:
    """(tenant_id, key prefix) for an active key, else None."""
    row = db.execute(select(ApiKey).where(ApiKey.key_hash == _hash_key(key), ApiKey.revoked_at.is_(None))).scalar()
    return (row.tenant_id, row.prefix) if row else None


# ---- files ------------------------------------------------------------------------------------


def save_file(db: Session, store: ObjectStore, tenant: str, filename: str, data: bytes, kind: str,
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
    # content-addressed inside the tenant's own prefix: identical uploads share bytes, tenants never do
    key = f"tenants/{tenant}/{kind}/{digest[:2]}/{digest}{ext}"
    if not store.exists(key):
        store.put(key, data)
    row = FileRow(id=_new_id("file"), tenant_id=tenant, sha256=digest, filename=Path(filename).name, kind=kind,
                  size=len(data), key=key)
    db.add(row)
    db.commit()
    return row


def get_file(db: Session, tenant: str, file_id: str, kind: str | None = None) -> FileRow:
    row = db.get(FileRow, file_id)
    if row is None or row.tenant_id != tenant or (kind and row.kind != kind):
        raise ServiceError("file_not_found", f"no {kind + ' ' if kind else ''}file {file_id}", 404)
    if row.deleted_at is not None:
        raise ServiceError("file_deleted", f"file {file_id} was deleted (retention or erasure)", 410)
    return row


def _delete_bytes_if_unused(db: Session, store: ObjectStore, row: FileRow) -> None:
    """Remove the stored bytes once no live file row of this tenant points at them."""
    others = db.execute(select(FileRow.id).where(FileRow.key == row.key, FileRow.id != row.id,
                                                 FileRow.deleted_at.is_(None))).first()
    if others is None:
        store.delete(row.key)


# ---- songs and references --------------------------------------------------------------------


def get_song(db: Session, tenant: str, song_id: str) -> Song:
    song = db.execute(select(Song).where(Song.tenant_id == tenant, Song.song_id == song_id)).scalar()
    if song is None:
        raise ServiceError("song_not_found", f"no song {song_id}", 404)
    return song


def list_songs(db: Session, tenant: str) -> list[Song]:
    return list(db.execute(select(Song).where(Song.tenant_id == tenant).order_by(Song.song_id)).scalars())


def references_of(db: Session, tenant: str, song_id: str) -> list[Reference]:
    return list(db.execute(select(Reference).where(Reference.tenant_id == tenant, Reference.song_id == song_id)
                           .order_by(Reference.version)).scalars())


def _parse_reference(store: ObjectStore, f: FileRow, bpm: float | None):
    try:
        return load_reference(store.get_path(f.key), bpm=bpm)
    except Exception as exc:
        raise ServiceError("reference_unreadable", f"could not read the reference: {exc}", 422) from exc


def add_reference(db: Session, store: ObjectStore, tenant: str, song_id: str, file_id: str,
                  bpm: float | None = None) -> Reference:
    get_song(db, tenant, song_id)
    f = get_file(db, tenant, file_id, "reference")
    ref = _parse_reference(store, f, bpm)
    if ref.n_notes == 0:
        raise ServiceError("reference_empty", "the reference has no notes", 422)
    version = (db.execute(select(func.max(Reference.version)).where(
        Reference.tenant_id == tenant, Reference.song_id == song_id)).scalar() or 0) + 1
    row = Reference(tenant_id=tenant, song_id=song_id, version=version, file_id=file_id, sha256=ref.sha256,
                    n_notes=ref.n_notes, n_measures=ref.n_measures, bpm=ref.bpm, warnings=ref.warnings)
    db.add(row)
    db.commit()
    return row


def create_song(db: Session, store: ObjectStore, tenant: str, song_id: str, title: str, reference_file_id: str,
                bpm: float | None = None) -> tuple[Song, Reference]:
    if not SLUG.match(song_id):
        raise ServiceError("invalid_song_id", "use letters, digits, '_', '.', '-' (max 64)", 422)
    if db.execute(select(Song.pk).where(Song.tenant_id == tenant, Song.song_id == song_id)).first():
        raise ServiceError("song_exists", f"song {song_id} already exists; add a new reference version", 409)
    song = Song(tenant_id=tenant, song_id=song_id, title=title)
    db.add(song)
    db.commit()
    try:
        ref = add_reference(db, store, tenant, song_id, reference_file_id, bpm)
    except ServiceError:
        db.delete(song)
        db.commit()
        raise
    return song, ref


def create_teacher_profile(db: Session, tenant: str, song_id: str, assessment_ids: list[str],
                           actor: str = "local") -> TeacherProfile:
    """Pool finished assessments of teacher recordings into the song's benchmark profile."""
    from ras.core.profile import build_profile

    get_song(db, tenant, song_id)
    if not assessment_ids:
        raise ServiceError("no_recordings", "give the assessment ids of 2-3 teacher recordings", 422)
    asms = [get_assessment(db, tenant, i) for i in dict.fromkeys(assessment_ids)]
    if any(a.song_id != song_id for a in asms):
        raise ServiceError("wrong_song", "every assessment must be of this song", 422)
    versions = {a.reference_version for a in asms}
    if len(versions) != 1:
        raise ServiceError("mixed_reference_versions", "the recordings were assessed against different "
                           "reference versions; re-assess them against one version", 422)
    if any(not a.result or a.status not in ("completed", "review_required", "reviewed") for a in asms):
        raise ServiceError("not_finished", "every assessment must have finished successfully", 409)
    version = versions.pop()
    ref = latest_reference(db, tenant, song_id, version)
    profile = build_profile([a.result.get("scoring_inputs", {}) for a in asms], ref.sha256)
    row = TeacherProfile(tenant_id=tenant, song_id=song_id, reference_version=version,
                         assessment_ids=[a.id for a in asms], profile=profile)
    db.add(row)
    db.commit()
    audit(db, tenant, actor, "teacher_profile.create", song_id, reference_version=version, recordings=len(asms))
    return row


def latest_teacher_profile(db: Session, tenant: str, song_id: str, reference_version: int) -> TeacherProfile | None:
    return db.execute(select(TeacherProfile).where(
        TeacherProfile.tenant_id == tenant, TeacherProfile.song_id == song_id,
        TeacherProfile.reference_version == reference_version).order_by(TeacherProfile.id.desc()).limit(1)).scalar()


def latest_reference(db: Session, tenant: str, song_id: str, version: int | None = None) -> Reference:
    q = select(Reference).where(Reference.tenant_id == tenant, Reference.song_id == song_id)
    q = q.where(Reference.version == version) if version else q.order_by(Reference.version.desc())
    row = db.execute(q.limit(1)).scalar()
    if row is None:
        raise ServiceError("reference_not_found", f"no reference for song {song_id}", 404)
    return row


# ---- rubrics ---------------------------------------------------------------------------------


def validate_rubric(body: dict) -> Rubric:
    try:
        return Rubric.model_validate(body)
    except Exception as exc:
        raise ServiceError("invalid_rubric", str(exc), 422) from exc


def create_rubric(db: Session, tenant: str, body: dict) -> RubricRow:
    rubric = validate_rubric(body)
    last = db.execute(select(func.max(RubricRow.version)).where(
        RubricRow.tenant_id == tenant, RubricRow.rubric_id == rubric.rubric_id)).scalar() or 0
    stored = rubric.model_copy(update={"version": last + 1})  # versions are assigned, never edited
    row = RubricRow(tenant_id=tenant, rubric_id=rubric.rubric_id, version=last + 1,
                    body=json.loads(stored.model_dump_json(by_alias=True)))
    db.add(row)
    db.commit()
    return row


def get_rubric(db: Session, tenant: str, rubric_id: str, version: int | None = None) -> RubricRow:
    q = select(RubricRow).where(RubricRow.tenant_id == tenant, RubricRow.rubric_id == rubric_id)
    q = q.where(RubricRow.version == version) if version else q.order_by(RubricRow.version.desc())
    row = db.execute(q.limit(1)).scalar()
    if row is None:
        raise ServiceError("rubric_not_found", f"no rubric {rubric_id}" + (f" v{version}" if version else ""), 404)
    return row


def list_rubrics(db: Session, tenant: str) -> list[RubricRow]:
    return list(db.execute(select(RubricRow).where(RubricRow.tenant_id == tenant)
                           .order_by(RubricRow.rubric_id, RubricRow.version)).scalars())


def try_rubric(db: Session, tenant: str, body: dict, song_id: str | None = None, limit: int = 200) -> dict:
    """Re-score past assessments with a candidate rubric before activating it. Nothing is changed."""
    candidate = validate_rubric(body)
    q = (select(Assessment).where(Assessment.tenant_id == tenant,
                                  Assessment.status.in_(("completed", "review_required", "reviewed")))
         .order_by(Assessment.created_at.desc()).limit(limit))
    if song_id:
        q = q.where(Assessment.song_id == song_id)
    rows, deltas, band_changes = [], [], 0
    for a in db.execute(q).scalars():
        inputs = (a.result or {}).get("scoring_inputs")
        if not inputs:
            continue  # results stored before scoring inputs were kept cannot be re-scored
        new = score_analysis(candidate, Analysis.from_scoring_inputs(inputs))
        old_band = ((a.result or {}).get("scoring") or {}).get("band")
        delta = None if a.total_score is None or new.total is None else round(new.total - a.total_score, 1)
        if delta is not None:
            deltas.append(delta)
        band_changes += int(old_band != new.band)
        rows.append({"assessment_id": a.id, "student_id": a.student_id, "song_id": a.song_id,
                     "current_score": a.total_score, "candidate_score": new.total, "delta": delta,
                     "current_band": old_band, "candidate_band": new.band})
    return {"rubric_id": candidate.rubric_id, "assessments": len(rows),
            "mean_delta": round(sum(deltas) / len(deltas), 2) if deltas else None,
            "max_abs_delta": max((abs(d) for d in deltas), default=None),
            "band_changes": band_changes, "rows": rows}


# ---- assessments -----------------------------------------------------------------------------


def create_assessment(db: Session, tenant: str, req: dict,
                      idempotency_key: str | None = None) -> tuple[Assessment, bool]:
    """Returns (assessment, created). The same Idempotency-Key and body returns the original job."""
    req_hash = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
    if idempotency_key:
        prior = db.execute(select(Assessment).where(Assessment.tenant_id == tenant,
                                                    Assessment.idempotency_key == idempotency_key)).scalar()
        if prior is not None:
            if prior.request_hash != req_hash:
                raise ServiceError("idempotency_key_reused",
                                   "this Idempotency-Key was used with a different request", 409)
            return prior, False
    cb = req.get("callback_url")
    if cb and not cb.lower().startswith(("https://", "http://")):
        raise ServiceError("invalid_callback_url", "callback_url must be an http(s) URL", 422)
    ref = latest_reference(db, tenant, req["song_id"], req.get("reference_version"))
    get_file(db, tenant, req["recording_file_id"], "audio")
    rub = get_rubric(db, tenant, req["rubric_id"], req.get("rubric_version"))
    a = Assessment(
        id=_new_id("asm"), tenant_id=tenant, song_id=req["song_id"], reference_version=ref.version,
        rubric_id=rub.rubric_id, rubric_version=rub.version, recording_file_id=req["recording_file_id"],
        student_id=req.get("student_id"), external_ref=req.get("external_ref"), start_s=req.get("start_s"),
        callback_url=req.get("callback_url"), idempotency_key=idempotency_key, request_hash=req_hash)
    db.add(a)
    db.commit()
    return a, True


def get_assessment(db: Session, tenant: str, asm_id: str) -> Assessment:
    a = db.get(Assessment, asm_id)
    if a is None or a.tenant_id != tenant:
        raise ServiceError("assessment_not_found", f"no assessment {asm_id}", 404)
    return a


def list_assessments(db: Session, tenant: str, status: str | None = None, student_id: str | None = None,
                     limit: int = 50) -> list[Assessment]:
    q = select(Assessment).where(Assessment.tenant_id == tenant).order_by(Assessment.created_at.desc()).limit(limit)
    if status:
        q = q.where(Assessment.status == status)
    if student_id:
        q = q.where(Assessment.student_id == student_id)
    return list(db.execute(q).scalars())


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


def record_review(db: Session, tenant: str, asm_id: str, reviewer: str, corrections: list[dict],
                  teacher_scores: dict, overall_score: float | None, comment: str | None,
                  actor: str = "local") -> Review:
    a = get_assessment(db, tenant, asm_id)
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
    rv = Review(tenant_id=tenant, assessment_id=asm_id, reviewer=reviewer, corrections=corrections,
                teacher_scores=teacher_scores, overall_score=overall_score, comment=comment)
    db.add(rv)
    a.status, a.needs_review = "reviewed", False
    db.commit()
    audit(db, tenant, actor, "review.create", asm_id, reviewer=reviewer, corrections=len(corrections))
    return rv


def list_reviews(db: Session, tenant: str, asm_id: str) -> list[Review]:
    get_assessment(db, tenant, asm_id)
    return list(db.execute(select(Review).where(Review.tenant_id == tenant, Review.assessment_id == asm_id)
                           .order_by(Review.id)).scalars())


def review_queue(db: Session, tenant: str, limit: int = 50) -> list[Assessment]:
    return list(db.execute(select(Assessment).where(Assessment.tenant_id == tenant,
                                                    Assessment.status == "review_required")
                           .order_by(Assessment.created_at).limit(limit)).scalars())


# ---- privacy: erasure and retention ------------------------------------------------------------


def delete_student(db: Session, store: ObjectStore, tenant: str, student_id: str, actor: str = "local") -> dict:
    """Right to erasure: remove a student's assessments, reviews and recordings for this tenant."""
    asms = db.execute(select(Assessment).where(Assessment.tenant_id == tenant,
                                               Assessment.student_id == student_id)).scalars().all()
    if not asms:
        raise ServiceError("student_not_found", f"no data for student {student_id}", 404)
    n_reviews = n_files = 0
    for a in asms:
        for rv in db.execute(select(Review).where(Review.assessment_id == a.id)).scalars():
            db.delete(rv)
            n_reviews += 1
        f = db.get(FileRow, a.recording_file_id)
        db.delete(a)
        db.flush()
        if f is not None and f.deleted_at is None:
            f.deleted_at = now()
            db.flush()
            _delete_bytes_if_unused(db, store, f)
            n_files += 1
    db.commit()
    audit(db, tenant, actor, "student.delete", student_id, assessments=len(asms), reviews=n_reviews,
          recordings=n_files)
    return {"student_id": student_id, "assessments_deleted": len(asms), "reviews_deleted": n_reviews,
            "recordings_deleted": n_files}


def purge_audio(db: Session, store: ObjectStore, older_than_days: int, tenant: str | None = None) -> int:
    """Retention: delete raw recordings of finished assessments older than N days. Results are kept."""
    import datetime as dt

    cutoff = now() - dt.timedelta(days=older_than_days)
    q = select(Assessment).where(Assessment.finished_at.is_not(None), Assessment.finished_at < cutoff)
    if tenant:
        q = q.where(Assessment.tenant_id == tenant)
    n = 0
    for a in db.execute(q).scalars():
        f = db.get(FileRow, a.recording_file_id)
        if f is not None and f.deleted_at is None:
            f.deleted_at = now()
            db.flush()
            _delete_bytes_if_unused(db, store, f)
            n += 1
            db.add(AuditLog(tenant_id=a.tenant_id, actor="system", action="recording.purge", resource=f.id,
                            detail={"older_than_days": older_than_days}))
    db.commit()
    return n


def audit_entries(db: Session, tenant: str, limit: int = 100) -> list[AuditLog]:
    return list(db.execute(select(AuditLog).where(AuditLog.tenant_id == tenant)
                           .order_by(AuditLog.id.desc()).limit(limit)).scalars())
