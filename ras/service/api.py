"""REST API under /v1. The web UI uses only this API.

Authentication: each tenant has its own API keys (`X-API-Key`). While no key exists and no shared key
is configured, the API runs open as the default "local" tenant, which keeps local development simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.background import BackgroundTask

from . import services as svc
from .config import Settings
from .db import DEFAULT_TENANT, Song, make_session_factory
from .queue import make_signal
from .storage import make_store

UI_DIR = Path(__file__).parent / "ui"


class SongIn(BaseModel):
    song_id: str
    title: str
    reference_file_id: str
    bpm: float | None = None


class ReferenceIn(BaseModel):
    reference_file_id: str
    bpm: float | None = None


class TeacherProfileIn(BaseModel):
    assessment_ids: list[str]


def profile_view(p) -> dict:
    return {"id": p.id, "song_id": p.song_id, "reference_version": p.reference_version,
            "recordings": p.profile.get("recordings"), "assessment_ids": p.assessment_ids,
            "tempo_bpm": p.profile.get("tempo_bpm"), "created_at": p.created_at.isoformat()}


class AssessmentIn(BaseModel):
    song_id: str
    recording_file_id: str
    rubric_id: str
    rubric_version: int | None = None
    reference_version: int | None = None
    student_id: str | None = None
    external_ref: str | None = None
    start_s: float | None = Field(None, ge=0)
    callback_url: str | None = None


class ReviewIn(BaseModel):
    reviewer: str
    corrections: list[dict] = Field(default_factory=list)
    teacher_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float | None = Field(None, ge=0, le=100)
    comment: str | None = None


class RubricTestIn(BaseModel):
    rubric: dict
    song_id: str | None = None
    limit: int = Field(200, ge=1, le=2000)


@dataclass
class Caller:
    tenant: str
    actor: str


def problem(status: int, code: str, detail: str) -> JSONResponse:
    """RFC 9457 problem details with a stable machine-readable code."""
    return JSONResponse({"type": f"https://ras.local/errors/{code}", "title": code.replace("_", " "),
                         "status": status, "code": code, "detail": detail},
                        status_code=status, media_type="application/problem+json")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.check()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Session = make_session_factory(settings.database_url)
    store = make_store(settings)
    signal = make_signal(settings)
    app = FastAPI(title="Recording Assessment System", version="0.4.0")
    app.state.settings, app.state.Session, app.state.store, app.state.signal = settings, Session, store, signal

    def db_session():
        with Session() as db:
            yield db

    def caller(x_api_key: str | None = Header(None), db=Depends(db_session)) -> Caller:
        if settings.api_key:  # one shared key for the default tenant (simple single-client setups)
            if x_api_key == settings.api_key:
                return Caller(DEFAULT_TENANT, "shared-key")
            if x_api_key is None or svc.resolve_api_key(db, x_api_key) is None:
                raise HTTPException(401, "missing or wrong X-API-Key")
        if x_api_key is not None:
            found = svc.resolve_api_key(db, x_api_key)
            if found is None:
                raise HTTPException(401, "unknown or revoked X-API-Key")
            return Caller(found[0], found[1])
        if svc.any_api_keys(db):
            raise HTTPException(401, "this server requires an X-API-Key")
        return Caller(DEFAULT_TENANT, "local")

    @app.exception_handler(svc.ServiceError)
    async def _service_error(_: Request, exc: svc.ServiceError):
        return problem(exc.status, exc.code, exc.detail)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return problem(exc.status_code, "unauthorized" if exc.status_code == 401 else "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):  # never leak internals
        return problem(500, "internal_error", "unexpected server error")

    # ---- health and UI
    @app.get("/v1/health")
    def health(db=Depends(db_session)):
        db.execute(select(Song.pk).limit(1))
        return {"status": "ok", "transcriber": settings.transcriber, "auth": "keys" if (
            settings.api_key or svc.any_api_keys(db)) else "open", "storage": settings.storage,
            "queue": settings.queue}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    @app.get("/ui/", include_in_schema=False)
    def ui():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/v1/me")
    def me(c: Caller = Depends(caller)):
        return {"tenant": c.tenant, "actor": c.actor}

    # ---- files
    @app.post("/v1/files", status_code=201)
    async def upload(file: UploadFile = File(...), kind: str = Query(..., pattern="^(audio|reference)$"),
                     c: Caller = Depends(caller), db=Depends(db_session)):
        data = await file.read()
        row = svc.save_file(db, store, c.tenant, file.filename or "upload", data, kind, settings.max_upload_mb)
        return {"file_id": row.id, "sha256": row.sha256, "size": row.size, "filename": row.filename, "kind": row.kind}

    @app.get("/v1/files/{file_id}/content")
    def file_content(file_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        row = svc.get_file(db, c.tenant, file_id)
        cleanup = None
        if row.kind == "audio":  # every access to a recording is audited
            svc.audit(db, c.tenant, c.actor, "recording.read", file_id)
            cleanup = BackgroundTask(store.release, row.key)  # no lingering copy of a recording on S3 setups
        return FileResponse(store.get_path(row.key), filename=row.filename, background=cleanup)

    # ---- songs
    def song_view(db, tenant: str, song):
        refs = svc.references_of(db, tenant, song.song_id)
        return {"song_id": song.song_id, "title": song.title, "references": [
            {"version": r.version, "sha256": r.sha256[:16], "notes": r.n_notes, "measures": r.n_measures,
             "bpm": r.bpm, "warnings": r.warnings} for r in refs]}

    @app.post("/v1/songs", status_code=201)
    def create_song(body: SongIn, c: Caller = Depends(caller), db=Depends(db_session)):
        song, _ = svc.create_song(db, store, c.tenant, body.song_id, body.title, body.reference_file_id, body.bpm)
        return song_view(db, c.tenant, song)

    @app.get("/v1/songs")
    def list_songs(c: Caller = Depends(caller), db=Depends(db_session)):
        return [song_view(db, c.tenant, s) for s in svc.list_songs(db, c.tenant)]

    @app.get("/v1/songs/{song_id}")
    def get_song(song_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        return song_view(db, c.tenant, svc.get_song(db, c.tenant, song_id))

    @app.post("/v1/songs/{song_id}/references", status_code=201)
    def add_reference(song_id: str, body: ReferenceIn, c: Caller = Depends(caller), db=Depends(db_session)):
        svc.add_reference(db, store, c.tenant, song_id, body.reference_file_id, body.bpm)
        return song_view(db, c.tenant, svc.get_song(db, c.tenant, song_id))

    @app.post("/v1/songs/{song_id}/teacher-profile", status_code=201)
    def create_teacher_profile(song_id: str, body: TeacherProfileIn, c: Caller = Depends(caller),
                               db=Depends(db_session)):
        """Benchmark from finished assessments of 2-3 teacher recordings; later student results compare with it."""
        return profile_view(svc.create_teacher_profile(db, c.tenant, song_id, body.assessment_ids, actor=c.actor))

    @app.get("/v1/songs/{song_id}/teacher-profile")
    def get_teacher_profile(song_id: str, reference_version: int | None = None, c: Caller = Depends(caller),
                            db=Depends(db_session)):
        version = reference_version or svc.latest_reference(db, c.tenant, song_id).version
        p = svc.latest_teacher_profile(db, c.tenant, song_id, version)
        if p is None:
            raise svc.ServiceError("profile_not_found", f"no teacher profile for {song_id} v{version}", 404)
        return profile_view(p)

    # ---- rubrics
    @app.post("/v1/rubrics", status_code=201)
    def create_rubric(body: dict, c: Caller = Depends(caller), db=Depends(db_session)):
        row = svc.create_rubric(db, c.tenant, body)
        return {"rubric_id": row.rubric_id, "version": row.version}

    @app.get("/v1/rubrics")
    def list_rubrics(c: Caller = Depends(caller), db=Depends(db_session)):
        return [{"rubric_id": r.rubric_id, "version": r.version, "criteria": [x["id"] for x in r.body["criteria"]]}
                for r in svc.list_rubrics(db, c.tenant)]

    @app.post("/v1/rubrics/test")
    def rubric_test(body: RubricTestIn, c: Caller = Depends(caller), db=Depends(db_session)):
        """Re-score recent assessments with a candidate rubric and show the change. Saves nothing."""
        return svc.try_rubric(db, c.tenant, body.rubric, body.song_id, body.limit)

    @app.get("/v1/rubrics/{rubric_id}")
    def get_rubric(rubric_id: str, version: int | None = None, c: Caller = Depends(caller), db=Depends(db_session)):
        return svc.get_rubric(db, c.tenant, rubric_id, version).body

    # ---- assessments
    @app.post("/v1/assessments", status_code=202)
    def create_assessment(body: AssessmentIn, idempotency_key: str | None = Header(None),
                          c: Caller = Depends(caller), db=Depends(db_session)):
        a, created = svc.create_assessment(db, c.tenant, body.model_dump(), idempotency_key)
        if created:
            signal.publish(a.id)
        return JSONResponse({"assessment_id": a.id, "status": a.status, "created": created},
                            status_code=202 if created else 200)

    @app.get("/v1/assessments")
    def list_assessments(status: str | None = None, student_id: str | None = None,
                         limit: int = Query(50, le=200), c: Caller = Depends(caller), db=Depends(db_session)):
        return [svc.assessment_view(a) for a in svc.list_assessments(db, c.tenant, status, student_id, limit)]

    @app.get("/v1/assessments/{asm_id}")
    def get_assessment(asm_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        return svc.assessment_view(svc.get_assessment(db, c.tenant, asm_id))

    @app.get("/v1/assessments/{asm_id}/notes")
    def get_notes(asm_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        a = svc.get_assessment(db, c.tenant, asm_id)
        if not a.result:
            raise svc.ServiceError("not_ready", f"assessment is {a.status}", 409)
        return {"assessment_id": a.id, "notes": a.result.get("notes", [])}

    @app.post("/v1/assessments/{asm_id}/review", status_code=201)
    def review(asm_id: str, body: ReviewIn, c: Caller = Depends(caller), db=Depends(db_session)):
        rv = svc.record_review(db, c.tenant, asm_id, body.reviewer, body.corrections, body.teacher_scores,
                               body.overall_score, body.comment, actor=c.actor)
        return {"review_id": rv.id, "assessment_id": asm_id, "status": "reviewed"}

    @app.get("/v1/assessments/{asm_id}/reviews")
    def list_reviews(asm_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        return [{"review_id": r.id, "reviewer": r.reviewer, "corrections": r.corrections,
                 "teacher_scores": r.teacher_scores, "overall_score": r.overall_score,
                 "comment": r.comment, "created_at": r.created_at.isoformat()}
                for r in svc.list_reviews(db, c.tenant, asm_id)]

    @app.get("/v1/review-queue")
    def queue(limit: int = Query(50, le=200), c: Caller = Depends(caller), db=Depends(db_session)):
        return [svc.assessment_view(a) for a in svc.review_queue(db, c.tenant, limit)]

    # ---- privacy
    @app.delete("/v1/students/{student_id}")
    def delete_student(student_id: str, c: Caller = Depends(caller), db=Depends(db_session)):
        """Erase a student's recordings, results and reviews (right to erasure)."""
        return svc.delete_student(db, store, c.tenant, student_id, actor=c.actor)

    @app.get("/v1/audit")
    def audit_log(limit: int = Query(100, le=1000), c: Caller = Depends(caller), db=Depends(db_session)):
        return [{"at": e.at.isoformat(), "actor": e.actor, "action": e.action, "resource": e.resource,
                 "detail": e.detail} for e in svc.audit_entries(db, c.tenant, limit)]

    return app
