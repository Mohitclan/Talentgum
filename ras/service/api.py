"""REST API under /v1. The web UI uses only this API."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import services as svc
from .config import Settings
from .db import Assessment, Review, RubricRow, Song, make_session_factory
from .storage import LocalStore

UI_DIR = Path(__file__).parent / "ui"


class SongIn(BaseModel):
    song_id: str
    title: str
    reference_file_id: str
    bpm: float | None = None


class ReferenceIn(BaseModel):
    reference_file_id: str
    bpm: float | None = None


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


def problem(status: int, code: str, detail: str) -> JSONResponse:
    """RFC 9457 problem details with a stable machine-readable code."""
    return JSONResponse({"type": f"https://ras.local/errors/{code}", "title": code.replace("_", " "),
                         "status": status, "code": code, "detail": detail},
                        status_code=status, media_type="application/problem+json")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Session = make_session_factory(settings.database_url)
    store = LocalStore(settings.data_dir / "objects")
    app = FastAPI(title="Recording Assessment System", version="0.2.0")
    app.state.settings, app.state.Session, app.state.store = settings, Session, store

    def db_session():
        with Session() as db:
            yield db

    def auth(x_api_key: str | None = Header(None)):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(401, "missing or wrong X-API-Key")

    guard = [Depends(auth)]

    @app.exception_handler(svc.ServiceError)
    async def _service_error(_: Request, exc: svc.ServiceError):
        return problem(exc.status, exc.code, exc.detail)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return problem(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):  # never leak internals
        return problem(500, "internal_error", "unexpected server error")

    # ---- health and UI
    @app.get("/v1/health")
    def health(db=Depends(db_session)):
        db.execute(select(Song.id).limit(1))
        return {"status": "ok", "transcriber": settings.transcriber}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    @app.get("/ui/", include_in_schema=False)
    def ui():
        return FileResponse(UI_DIR / "index.html")

    # ---- files
    @app.post("/v1/files", status_code=201, dependencies=guard)
    async def upload(file: UploadFile = File(...), kind: str = Query(..., pattern="^(audio|reference)$"),
                     db=Depends(db_session)):
        data = await file.read()
        row = svc.save_file(db, store, file.filename or "upload", data, kind, settings.max_upload_mb)
        return {"file_id": row.id, "sha256": row.sha256, "size": row.size, "filename": row.filename, "kind": row.kind}

    @app.get("/v1/files/{file_id}/content", dependencies=guard)
    def file_content(file_id: str, db=Depends(db_session)):
        row = svc.get_file(db, file_id)
        return FileResponse(store.get_path(row.key), filename=row.filename)

    # ---- songs
    def song_view(db, song: Song):
        refs = db.execute(select(svc.Reference).where(svc.Reference.song_id == song.id)
                          .order_by(svc.Reference.version)).scalars().all()
        return {"song_id": song.id, "title": song.title, "references": [
            {"version": r.version, "sha256": r.sha256[:16], "notes": r.n_notes, "measures": r.n_measures,
             "bpm": r.bpm, "warnings": r.warnings} for r in refs]}

    @app.post("/v1/songs", status_code=201, dependencies=guard)
    def create_song(body: SongIn, db=Depends(db_session)):
        song, _ = svc.create_song(db, store, body.song_id, body.title, body.reference_file_id, body.bpm)
        return song_view(db, song)

    @app.get("/v1/songs", dependencies=guard)
    def list_songs(db=Depends(db_session)):
        return [song_view(db, s) for s in db.execute(select(Song).order_by(Song.id)).scalars()]

    @app.get("/v1/songs/{song_id}", dependencies=guard)
    def get_song(song_id: str, db=Depends(db_session)):
        song = db.get(Song, song_id)
        if song is None:
            raise svc.ServiceError("song_not_found", f"no song {song_id}", 404)
        return song_view(db, song)

    @app.post("/v1/songs/{song_id}/references", status_code=201, dependencies=guard)
    def add_reference(song_id: str, body: ReferenceIn, db=Depends(db_session)):
        svc.add_reference(db, store, song_id, body.reference_file_id, body.bpm)
        return song_view(db, db.get(Song, song_id))

    # ---- rubrics
    @app.post("/v1/rubrics", status_code=201, dependencies=guard)
    def create_rubric(body: dict, db=Depends(db_session)):
        row = svc.create_rubric(db, body)
        return {"rubric_id": row.rubric_id, "version": row.version}

    @app.get("/v1/rubrics", dependencies=guard)
    def list_rubrics(db=Depends(db_session)):
        rows = db.execute(select(RubricRow).order_by(RubricRow.rubric_id, RubricRow.version)).scalars()
        return [{"rubric_id": r.rubric_id, "version": r.version,
                 "criteria": [c["id"] for c in r.body["criteria"]]} for r in rows]

    @app.get("/v1/rubrics/{rubric_id}", dependencies=guard)
    def get_rubric(rubric_id: str, version: int | None = None, db=Depends(db_session)):
        return svc.get_rubric(db, rubric_id, version).body

    # ---- assessments
    @app.post("/v1/assessments", status_code=202, dependencies=guard)
    def create_assessment(body: AssessmentIn, idempotency_key: str | None = Header(None),
                          db=Depends(db_session)):
        a, created = svc.create_assessment(db, body.model_dump(), idempotency_key)
        return JSONResponse({"assessment_id": a.id, "status": a.status, "created": created},
                            status_code=202 if created else 200)

    @app.get("/v1/assessments", dependencies=guard)
    def list_assessments(status: str | None = None, student_id: str | None = None,
                         limit: int = Query(50, le=200), db=Depends(db_session)):
        q = select(Assessment).order_by(Assessment.created_at.desc()).limit(limit)
        if status:
            q = q.where(Assessment.status == status)
        if student_id:
            q = q.where(Assessment.student_id == student_id)
        return [svc.assessment_view(a) for a in db.execute(q).scalars()]

    @app.get("/v1/assessments/{asm_id}", dependencies=guard)
    def get_assessment(asm_id: str, db=Depends(db_session)):
        return svc.assessment_view(svc.get_assessment(db, asm_id))

    @app.get("/v1/assessments/{asm_id}/notes", dependencies=guard)
    def get_notes(asm_id: str, db=Depends(db_session)):
        a = svc.get_assessment(db, asm_id)
        if not a.result:
            raise svc.ServiceError("not_ready", f"assessment is {a.status}", 409)
        return {"assessment_id": a.id, "notes": a.result.get("notes", [])}

    @app.post("/v1/assessments/{asm_id}/review", status_code=201, dependencies=guard)
    def review(asm_id: str, body: ReviewIn, db=Depends(db_session)):
        rv = svc.record_review(db, asm_id, body.reviewer, body.corrections, body.teacher_scores,
                               body.overall_score, body.comment)
        return {"review_id": rv.id, "assessment_id": asm_id, "status": "reviewed"}

    @app.get("/v1/assessments/{asm_id}/reviews", dependencies=guard)
    def list_reviews(asm_id: str, db=Depends(db_session)):
        svc.get_assessment(db, asm_id)
        rows = db.execute(select(Review).where(Review.assessment_id == asm_id)).scalars()
        return [{"review_id": r.id, "reviewer": r.reviewer, "corrections": r.corrections,
                 "teacher_scores": r.teacher_scores, "overall_score": r.overall_score,
                 "comment": r.comment, "created_at": r.created_at.isoformat()} for r in rows]

    @app.get("/v1/review-queue", dependencies=guard)
    def queue(limit: int = Query(50, le=200), db=Depends(db_session)):
        return [svc.assessment_view(a) for a in svc.review_queue(db, limit)]

    return app
