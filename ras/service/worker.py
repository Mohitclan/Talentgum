"""Worker: claims queued assessments, runs the pipeline, stores results, signs webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import traceback

from ras.core.rubric import Rubric
from ras.core.score.reference import load_reference
from ras.pipeline import PipelineConfig, assess

from .config import Settings
from .db import Assessment, FileRow, Reference, RubricRow, make_session_factory, now
from .queue import claim_next, fail_or_retry
from .storage import LocalStore

log = logging.getLogger("ras.worker")


def sign_webhook(secret: str, body: bytes, timestamp: int | None = None) -> str:
    """`t=<unix>,v1=<hex hmac-sha256 of 't.body'>`; the timestamp lets receivers reject replays."""
    ts = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_webhook(secret: str, body: bytes, header: str, tolerance_s: int = 300) -> bool:
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
    except Exception:
        return False
    if abs(time.time() - ts) > tolerance_s:
        return False
    return hmac.compare_digest(sign_webhook(secret, body, ts), header)


def default_transcriber_factory(settings: Settings):
    from ras.core.transcribe import get_transcriber

    inner = get_transcriber(settings.transcriber)
    if settings.transcription_cache:
        from ras.core.transcribe.cache import CachedTranscriber

        return CachedTranscriber(inner, settings.data_dir / "transcription_cache")
    return inner


class Worker:
    def __init__(self, settings: Settings | None = None, transcriber_factory=None, session_factory=None,
                 store=None, webhook_post=None):
        self.settings = settings or Settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.Session = session_factory or make_session_factory(self.settings.database_url)
        self.store = store or LocalStore(self.settings.data_dir / "objects")
        self._factory = transcriber_factory or (lambda: default_transcriber_factory(self.settings))
        self._transcriber = None
        self._post = webhook_post

    @property
    def transcriber(self):
        if self._transcriber is None:  # load the model once, on first use
            self._transcriber = self._factory()
        return self._transcriber

    def run_once(self) -> str | None:
        """Process one job. Returns its id, or None if the queue is empty."""
        with self.Session() as db:
            a = claim_next(db, self.settings.visibility_timeout_s, self.settings.max_attempts)
            if a is None:
                return None
            asm_id = a.id
            try:
                self._process(db, a)
            except Exception as exc:  # unexpected: retry, then dead-letter
                log.exception("job %s crashed", asm_id)
                fail_or_retry(db, a, "worker_error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
                              self.settings.max_attempts)
            if a.status in ("completed", "review_required", "rejected", "failed"):
                self._notify(a)
            return asm_id

    def run_forever(self, poll_s: float = 1.0) -> None:  # pragma: no cover - loop
        log.info("worker started")
        while True:
            if self.run_once() is None:
                time.sleep(poll_s)

    def _process(self, db, a: Assessment) -> None:
        ref_row = db.query(Reference).filter_by(song_id=a.song_id, version=a.reference_version).one()
        ref_file = db.get(FileRow, ref_row.file_id)
        rec_file = db.get(FileRow, a.recording_file_id)
        rub = db.query(RubricRow).filter_by(rubric_id=a.rubric_id, version=a.rubric_version).one()
        rubric = Rubric.model_validate(rub.body)
        reference = load_reference(self.store.get_path(ref_file.key))
        result = assess(reference, self.store.get_path(rec_file.key), rubric, self.transcriber,
                        start_s=a.start_s, config=PipelineConfig())
        d = result.model_dump(mode="json")
        d["versions"]["reference_version"] = a.reference_version
        a.result = d
        a.finished_at = now()
        a.error_code, a.error_detail = result.error_code, result.error_detail
        if result.status in ("rejected", "failed"):  # deterministic outcomes: retrying cannot help
            a.status = result.status
        else:
            a.total_score = result.scoring.get("total_score")
            a.confidence = result.confidence.get("value")
            a.needs_review = bool(result.confidence.get("needs_review"))
            a.status = "review_required" if a.needs_review else "completed"
        db.commit()

    def _notify(self, a: Assessment) -> None:
        if not a.callback_url:
            return
        body = json.dumps({"event": f"assessment.{a.status}", "assessment_id": a.id, "status": a.status,
                           "external_ref": a.external_ref}).encode()
        headers = {"Content-Type": "application/json",
                   "RAS-Signature": sign_webhook(self.settings.webhook_secret, body)}
        post = self._post
        if post is None:
            import httpx

            post = lambda url, content, headers: httpx.post(url, content=content, headers=headers, timeout=10)  # noqa: E731
        for attempt in range(3):
            try:
                if post(a.callback_url, body, headers).status_code < 300:
                    return
            except Exception as exc:
                log.warning("webhook to %s failed (%s)", a.callback_url, exc)
            time.sleep(0.2 * (attempt + 1))
