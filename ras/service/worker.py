"""Worker: claims queued assessments, runs the pipeline, stores results, signs webhooks."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import signal as os_signal
import socket
import threading
import time
import traceback
from urllib.parse import urlsplit

from ras.core.rubric import Rubric
from ras.core.score.reference import load_reference
from ras.pipeline import PipelineConfig, assess

from .config import Settings
from .db import Assessment, FileRow, Reference, RubricRow, make_session_factory, now
from .queue import claim_next, fail_or_retry, make_signal
from .storage import make_store

log = logging.getLogger("ras.worker")


def emit_metric(status: str, seconds: float, audio_s: float | None) -> None:
    """One CloudWatch Embedded Metric Format line per job. CloudWatch Logs turns it into metrics with no
    agent; locally it is just a log line."""
    doc = {"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
        "Namespace": "RAS", "Dimensions": [["Status"], []],
        "Metrics": [{"Name": "Jobs", "Unit": "Count"}, {"Name": "JobSeconds", "Unit": "Seconds"},
                    {"Name": "RealTimeFactor", "Unit": "None"}]}]},
        "Status": status, "Jobs": 1, "JobSeconds": round(seconds, 3)}
    if audio_s:
        doc["RealTimeFactor"] = round(seconds / audio_s, 4)
    print(json.dumps(doc), flush=True)


def callback_allowed(url: str, allow_private: bool) -> bool:
    """Refuse webhooks to internal addresses (cloud metadata service, VPC hosts) unless allowed.
    Checked at send time, after DNS resolution, so a hostname cannot be re-pointed after submission."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if allow_private:
        return True
    if parts.scheme != "https":
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            return False
    return True


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
                 store=None, webhook_post=None, signal=None):
        self.settings = settings or Settings()
        self.settings.check()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.Session = session_factory or make_session_factory(self.settings.database_url)
        self.store = store or make_store(self.settings)
        self.signal = signal or make_signal(self.settings)
        self._factory = transcriber_factory or (lambda: default_transcriber_factory(self.settings))
        self._transcriber = None
        self._post = webhook_post
        self.stopping = threading.Event()

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
            self.signal.consume_one()  # keeps the broker's queue depth close to the real backlog
            t0 = time.perf_counter()
            try:
                self._process(db, a)
            except Exception as exc:  # unexpected: retry, then dead-letter
                log.exception("job %s crashed", asm_id)
                fail_or_retry(db, a, "worker_error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
                              self.settings.max_attempts)
                if a.status == "queued":
                    self.signal.publish(asm_id)
            audio_s = ((a.result or {}).get("quality") or {}).get("duration_s")
            emit_metric(a.status, time.perf_counter() - t0, audio_s)
            if a.status in ("completed", "review_required", "rejected", "failed"):
                self._notify(a)
            return asm_id

    def run_forever(self, poll_s: float = 1.0, install_signal_handlers: bool = False) -> None:  # pragma: no cover
        """Loop until stopped. On SIGTERM (ECS scale-in, deploys) the current job finishes first; a job cut
        off anyway is recovered by another worker after the visibility timeout."""
        if install_signal_handlers:
            for sig in (os_signal.SIGTERM, os_signal.SIGINT):
                os_signal.signal(sig, lambda *_: self.stopping.set())
        log.info("worker started (storage=%s, queue=%s)", self.settings.storage, self.settings.queue)
        wait_s = poll_s if self.settings.queue == "db" else 20.0
        while not self.stopping.is_set():
            if self.run_once() is None:
                self.signal.wait(wait_s)
        log.info("worker stopped")

    def _process(self, db, a: Assessment) -> None:
        ref_row = db.query(Reference).filter_by(tenant_id=a.tenant_id, song_id=a.song_id,
                                                version=a.reference_version).one()
        ref_file = db.get(FileRow, ref_row.file_id)
        rec_file = db.get(FileRow, a.recording_file_id)
        if rec_file is None or rec_file.deleted_at is not None:
            a.status, a.error_code, a.finished_at = "failed", "recording_deleted", now()
            a.error_detail = "the recording was deleted before it could be assessed"
            db.commit()
            return
        rub = db.query(RubricRow).filter_by(tenant_id=a.tenant_id, rubric_id=a.rubric_id,
                                            version=a.rubric_version).one()
        rubric = Rubric.model_validate(rub.body)
        reference = load_reference(self.store.get_path(ref_file.key))
        try:
            result = assess(reference, self.store.get_path(rec_file.key), rubric, self.transcriber,
                            start_s=a.start_s, config=PipelineConfig())
        finally:
            self.store.release(rec_file.key)
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
        if not callback_allowed(a.callback_url, self.settings.allow_private_webhooks):
            log.warning("webhook for %s not sent: %s is not an allowed public https address", a.id, a.callback_url)
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
