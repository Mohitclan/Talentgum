"""Job queue on the assessments table: claim, retry, dead-letter.

Compare-and-set claims work on SQLite and PostgreSQL alike; an SQS-backed queue can replace this
module because callers only use enqueue/claim/complete/fail.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db import Assessment, now


def claim_next(db: Session, visibility_timeout_s: int = 900, max_attempts: int = 3) -> Assessment | None:
    """Atomically take the oldest queued job. Also recovers jobs whose worker died."""
    requeue_stuck(db, visibility_timeout_s, max_attempts)
    for _ in range(5):  # another worker may win the race for a candidate
        cand = db.execute(select(Assessment.id).where(Assessment.status == "queued")
                          .order_by(Assessment.created_at).limit(1)).scalar()
        if cand is None:
            return None
        res = db.execute(update(Assessment).where(Assessment.id == cand, Assessment.status == "queued")
                         .values(status="processing", locked_at=now(), attempts=Assessment.attempts + 1))
        db.commit()
        if res.rowcount == 1:
            return db.get(Assessment, cand)
    return None


def requeue_stuck(db: Session, visibility_timeout_s: int, max_attempts: int) -> int:
    cutoff = now() - dt.timedelta(seconds=visibility_timeout_s)
    stuck = db.execute(select(Assessment).where(Assessment.status == "processing",
                                                Assessment.locked_at < cutoff)).scalars().all()
    for a in stuck:
        if a.attempts >= max_attempts:
            a.status, a.error_code = "failed", "worker_timeout"
            a.error_detail = "the job was claimed but never finished"
            a.finished_at = now()
        else:
            a.status = "queued"
    db.commit()
    return len(stuck)


def fail_or_retry(db: Session, a: Assessment, code: str, detail: str, max_attempts: int) -> str:
    """Transient failure: back to the queue until attempts run out, then the dead-letter state."""
    a.error_code, a.error_detail = code, detail[:2000]
    if a.attempts >= max_attempts:
        a.status, a.finished_at = "failed", now()
    else:
        a.status = "queued"
    db.commit()
    return a.status
