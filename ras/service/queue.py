"""Job queue on the assessments table: claim, retry, dead-letter.

The table is the source of truth: compare-and-set claims work on SQLite and PostgreSQL alike, and a
crashed worker's job is recovered after the visibility timeout. A *signal* only says "there may be
work": the DB signal is plain polling; the SQS signal lets idle workers block on a long poll instead,
and gives CloudWatch a queue depth to scale workers on. A lost or duplicated message therefore never
loses or doubles a job.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db import Assessment, now

log = logging.getLogger("ras.queue")


class Signal(Protocol):
    def publish(self, assessment_id: str) -> None: ...
    def wait(self, timeout_s: float) -> bool: ...  # True if a signal arrived (it is consumed)
    def consume_one(self) -> None: ...  # take one pending signal without waiting, if any


class PollingSignal:
    """No broker: workers poll the table."""

    def publish(self, assessment_id: str) -> None:
        pass

    def wait(self, timeout_s: float) -> bool:
        time.sleep(timeout_s)
        return False

    def consume_one(self) -> None:
        pass


class SqsSignal:
    def __init__(self, queue_url: str | None = None, queue_name: str | None = None, client=None,
                 create=False):
        if client is None:
            import boto3

            client = boto3.client("sqs")
        self.sqs = client
        if create:  # local stand-ins only; on AWS the queue comes from Terraform
            queue_url = client.create_queue(QueueName=queue_name)["QueueUrl"]
        elif not queue_url:
            queue_url = client.get_queue_url(QueueName=queue_name)["QueueUrl"]
        self.url = queue_url

    def publish(self, assessment_id: str) -> None:
        try:
            self.sqs.send_message(QueueUrl=self.url, MessageBody=assessment_id)
        except Exception:  # the job is safe in the table; a worker will find it on its next scan
            log.warning("could not signal job %s", assessment_id, exc_info=True)

    def _receive(self, wait_s: int) -> bool:
        msgs = self.sqs.receive_message(QueueUrl=self.url, MaxNumberOfMessages=1,
                                        WaitTimeSeconds=wait_s).get("Messages", [])
        for m in msgs:  # the table claim is authoritative, so the message is done once seen
            self.sqs.delete_message(QueueUrl=self.url, ReceiptHandle=m["ReceiptHandle"])
        return bool(msgs)

    def wait(self, timeout_s: float) -> bool:
        try:
            return self._receive(max(0, min(20, int(timeout_s))))
        except Exception:
            log.warning("queue receive failed; polling the table instead", exc_info=True)
            time.sleep(min(timeout_s, 5))
            return False

    def consume_one(self) -> None:
        try:
            self._receive(0)
        except Exception:
            log.debug("queue receive failed", exc_info=True)


def make_signal(settings) -> Signal:
    if settings.queue == "sqs":
        return SqsSignal(settings.sqs_queue_url, settings.sqs_queue_name, create=settings.create_resources)
    return PollingSignal()


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
