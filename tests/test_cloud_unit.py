"""Phase 3 pieces that need no AWS: settings guard, webhook address check, store and signal behaviour with
fake clients, metric lines, licence register. The real S3/SQS/PostgreSQL run is tests/test_cloud_stack.py."""

import io
import json

import pytest

from ras.service.config import DEV_WEBHOOK_SECRET, Settings
from ras.service.queue import PollingSignal, SqsSignal
from ras.service.storage import LocalStore, S3Store
from ras.service.worker import callback_allowed, emit_metric


def test_production_refuses_dev_defaults(tmp_path):
    s = Settings(env="production", data_dir=tmp_path, webhook_secret=DEV_WEBHOOK_SECRET, db_url=None,
                 storage="local")
    with pytest.raises(RuntimeError) as e:
        s.check()
    msg = str(e.value)
    assert "RAS_WEBHOOK_SECRET" in msg and "SQLite" in msg and "RAS_STORAGE" in msg
    ok = Settings(env="production", data_dir=tmp_path, webhook_secret="x" * 32, db_url="postgresql+psycopg://h/db",
                  storage="s3", s3_bucket="b")
    ok.check()
    Settings(data_dir=tmp_path).check()  # local is always allowed


def test_database_url_from_parts(monkeypatch, tmp_path):
    for var in ("RAS_DB_URL", "RAS_DB_USER", "RAS_DB_NAME", "RAS_DB_PORT", "RAS_DB_SSLMODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RAS_DB_HOST", "db.internal")
    monkeypatch.setenv("RAS_DB_PASSWORD", "p@ss/word")
    url = Settings(data_dir=tmp_path).database_url
    assert url == "postgresql+psycopg://ras:p%40ss%2Fword@db.internal:5432/ras?sslmode=require"


def test_webhook_targets():
    assert not callback_allowed("http://169.254.169.254/latest/meta-data/", allow_private=False)
    assert not callback_allowed("https://127.0.0.1/hook", allow_private=False)
    assert not callback_allowed("https://10.0.0.5/hook", allow_private=False)
    assert not callback_allowed("http://example.com/hook", allow_private=False)  # https only outside dev
    assert not callback_allowed("file:///etc/passwd", allow_private=True)
    assert callback_allowed("http://localhost:9000/hook", allow_private=True)
    assert callback_allowed("https://8.8.8.8/hook", allow_private=False)


def test_local_store_delete_and_escape(tmp_path):
    s = LocalStore(tmp_path)
    s.put("tenants/t/audio/ab/x.wav", b"abc")
    assert s.exists("tenants/t/audio/ab/x.wav")
    s.release("tenants/t/audio/ab/x.wav")  # no-op locally
    assert s.get_path("tenants/t/audio/ab/x.wav").read_bytes() == b"abc"
    s.delete("tenants/t/audio/ab/x.wav")
    assert not s.exists("tenants/t/audio/ab/x.wav")
    with pytest.raises(ValueError):
        s.get_path("../outside")


class FakeS3:
    def __init__(self):
        self.objects, self.tags = {}, {}

    def put_object(self, Bucket, Key, Body, Tagging):
        self.objects[(Bucket, Key)], self.tags[(Bucket, Key)] = Body, Tagging

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def test_s3_store_prefix_tags_cache(tmp_path):
    fake = FakeS3()
    s = S3Store("bkt", tmp_path / "cache", prefix="ras", client=fake)
    s.put("tenants/t/audio/ab/x.wav", b"wav")
    s.put("tenants/t/reference/cd/y.xml", b"xml")
    assert fake.tags[("bkt", "ras/tenants/t/audio/ab/x.wav")] == "kind=audio"
    assert fake.tags[("bkt", "ras/tenants/t/reference/cd/y.xml")] == "kind=other"
    p = s.get_path("tenants/t/audio/ab/x.wav")
    assert p.read_bytes() == b"wav" and p.is_relative_to(tmp_path / "cache")
    s.release("tenants/t/audio/ab/x.wav")
    assert not p.exists()
    s.delete("tenants/t/audio/ab/x.wav")
    assert ("bkt", "ras/tenants/t/audio/ab/x.wav") not in fake.objects
    with pytest.raises(ValueError):
        s.put("tenants/../../etc", b"")


class FakeSqs:
    def __init__(self):
        self.msgs, self.deleted, self.n = [], [], 0

    def send_message(self, QueueUrl, MessageBody):
        self.n += 1
        self.msgs.append({"Body": MessageBody, "ReceiptHandle": f"r{self.n}"})

    def receive_message(self, QueueUrl, MaxNumberOfMessages, WaitTimeSeconds):
        return {"Messages": self.msgs[:MaxNumberOfMessages]}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append(ReceiptHandle)
        self.msgs = [m for m in self.msgs if m["ReceiptHandle"] != ReceiptHandle]


def test_sqs_signal_is_only_a_wakeup():
    fake = FakeSqs()
    sig = SqsSignal(queue_url="q", client=fake)
    sig.publish("asm_1")
    sig.publish("asm_2")
    assert sig.wait(20) is True and fake.deleted == ["r1"]
    sig.consume_one()
    assert fake.msgs == [] and sig.wait(0) is False
    assert PollingSignal().wait(0) is False


def test_sqs_failure_does_not_lose_the_job():
    class Broken(FakeSqs):
        def send_message(self, **kw):
            raise ConnectionError("down")

    SqsSignal(queue_url="q", client=Broken()).publish("asm_1")  # logged, not raised: the DB row is the job


def test_emf_metric_line(capsys):
    emit_metric("completed", 12.5, 25.0)
    doc = json.loads(capsys.readouterr().out)
    assert doc["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "RAS"
    assert doc["Status"] == "completed" and doc["RealTimeFactor"] == 0.5


def test_licence_register(tmp_path):
    from ras.tools.sbom import write_register

    md, cdx, flagged = write_register(tmp_path, "test")
    bom = json.loads(cdx.read_text())
    names = {c["name"].lower() for c in bom["components"]}
    assert {"numpy", "partitura", "transkun-v2-weights", "ffmpeg"} <= names
    assert bom["bomFormat"] == "CycloneDX" and "| numpy |" in md.read_text()
    assert any(f.startswith("ffmpeg") for f in flagged)  # GPL build: needs a decision
