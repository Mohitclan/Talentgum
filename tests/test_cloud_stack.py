"""The service on its cloud backends: PostgreSQL, S3 (SeaweedFS) and SQS (ElasticMQ).

Runs only inside the docker compose stack, against its own test database, bucket and queue:
    docker compose run --rm tests
"""

import json
import os
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.cloud

if os.environ.get("RAS_STORAGE") != "s3":
    pytest.skip("needs the docker compose stack (RAS_STORAGE=s3)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from ras.core.transcribe.ground_truth import GroundTruthTranscriber  # noqa: E402
from ras.service.api import create_app  # noqa: E402
from ras.service.config import Settings  # noqa: E402
from ras.service.db import Assessment, Base, FileRow, make_session_factory  # noqa: E402
from ras.service.worker import Worker  # noqa: E402
from ras.synth.musicxml import write_musicxml  # noqa: E402
from ras.synth.performance import PerfSpec, simulate_performance  # noqa: E402
from ras.synth.render import render_notes, write_audio  # noqa: E402
from ras.synth.songs import SONGS  # noqa: E402

RUBRIC = json.loads((Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json").read_text())


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    settings = Settings(data_dir=tmp_path_factory.mktemp("svc"), transcriber="ground-truth",
                        transcription_cache=False)
    assert settings.database_url.startswith("postgresql") and settings.queue == "sqs"
    Session = make_session_factory(settings.database_url)
    Base.metadata.drop_all(Session.kw["bind"])  # a clean test database each run
    app = create_app(settings)
    signal = app.state.signal
    while signal.wait(0):  # drain the test queue
        pass
    client = TestClient(app, raise_server_exceptions=False)
    tmp = tmp_path_factory.mktemp("files")

    def upload(path, kind):
        r = client.post(f"/v1/files?kind={kind}", files={"file": (Path(path).name, Path(path).read_bytes())})
        assert r.status_code == 201, r.text
        return r.json()["file_id"]

    ref = write_musicxml(SONGS["twinkle"], tmp / "twinkle.musicxml")
    assert client.post("/v1/songs", json={"song_id": "twinkle", "title": "Twinkle",
                                          "reference_file_id": upload(ref, "reference")}).status_code == 201
    assert client.post("/v1/rubrics", json=RUBRIC).status_code == 201
    return {"settings": settings, "app": app, "client": client, "upload": upload, "tmp": tmp}


def _recording(stack, twinkle, seed, **spec):
    truth = simulate_performance(twinkle, PerfSpec(seed=seed, **spec))
    audio = write_audio(render_notes(truth.notes), stack["tmp"] / f"take{seed}.m4a")
    return truth, stack["upload"](audio, "audio")


def _worker(stack, notes):
    tr = GroundTruthTranscriber()
    tr.set_notes(notes)
    app = stack["app"]
    return Worker(stack["settings"], transcriber_factory=lambda: tr, session_factory=app.state.Session,
                  store=app.state.store, signal=app.state.signal)


def test_schema_creation_is_safe_concurrently(stack):
    errors = []

    def go():
        try:
            make_session_factory(stack["settings"].database_url)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_s3_objects_and_tags(stack):
    store = stack["app"].state.store
    s3 = store.s3
    store.put("tenants/local/audio/zz/probe.wav", b"probe")
    assert store.exists("tenants/local/audio/zz/probe.wav")
    tags = s3.get_object_tagging(Bucket=store.bucket, Key=store._k("tenants/local/audio/zz/probe.wav"))["TagSet"]
    assert {"Key": "kind", "Value": "audio"} in tags
    assert store.get_path("tenants/local/audio/zz/probe.wav").read_bytes() == b"probe"
    store.delete("tenants/local/audio/zz/probe.wav")
    assert not store.exists("tenants/local/audio/zz/probe.wav")


def test_assessment_flow_on_cloud_backends(stack, twinkle):
    client = stack["client"]
    truth, fid = _recording(stack, twinkle, 1, n_wrong_pitch=2)
    r = client.post("/v1/assessments", json={"song_id": "twinkle", "recording_file_id": fid,
                                             "rubric_id": "dev-placeholder", "student_id": "cloud-s1"})
    assert r.status_code == 202, r.text
    asm = r.json()["assessment_id"]
    w = _worker(stack, truth.notes)
    assert w.signal.wait(5) is True  # the API published a wake-up on SQS
    assert w.run_once() == asm
    res = client.get(f"/v1/assessments/{asm}").json()
    assert res["status"] in ("completed", "review_required") and res["summary"]["wrong_pitch"] == 2
    # the worker dropped its local copy of the recording; the object itself stays in S3
    store = stack["app"].state.store
    with stack["app"].state.Session() as db:
        key = db.get(FileRow, fid).key
    assert not (store.cache / key).exists() and store.exists(key)
    # erasure removes the bytes from S3
    assert client.delete("/v1/students/cloud-s1").status_code == 200
    assert not store.exists(key)


def test_concurrent_workers_claim_each_job_once(stack, twinkle):
    client = stack["client"]
    truth, fid = _recording(stack, twinkle, 2)
    ids = set()
    for i in range(6):
        r = client.post("/v1/assessments", json={"song_id": "twinkle", "recording_file_id": fid,
                                                 "rubric_id": "dev-placeholder", "external_ref": f"c{i}"})
        ids.add(r.json()["assessment_id"])
    done, lock = [], threading.Lock()

    def run():
        w = _worker(stack, truth.notes)
        while (got := w.run_once()) is not None:
            with lock:
                done.append(got)

    threads = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(done) == sorted(ids)  # every job exactly once
    with stack["app"].state.Session() as db:
        assert all(db.get(Assessment, i).attempts == 1 for i in ids)
