import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.service.api import create_app
from ras.service.config import Settings
from ras.service.db import Assessment, now
from ras.service.worker import Worker, sign_webhook, verify_webhook
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import render_notes, write_audio
from ras.synth.songs import SONGS

RUBRIC = json.loads((Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json").read_text())


class Env:
    def __init__(self, tmp_path, twinkle):
        self.settings = Settings(data_dir=tmp_path / "svc", transcriber="ground-truth", transcription_cache=False)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.tr = GroundTruthTranscriber()
        self.worker = Worker(self.settings, transcriber_factory=lambda: self.tr,
                             session_factory=self.app.state.Session, store=self.app.state.store)
        self.twinkle = twinkle
        self.tmp = tmp_path

    def upload(self, path, kind):
        with open(path, "rb") as f:
            r = self.client.post(f"/v1/files?kind={kind}", files={"file": (Path(path).name, f.read())})
        assert r.status_code == 201, r.text
        return r.json()["file_id"]

    def setup_song_and_rubric(self):
        ref = write_musicxml(SONGS["twinkle"], self.tmp / "twinkle.musicxml")
        r = self.client.post("/v1/songs", json={"song_id": "twinkle", "title": "Twinkle",
                                                "reference_file_id": self.upload(ref, "reference")})
        assert r.status_code == 201, r.text
        assert self.client.post("/v1/rubrics", json=RUBRIC).status_code == 201

    def recording(self, name="take", **spec):
        truth = simulate_performance(self.twinkle, PerfSpec(seed=1, **spec))
        audio = write_audio(render_notes(truth.notes), self.tmp / f"{name}.m4a")
        return truth, self.upload(audio, "audio")

    def submit(self, file_id, key=None, **extra):
        headers = {"Idempotency-Key": key} if key else {}
        return self.client.post("/v1/assessments", headers=headers, json={
            "song_id": "twinkle", "recording_file_id": file_id, "rubric_id": "dev-placeholder",
            "student_id": "s1", **extra})


@pytest.fixture
def env(tmp_path, twinkle):
    e = Env(tmp_path, twinkle)
    e.setup_song_and_rubric()
    return e


def test_full_flow_upload_assess_review(env):
    truth, fid = env.recording(n_wrong_pitch=2, n_missing=2)
    env.tr.set_notes(truth.notes)
    r = env.submit(fid)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    asm = r.json()["assessment_id"]
    assert env.client.get(f"/v1/assessments/{asm}").json()["status"] == "queued"

    assert env.worker.run_once() == asm
    res = env.client.get(f"/v1/assessments/{asm}").json()
    assert res["status"] in ("completed", "review_required")
    assert res["summary"]["wrong_pitch"] == 2 and res["total_score"] is not None
    assert res["rubric"] == {"id": "dev-placeholder", "version": 1} and res["reference_version"] == 1
    assert res["versions"]["reference_version"] == 1 and res["feedback"]
    notes = env.client.get(f"/v1/assessments/{asm}/notes").json()["notes"]
    assert sum(1 for n in notes if n["status"] == "wrong_pitch") == 2
    assert env.worker.run_once() is None  # queue is empty

    wrong = next(n for n in notes if n["status"] == "wrong_pitch")
    rv = env.client.post(f"/v1/assessments/{asm}/review", json={
        "reviewer": "teacher-1", "overall_score": 70, "teacher_scores": {"note_accuracy": 65},
        "corrections": [{"kind": "note", "ref_index": wrong["ref_index"], "status": "correct"}]})
    assert rv.status_code == 201
    assert env.client.get(f"/v1/assessments/{asm}").json()["status"] == "reviewed"
    reviews = env.client.get(f"/v1/assessments/{asm}/reviews").json()
    assert reviews[0]["reviewer"] == "teacher-1" and reviews[0]["overall_score"] == 70


def test_low_confidence_goes_to_review_queue(env):
    truth, fid = env.recording()
    env.tr.set_notes([])  # transcriber hears nothing: cannot trust the result
    asm = env.submit(fid).json()["assessment_id"]
    env.worker.run_once()
    q = env.client.get("/v1/review-queue").json()
    assert [a["assessment_id"] for a in q] == [asm] and q[0]["needs_review"]


def test_idempotency_key(env):
    _, fid = env.recording()
    a = env.submit(fid, key="k1")
    b = env.submit(fid, key="k1")
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["assessment_id"] == b.json()["assessment_id"] and b.json()["created"] is False
    c = env.submit(fid, key="k1", student_id="someone-else")
    assert c.status_code == 409 and c.json()["code"] == "idempotency_key_reused"
    assert c.headers["content-type"].startswith("application/problem+json")


def test_validation_errors_are_problem_json(env, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    r = env.client.post("/v1/files?kind=audio", files={"file": ("x.txt", b"hi")})
    assert r.status_code == 415 and r.json()["code"] == "unsupported_file_type"
    assert env.client.post("/v1/files?kind=audio", files={"file": ("x.wav", b"")}).json()["code"] == "empty_file"
    assert env.client.post("/v1/rubrics", json={"rubric_id": "bad", "criteria": [
        {"id": "a", "weight": 0.5, "metric": "note_accuracy_pct"}]}).json()["code"] == "invalid_rubric"
    assert env.submit("file_nope").json()["code"] == "file_not_found"
    assert env.client.get("/v1/assessments/asm_nope").status_code == 404
    junk = env.client.post("/v1/files?kind=reference", files={"file": ("a.musicxml", b"<not xml")}).json()["file_id"]
    r = env.client.post("/v1/songs", json={"song_id": "junk", "title": "J", "reference_file_id": junk})
    assert r.status_code == 422 and r.json()["code"] == "reference_unreadable"
    assert env.client.get("/v1/songs/junk").status_code == 404  # failed creation leaves nothing behind


def test_song_references_and_rubrics_are_versioned(env, tmp_path):
    ref2 = write_musicxml(SONGS["twinkle"], tmp_path / "t2.musicxml", repeat=True)
    r = env.client.post("/v1/songs/twinkle/references", json={"reference_file_id": env.upload(ref2, "reference")})
    assert [x["version"] for x in r.json()["references"]] == [1, 2]
    assert r.json()["references"][1]["notes"] == 2 * r.json()["references"][0]["notes"]
    assert env.client.post("/v1/rubrics", json=RUBRIC).json()["version"] == 2
    assert env.client.get("/v1/rubrics/dev-placeholder?version=1").json()["version"] == 1
    assert env.client.post("/v1/songs", json={"song_id": "twinkle", "title": "again",
                                              "reference_file_id": "x"}).json()["code"] == "song_exists"


def test_rejected_recording_is_final_not_retried(env, tmp_path):
    import numpy as np
    import soundfile as sf

    p = tmp_path / "silent.wav"
    sf.write(str(p), np.zeros(44100 * 5, dtype=np.float32), 44100)
    asm = env.submit(env.upload(p, "audio")).json()["assessment_id"]
    env.worker.run_once()
    a = env.client.get(f"/v1/assessments/{asm}").json()
    assert a["status"] == "rejected" and a["error_code"] == "recording_quality_low" and a["attempts"] == 1


def test_crashing_job_is_retried_then_dead_lettered(env):
    _, fid = env.recording()

    def boom(*_):
        raise RuntimeError("model exploded")

    env.tr.transcribe = boom
    asm = env.submit(fid).json()["assessment_id"]
    for expected in ("queued", "queued", "failed"):
        env.worker.run_once()
        a = env.client.get(f"/v1/assessments/{asm}").json()
        assert a["status"] == expected
    assert a["attempts"] == 3 and a["error_code"] == "worker_error" and "exploded" in a["error_detail"]
    assert env.worker.run_once() is None


def test_stuck_job_is_recovered(env):
    _, fid = env.recording()
    asm = env.submit(fid).json()["assessment_id"]
    with env.app.state.Session() as db:
        a = db.get(Assessment, asm)
        a.status, a.attempts, a.locked_at = "processing", 1, now() - dt.timedelta(hours=1)
        db.commit()
    env.tr.set_notes([])
    assert env.worker.run_once() == asm  # reclaimed after the visibility timeout


def test_api_key_is_enforced_when_configured(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "s", api_key="secret", transcriber="ground-truth"))
    c = TestClient(app)
    assert c.get("/v1/songs").status_code == 401
    assert c.get("/v1/songs", headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/v1/health").status_code == 200  # health stays open for load balancers


def test_webhook_is_signed_and_replay_safe(env):
    body = b'{"event":"assessment.completed"}'
    header = sign_webhook("s3cret", body, timestamp=1_700_000_000)
    assert verify_webhook("s3cret", body, header, tolerance_s=10**12)
    assert not verify_webhook("other", body, header, tolerance_s=10**12)
    assert not verify_webhook("s3cret", body + b"x", header, tolerance_s=10**12)
    assert not verify_webhook("s3cret", body, header)  # old timestamp: rejected as replay


def test_completion_triggers_signed_webhook(env):
    calls = []
    env.worker._post = lambda url, content, headers: calls.append((url, content, headers)) or type(
        "R", (), {"status_code": 200})()
    truth, fid = env.recording()
    env.tr.set_notes(truth.notes)
    asm = env.submit(fid, callback_url="https://client.example/hook", external_ref="sub-1").json()["assessment_id"]
    env.worker.run_once()
    url, content, headers = calls[0]
    assert url == "https://client.example/hook" and json.loads(content)["assessment_id"] == asm
    assert verify_webhook(env.settings.webhook_secret, content, headers["RAS-Signature"])


def test_ui_is_served(env):
    r = env.client.get("/ui/")
    assert r.status_code == 200 and "Recording Assessment" in r.text
