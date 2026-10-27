import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.service import services as svc
from ras.service.api import create_app
from ras.service.config import Settings
from ras.service.db import Assessment, FileRow, SchemaError, make_session_factory, now
from ras.service.worker import Worker
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import render_notes, write_audio
from ras.synth.songs import SONGS

RUBRIC = json.loads((Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json").read_text())


@pytest.fixture
def sys_(tmp_path, twinkle):
    settings = Settings(data_dir=tmp_path / "svc", transcriber="ground-truth", transcription_cache=False)
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=False)
    tr = GroundTruthTranscriber()
    worker = Worker(settings, transcriber_factory=lambda: tr, session_factory=app.state.Session,
                    store=app.state.store)
    with app.state.Session() as db:
        for t in ("acme", "beta"):
            svc.create_tenant(db, t, t.title())
        keys = {t: svc.create_api_key(db, t) for t in ("acme", "beta")}
    ref = write_musicxml(SONGS["twinkle"], tmp_path / "t.musicxml")
    truth = simulate_performance(twinkle, PerfSpec(seed=1, n_wrong_pitch=2))
    audio = write_audio(render_notes(truth.notes), tmp_path / "a.m4a")

    def as_(tenant):
        return {"X-API-Key": keys[tenant]}

    def setup(tenant, student="s1"):
        h = as_(tenant)
        rf = client.post("/v1/files?kind=reference", headers=h,
                         files={"file": ("t.musicxml", ref.read_bytes())}).json()["file_id"]
        client.post("/v1/songs", headers=h, json={"song_id": "twinkle", "title": "T", "reference_file_id": rf})
        client.post("/v1/rubrics", headers=h, json=RUBRIC)
        af = client.post("/v1/files?kind=audio", headers=h, files={"file": ("a.m4a", audio.read_bytes())}).json()
        r = client.post("/v1/assessments", headers=h, json={"song_id": "twinkle", "recording_file_id": af["file_id"],
                                                            "rubric_id": "dev-placeholder", "student_id": student})
        tr.set_notes(truth.notes)
        worker.run_once()
        return r.json()["assessment_id"], af["file_id"]

    return type("S", (), dict(app=app, client=client, worker=worker, keys=keys, as_=staticmethod(as_),
                              setup=staticmethod(setup), settings=settings))


def test_keys_are_required_once_any_exist(sys_):
    c = sys_.client
    assert c.get("/v1/songs").status_code == 401
    assert c.get("/v1/songs", headers={"X-API-Key": "ras_nope"}).json()["code"] == "unauthorized"
    assert c.get("/v1/me", headers=sys_.as_("acme")).json()["tenant"] == "acme"
    assert c.get("/v1/health").json()["auth"] == "keys"


def test_tenants_cannot_see_each_others_data(sys_):
    c = sys_.client
    asm, fid = sys_.setup("acme")
    b = sys_.as_("beta")
    assert c.get("/v1/songs", headers=b).json() == []
    assert c.get(f"/v1/assessments/{asm}", headers=b).status_code == 404
    assert c.get(f"/v1/assessments/{asm}/notes", headers=b).status_code == 404
    assert c.get(f"/v1/files/{fid}/content", headers=b).status_code == 404
    assert c.get("/v1/assessments", headers=b).json() == []
    assert c.post(f"/v1/assessments/{asm}/review", headers=b, json={"reviewer": "x"}).status_code == 404
    assert c.post("/v1/assessments", headers=b, json={"song_id": "twinkle", "recording_file_id": fid,
                                                       "rubric_id": "dev-placeholder"}).status_code == 404
    # the same song id and the same audio can exist in both tenants, independently
    asm_b, fid_b = sys_.setup("beta")
    assert c.get(f"/v1/assessments/{asm_b}", headers=b).json()["status"] in ("completed", "review_required")
    with sys_.app.state.Session() as db:
        ka, kb = db.get(FileRow, fid).key, db.get(FileRow, fid_b).key
    assert ka.startswith("tenants/acme/") and kb.startswith("tenants/beta/")


def test_key_rotation_and_revocation(sys_):
    with sys_.app.state.Session() as db:
        new = svc.create_api_key(db, "acme", "rotation")
        assert svc.revoke_api_key(db, sys_.keys["acme"][:12]) == 1
    c = sys_.client
    assert c.get("/v1/songs", headers=sys_.as_("acme")).status_code == 401
    assert c.get("/v1/songs", headers={"X-API-Key": new}).status_code == 200
    with sys_.app.state.Session() as db:
        from ras.service.db import ApiKey

        assert all(k.key_hash != new for k in db.query(ApiKey))  # only hashes are stored


def test_rubric_test_previews_score_changes_without_saving(sys_):
    c, h = sys_.client, sys_.as_("acme")
    asm, _ = sys_.setup("acme")
    before = c.get(f"/v1/assessments/{asm}", headers=h).json()["total_score"]
    strict = json.loads(json.dumps(RUBRIC))
    strict["criteria"][0]["map"]["from"] = [80, 100]  # much harsher on note accuracy
    r = c.post("/v1/rubrics/test", headers=h, json={"rubric": strict}).json()
    assert r["assessments"] == 1 and r["rows"][0]["current_score"] == before
    assert r["rows"][0]["candidate_score"] < before and r["mean_delta"] < 0
    same = c.post("/v1/rubrics/test", headers=h, json={"rubric": RUBRIC}).json()
    assert same["rows"][0]["delta"] == 0.0
    assert c.get(f"/v1/assessments/{asm}", headers=h).json()["total_score"] == before  # nothing changed
    bad = c.post("/v1/rubrics/test", headers=h, json={"rubric": {"rubric_id": "x", "criteria": []}})
    assert bad.status_code == 422


def test_student_erasure_removes_data_and_bytes_and_is_audited(sys_):
    c, h = sys_.client, sys_.as_("acme")
    asm, fid = sys_.setup("acme", student="kid-7")
    keep, _ = sys_.setup("acme", student="kid-8")
    with sys_.app.state.Session() as db:
        path = sys_.app.state.store.get_path(db.get(FileRow, fid).key)
    assert path.exists()
    c.get(f"/v1/files/{fid}/content", headers=h)
    out = c.delete("/v1/students/kid-7", headers=h).json()
    assert out["assessments_deleted"] == 1 and out["recordings_deleted"] == 1
    assert c.get(f"/v1/assessments/{asm}", headers=h).status_code == 404
    assert c.get(f"/v1/files/{fid}/content", headers=h).status_code == 410
    # kid-8 uploaded identical bytes: they are shared, so they must survive kid-7's erasure
    assert path.exists() and c.get(f"/v1/assessments/{keep}", headers=h).status_code == 200
    actions = [e["action"] for e in c.get("/v1/audit", headers=h).json()]
    assert "student.delete" in actions and "recording.read" in actions
    assert c.delete("/v1/students/kid-7", headers=h).status_code == 404
    assert c.get("/v1/audit", headers=sys_.as_("beta")).json() == []  # audit is per tenant too


def test_retention_purges_old_recordings_but_keeps_results(sys_):
    c, h = sys_.client, sys_.as_("acme")
    asm, fid = sys_.setup("acme")
    with sys_.app.state.Session() as db:
        db.get(Assessment, asm).finished_at = now() - dt.timedelta(days=40)
        db.commit()
        assert svc.purge_audio(db, sys_.app.state.store, older_than_days=30) == 1
        assert svc.purge_audio(db, sys_.app.state.store, older_than_days=30) == 0
    assert c.get(f"/v1/files/{fid}/content", headers=h).status_code == 410
    assert c.get(f"/v1/assessments/{asm}", headers=h).json()["total_score"] is not None


def test_old_schema_is_refused_with_a_clear_message(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("create table assessments (id text primary key, status text)")
    con.commit()
    con.close()
    with pytest.raises(SchemaError, match="older schema"):
        make_session_factory(f"sqlite:///{db}")
