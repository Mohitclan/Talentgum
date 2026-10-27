import json
from pathlib import Path

from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.service import services as svc
from ras.service.config import Settings
from ras.service.db import Assessment, make_session_factory
from ras.service.evaluation import ReviewedCase, calibrate_confidence, evaluate, load_cases
from ras.service.ingest import ingest_manifest
from ras.service.storage import LocalStore
from ras.service.worker import Worker
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import render_notes, write_audio
from ras.synth.songs import SONGS

RUBRIC_PATH = Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json"


def _system(tmp_path):
    s = Settings(data_dir=tmp_path / "svc", transcriber="ground-truth", transcription_cache=False)
    S = make_session_factory(s.database_url)
    store = LocalStore(s.data_dir / "objects")
    tr = GroundTruthTranscriber()
    return s, S, store, tr, Worker(s, transcriber_factory=lambda: tr, session_factory=S, store=store)


def test_ingest_manifest_queues_everything_and_reports_bad_rows(tmp_path, twinkle):
    d = tmp_path / "client"
    write_musicxml(SONGS["twinkle"], d / "refs" / "twinkle.musicxml")
    (d / "rubric.json").write_text(RUBRIC_PATH.read_text())
    truth = simulate_performance(twinkle, PerfSpec(seed=1))
    write_audio(render_notes(truth.notes), d / "audio" / "a.m4a")
    (d / "manifest.json").write_text(json.dumps({
        "rubric": "rubric.json",
        "songs": [{"song_id": "twinkle", "title": "Twinkle", "reference": "refs/twinkle.musicxml"}],
        "recordings": [{"song_id": "twinkle", "audio": "audio/a.m4a", "student_id": "s1", "start_s": 1.0},
                       {"song_id": "nope", "audio": "audio/a.m4a"},
                       {"song_id": "twinkle", "audio": "audio/missing.m4a"}]}))
    s, S, store, _, _ = _system(tmp_path)
    with S() as db:
        out = ingest_manifest(db, store, d / "manifest.json")
        assert out["songs"] == 1 and out["rubric"] == "dev-placeholder@1" and len(out["queued"]) == 1
        assert len(out["errors"]) == 2
        assert db.get(Assessment, out["queued"][0]).start_s == 1.0


def _reviewed_case(twinkle, tmp_path, name, *, spec, reject=0):
    """Assess a simulated take, then have a 'teacher' who knows the truth correct the result."""
    s, S, store, tr, worker = _system(tmp_path / name)
    truth = simulate_performance(twinkle, spec)
    with S() as db:
        ref = write_musicxml(SONGS["twinkle"], tmp_path / name / "t.musicxml")
        rf = svc.save_file(db, store, "local", "t.musicxml", Path(ref).read_bytes(), "reference")
        svc.create_song(db, store, "local", "twinkle", "Twinkle", rf.id)
        svc.create_rubric(db, "local", json.loads(RUBRIC_PATH.read_text()))
        audio = write_audio(render_notes(truth.notes), tmp_path / name / "a.m4a")
        af = svc.save_file(db, store, "local", "a.m4a", audio.read_bytes(), "audio")
        a, _ = svc.create_assessment(db, "local", {"song_id": "twinkle", "recording_file_id": af.id,
                                          "rubric_id": "dev-placeholder", "student_id": name})
        asm = a.id
    tr.set_notes(truth.notes)
    worker.run_once()
    with S() as db:
        a = db.get(Assessment, asm)
        notes = a.result["notes"]
        corrections = []
        # the teacher overturns `reject` of the system's wrong-pitch claims
        for n in [n for n in notes if n["status"] == "wrong_pitch"][:reject]:
            corrections.append({"kind": "note", "ref_index": n["ref_index"], "status": "correct"})
        svc.record_review(db, "local", asm, "t", corrections, {}, a.total_score, None)
        return load_cases(db)[0]


def test_evaluation_counts_agreement_and_overturned_claims(twinkle, tmp_path):
    agree = _reviewed_case(twinkle, tmp_path, "agree", spec=PerfSpec(seed=2, n_wrong_pitch=3, n_missing=2))
    assert agree.counts["wrong_pitch"] == {"tp": 3, "fp": 0, "fn": 0}
    assert agree.counts["missing"] == {"tp": 2, "fp": 0, "fn": 0}
    overturned = _reviewed_case(twinkle, tmp_path, "overturn", spec=PerfSpec(seed=3, n_wrong_pitch=3), reject=2)
    assert overturned.counts["wrong_pitch"] == {"tp": 1, "fp": 2, "fn": 0}

    rep = evaluate([agree, overturned])
    d = rep["error_detection"]
    assert rep["n_reviewed"] == 2 and 0.5 < d["precision"] < 1.0 and d["recall"] == 1.0
    assert rep["score_agreement"]["mae"] == 0.0  # the teacher kept the system score


def test_confidence_calibration_picks_the_lowest_safe_threshold():
    def case(conf, tp, fp):
        return ReviewedCase("a", "s", "song", conf, 80, 80, {}, {},
                            {"wrong_pitch": {"tp": tp, "fp": fp, "fn": 0}, "wrong_octave": {"tp": 0, "fp": 0, "fn": 0},
                             "missing": {"tp": 0, "fp": 0, "fn": 0}, "extra": {"tp": 0, "fp": 0, "fn": 0}})
    cases = [case(0.95, 10, 0), case(0.9, 9, 1), case(0.5, 3, 5), case(0.3, 1, 6)]
    cal = calibrate_confidence(cases, target_false_alarm=0.10)
    assert cal["recommended_threshold"] == 0.9  # at 0.5 and below too many claims are wrong
    row = {r["threshold"]: r for r in cal["table"]}
    assert row[0.9]["passed"] == 2 and row[0.9]["false_alarm_rate"] == 0.05
    assert row[0.0]["false_alarm_rate"] > 0.3
