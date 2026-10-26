import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ras.core.rubric import load_rubric
from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.pipeline import PipelineConfig, assess
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import degrade, render_notes, write_audio
from ras.types import PerfNote

RUBRIC = load_rubric(Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json")


@pytest.fixture(scope="module")
def take(twinkle, tmp_path_factory):
    d = tmp_path_factory.mktemp("take")
    truth = simulate_performance(twinkle, PerfSpec(seed=1, n_wrong_pitch=2, n_missing=2, n_extra=1,
                                                   start_offset_s=3.0))
    y = degrade(render_notes(truth.notes), rt60=0.3, snr_db=30)
    return truth, write_audio(y, d / "take.m4a")


def test_end_to_end_result_is_complete_and_json(song_dir, take):
    truth, audio = take
    start = min(n.onset for n in truth.notes) - 0.1
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber(truth.notes),
                 start_s=start, config=PipelineConfig(verify_missing=False))
    assert res.status == "completed"
    d = json.loads(json.dumps(res.model_dump(mode="json")))  # plain JSON, no numpy leaks
    assert d["summary"]["wrong_pitch"] == 2 and d["summary"]["missing"] == 2
    assert d["scoring"]["total_score"] is not None
    assert d["versions"]["rubric"] == "dev-placeholder@1" and d["versions"]["reference_sha256"]
    assert 0 <= d["confidence"]["value"] <= 1
    assert any("bar" in line for line in d["feedback"])
    assert all(k in d["timings_s"] for k in ("decode_and_quality", "transcribe", "total"))


def test_crop_keeps_times_in_the_original_timeline(song_dir, take):
    truth, audio = take
    start = min(n.onset for n in truth.notes) - 0.1  # ~2.9 s, so the pipeline crops
    offset = start - 1.0
    shifted = [PerfNote(n.pitch, n.onset - offset, n.offset - offset, n.velocity) for n in truth.notes]
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber(shifted),
                 start_s=start)
    onsets = [n["perf_onset_s"] for n in res.notes if n["perf_onset_s"] is not None]
    assert abs(min(onsets) - min(n.onset for n in truth.notes)) < 1e-6
    assert not res.start_timestamp["suspect"]


def test_wrong_start_timestamp_is_flagged(song_dir, take):
    truth, audio = take
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber(truth.notes),
                 start_s=0.0)
    assert res.start_timestamp["suspect"] and "start_timestamp_suspect" in res.warnings


def test_silent_recording_is_rejected(song_dir, tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(str(p), np.zeros(44100 * 5, dtype=np.float32), 44100)
    res = assess(song_dir / "twinkle.musicxml", p, RUBRIC, GroundTruthTranscriber([]))
    assert res.status == "rejected" and res.error_code == "recording_quality_low"
    assert res.scoring == {}


def test_undecodable_file_fails_cleanly(song_dir, tmp_path):
    p = tmp_path / "junk.m4a"
    p.write_bytes(b"junk")
    res = assess(song_dir / "twinkle.musicxml", p, RUBRIC, GroundTruthTranscriber([]))
    assert res.status == "failed" and res.error_code == "decode_failed"


def test_transposed_playing_is_scored_as_transposed(song_dir, twinkle, tmp_path):
    truth = simulate_performance(twinkle, PerfSpec(seed=2, transpose=2))
    audio = write_audio(render_notes(truth.notes), tmp_path / "t.wav")
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber(truth.notes))
    assert res.transposition["detected_semitones"] == 2 and res.transposition["applied"] == 2
    assert res.summary["correct"] == 78

    strict = RUBRIC.model_copy(deep=True)
    strict.options.allow_transposition = False
    res2 = assess(song_dir / "twinkle.musicxml", audio, strict, GroundTruthTranscriber(truth.notes))
    assert res2.confidence["needs_review"] and "performance_transposed" in res2.confidence["reasons"]


def test_no_notes_detected_goes_to_review(song_dir, take):
    _, audio = take
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber([]))
    assert "no_notes_detected" in res.warnings
    assert res.confidence["needs_review"]


def test_result_claiming_most_of_the_piece_is_wrong_goes_to_a_teacher(song_dir, twinkle, tmp_path):
    truth = simulate_performance(twinkle, PerfSpec(seed=3, n_missing=40))
    audio = write_audio(render_notes(truth.notes), tmp_path / "bad.wav")
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, GroundTruthTranscriber(truth.notes))
    assert res.confidence["needs_review"] and "many_error_claims" in res.confidence["reasons"]
    ok = simulate_performance(twinkle, PerfSpec(seed=3, n_missing=4))
    audio2 = write_audio(render_notes(ok.notes), tmp_path / "ok.wav")
    res2 = assess(song_dir / "twinkle.musicxml", audio2, RUBRIC, GroundTruthTranscriber(ok.notes))
    assert not res2.confidence["needs_review"]
