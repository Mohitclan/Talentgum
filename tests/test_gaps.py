"""Restarts, tempo changes, pedal use and a too-late start timestamp."""

from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from ras.core.align import match_notes
from ras.core.errors import analyze
from ras.core.rubric import Rubric, load_rubric, score_analysis
from ras.core.score.reference import load_reference
from ras.pipeline import assess
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import render_notes, write_audio
from ras.synth.songs import SONGS
from ras.types import PedalEvent, PerfNote, Transcription

RUBRIC = load_rubric(Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json")


# ---- restarts ---------------------------------------------------------------------------------


def test_restart_is_detected_and_not_counted_as_extra_notes(twinkle):
    truth = simulate_performance(twinkle, PerfSpec(seed=2, restarts=[(30, 0, 2.0)], n_wrong_pitch=2))
    a = analyze(twinkle, truth.notes, match_notes(twinkle, truth.notes))
    assert len(a.restarts) == 1 and a.restarts[0]["from_measure"] == 1
    assert a.n_replayed == len(truth.replayed) and a.n_extra == 0
    assert a.n_wrong_pitch == 2 and a.n_correct == twinkle.n_notes - 2
    assert a.summary()["restart_count"] == 1


def test_restarts_across_songs_keep_judgement_accurate(song_dir):
    correct = total = detected = runs = 0
    for sid in SONGS:
        ref = load_reference(song_dir / f"{sid}.musicxml")
        for seed in range(3):
            at = int(ref.n_notes * (0.4 + 0.1 * seed))
            t = simulate_performance(ref, PerfSpec(seed=seed, restarts=[(at, at - int(ref.n_notes * 0.25), 1.5)]))
            a = analyze(ref, t.notes, match_notes(ref, t.notes))
            correct += a.n_correct
            total += ref.n_notes
            detected += len(a.restarts) == 1
            runs += 1
    assert correct / total > 0.97
    assert detected / runs >= 0.8


def test_no_restart_reported_for_ordinary_extra_notes(twinkle):
    t = simulate_performance(twinkle, PerfSpec(seed=3, n_extra=3))
    a = analyze(twinkle, t.notes, match_notes(twinkle, t.notes))
    assert a.restarts == [] and a.n_extra == 3


def test_restart_is_reported_in_feedback_and_scored_by_rubric(twinkle):
    t = simulate_performance(twinkle, PerfSpec(seed=2, restarts=[(30, 0, 2.0)]))
    a = analyze(twinkle, t.notes, match_notes(twinkle, t.notes))
    from ras.core.feedback import make_feedback

    assert any("started again" in line for line in make_feedback(a, twinkle))
    rubric = Rubric.model_validate({"rubric_id": "t", "criteria": [
        {"id": "fluency", "weight": 1.0, "metric": "restart_count",
         "map": {"type": "step", "points": [[0, 100], [1, 60], [3, 0]]}}]})
    assert score_analysis(rubric, a).total == 60


# ---- tempo changes ----------------------------------------------------------------------------


def test_musicxml_tempo_change_is_applied(tmp_path):
    ref = load_reference(write_musicxml(SONGS["mary"], tmp_path / "m.musicxml", tempo_changes={5: 60}))
    assert ref.bpm == 120 and "1_tempo_change(s)_applied" in ref.warnings
    bar5 = ref.notes[ref.notes["measure"] == 5][0]
    assert bar5["onset_sec"] == pytest.approx(8.0)  # 16 beats at 120 bpm
    assert ref.notes[-1]["onset_sec"] == pytest.approx(8.0 + 12 * 1.0)  # then 12 beats at 60 bpm


def test_midi_tempo_change_keeps_beats_right(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    # 60 bpm from 2 s (beat 4) on: tick-based beats must stay 0..7
    pm._tick_scales.append((pm.time_to_tick(2.0), 60.0 / (60 * pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(30.0))
    inst = pretty_midi.Instrument(program=0)
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    for t in times:
        inst.notes.append(pretty_midi.Note(80, 60, t, t + 0.4))
    pm.instruments.append(inst)
    path = tmp_path / "tc.mid"
    pm.write(str(path))
    ref = load_reference(path)
    assert list(np.round(ref.notes["onset_beat"], 3)) == [0, 1, 2, 3, 4, 5, 6, 7]


# ---- pedal -------------------------------------------------------------------------------------


class _Fixed:
    name, sample_rate, version = "fixed", 44100, "fixed"

    def __init__(self, notes, pedal=(), supports_pedal=True):
        self.notes, self.pedal, self.supports_pedal = notes, list(pedal), supports_pedal

    def transcribe(self, audio, sr):
        return Transcription([PerfNote(n.pitch, n.onset, n.offset, n.velocity) for n in self.notes],
                             self.pedal, "fixed", "fixed", supports_pedal=self.supports_pedal)


def test_pedal_use_is_measured_only_when_the_model_reports_pedal(song_dir, twinkle, tmp_path):
    t = simulate_performance(twinkle, PerfSpec(seed=1))
    audio = write_audio(render_notes(t.notes), tmp_path / "p.wav")
    lo, hi = min(n.onset for n in t.notes), max(n.offset for n in t.notes)
    half = [PedalEvent(lo, lo + (hi - lo) / 2)]
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, _Fixed(t.notes, half))
    assert res.summary["pedal_events"] == 1 and res.summary["pedal_down_share"] == pytest.approx(0.5, abs=0.01)
    res2 = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, _Fixed(t.notes, supports_pedal=False))
    assert res2.summary["pedal_events"] is None and res2.summary["pedal_down_share"] is None


# ---- start timestamp ---------------------------------------------------------------------------


class _Cropping:
    """Behaves like a real transcriber: only hears notes inside the audio it is given."""

    name, sample_rate, version = "cropping", 44100, "cropping"

    def __init__(self, notes, full_samples):
        self.notes, self.full, self.calls = notes, full_samples, 0

    def transcribe(self, audio, sr):
        self.calls += 1
        cut = (self.full - len(audio)) / sr
        return Transcription([PerfNote(n.pitch, n.onset - cut, n.offset - cut, n.velocity)
                              for n in self.notes if n.onset - cut >= 0], transcriber="cropping")


def test_too_late_start_timestamp_falls_back_to_the_full_recording(song_dir, twinkle, tmp_path):
    t = simulate_performance(twinkle, PerfSpec(seed=1, start_offset_s=1.0))
    y = render_notes(t.notes)
    audio = write_audio(y, tmp_path / "late.wav")
    from ras.core.audio.decode import decode_audio

    n = len(decode_audio(audio, 44100))
    late = _Cropping(t.notes, n)
    res = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, late, start_s=9.0)  # playing began at 1 s
    assert late.calls == 2 and "start_timestamp_too_late_reassessed_on_full_recording" in res.warnings
    assert res.summary["correct"] == twinkle.n_notes and res.start_timestamp["suspect"]

    good = _Cropping(t.notes, n)
    res2 = assess(song_dir / "twinkle.musicxml", audio, RUBRIC, good, start_s=0.95)
    assert good.calls == 1 and res2.summary["correct"] == twinkle.n_notes
