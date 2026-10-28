"""Cutting class recordings into song attempts: region finding, song identification, splitting attempts,
chunked transcription, the written record and clips, and the evaluation against labels."""

import json

import numpy as np
import pytest
import soundfile as sf

from ras.core.score.reference import load_reference
from ras.segment import (
    SegmentConfig,
    annotate_class,
    evaluate,
    find_clips,
    load_songs,
    merge_chunks,
    song_model,
    student_id,
    transcribe_long,
)
from ras.segment.align import local_align
from ras.segment.find import melody_of_notes
from ras.synth.performance import PerfSpec, simulate_performance
from ras.types import PerfNote, Transcription

SONG_IDS = ["twinkle", "mary", "ode_to_joy", "frere_jacques", "au_clair"]


@pytest.fixture(scope="module")
def refs(song_dir):
    return {s: load_reference(song_dir / f"{s}.musicxml") for s in SONG_IDS}


@pytest.fixture(scope="module")
def models(refs):
    return [song_model(s, r) for s, r in refs.items()]


def perform(ref, start_s, transpose=0, right_hand_only=False, **spec):
    """The notes of one attempt, starting at start_s on the class clock."""
    truth = simulate_performance(ref, PerfSpec(seed=spec.pop("seed", 1), transpose=transpose, **spec))
    notes = [PerfNote(n.pitch, n.onset - 1.0 + start_s, n.offset - 1.0 + start_s, n.velocity) for n in truth.notes]
    if right_hand_only:  # keep the highest note of each onset
        notes = [n for n in notes if n.pitch == max(m.pitch for m in notes if abs(m.onset - n.onset) < 0.07)]
    return notes


def clutter(duration_s, seed=0, n_isolated=60, burst_every_s=30):
    """Transcription noise from talking: isolated notes and short sparse bursts."""
    rng = np.random.default_rng(seed)
    times = rng.uniform(0, duration_s, n_isolated)
    notes = [PerfNote(int(rng.integers(40, 90)), float(t), float(t) + 0.2, 40) for t in times]
    for t0 in np.arange(10, duration_s, burst_every_s):
        notes += [PerfNote(int(rng.integers(45, 85)), float(t0 + rng.uniform(0, 1.5)), float(t0 + 1.8), 35)
                  for _ in range(4)]
    return notes


def test_local_alignment_ignores_junk_and_absorbs_a_wrong_step():
    song = np.array([0, 2, -2, 5, -1, -2, 0, 2, -2, 5, -1, -2, -3])
    heard = np.array([7, 7, 1, 0, 2, -2, 5, 9, -1, -2, 0, 2, -2, 5, -1, -2, -3, 4, 4])
    al = local_align(heard, song)
    assert al.matches == 13 and al.precision > 0.9 and (al.a0, al.b0) == (3, 0)


def test_two_songs_in_one_class_are_found_and_named(refs, models):
    notes = (perform(refs["twinkle"], 300) + perform(refs["ode_to_joy"], 900, seed=2) + clutter(1500))
    out = find_clips(notes, models)
    assert [c["song_id"] for c in out["clips"]] == ["twinkle", "ode_to_joy"]
    assert out["clips"][0]["start_s"] == pytest.approx(300, abs=1.0)
    assert out["clips"][1]["start_s"] == pytest.approx(900, abs=1.0)
    assert all(c["coverage"] == "full" for c in out["clips"])
    assert out["clips"][0]["status"] == "auto" and out["clips"][0]["song_bars"][0] == 1
    # a noise burst fell inside the second attempt and stepwise songs look alike: a teacher should confirm it
    assert out["clips"][1]["match_score"] < out["clips"][0]["match_score"] or out["clips"][1]["status"] == "auto"


def test_talking_noise_alone_gives_no_clips(models):
    out = find_clips(clutter(2700, seed=3, n_isolated=200), models)
    assert out["clips"] == []


def test_restart_gives_a_partial_clip_then_a_full_one(refs, models):
    ref = refs["twinkle"]
    notes = perform(ref, 100, restarts=[(26, 0, 4.0)])  # stops after about a dozen melody notes, starts again
    clips = find_clips(notes, models)["clips"]
    assert [c["coverage"] for c in clips] == ["partial", "full"]
    assert all(c["song_id"] == "twinkle" for c in clips)
    assert clips[0]["end_s"] < clips[1]["start_s"]


def test_same_song_twice_with_a_pause_is_two_clips(refs, models):
    notes = perform(refs["mary"], 100) + perform(refs["mary"], 220, seed=4)
    clips = find_clips(notes, models)["clips"]
    assert len(clips) == 2 and {c["song_id"] for c in clips} == {"mary"}


def test_transposed_wrong_notes_and_one_hand_still_match(refs, models):
    sloppy = perform(refs["twinkle"], 50, transpose=5, n_wrong_pitch=3, n_missing=2, seed=5)
    right_hand = perform(refs["twinkle"], 400, right_hand_only=True, seed=6)
    clips = find_clips(sloppy + right_hand, models)["clips"]
    assert [c["song_id"] for c in clips] == ["twinkle", "twinkle"]
    assert clips[0]["match_score"] < 1.0 and clips[1]["match_score"] > 0.95


def test_short_fragment_is_ignored(refs, models):
    fragment = [n for n in perform(refs["twinkle"], 100)][:6]
    assert find_clips(fragment, models)["clips"] == []


def test_ambiguous_match_goes_to_review(refs):
    twin = song_model("twinkle", refs["twinkle"])
    copy = song_model("twinkle_copy", refs["twinkle"])  # two songs with the same melody cannot be told apart
    clip = find_clips(perform(refs["twinkle"], 10), [twin, copy])["clips"][0]
    assert clip["status"] == "needs_review" and any(r.startswith("could_be_") for r in clip["review_reasons"])


def test_left_hand_alone_is_not_recognised(refs, models):
    bass = perform(refs["twinkle"], 10)
    top = {round(n.onset, 1) for n in bass if n.pitch == max(m.pitch for m in bass if abs(m.onset - n.onset) < 0.07)}
    lower = [n for n in bass if round(n.onset, 1) not in top or n.pitch < 60]
    clips = find_clips(lower, models)["clips"]
    assert all(c["coverage"] == "partial" or c["status"] == "needs_review" for c in clips) or not clips


def test_melody_takes_the_top_note_of_each_chord():
    notes = [PerfNote(48, 1.0, 2.0), PerfNote(64, 1.01, 2.0), PerfNote(60, 1.02, 2.0), PerfNote(65, 2.0, 3.0)]
    m = melody_of_notes(notes)
    assert list(m.pitch) == [64, 65] and list(m.onset) == [1.0, 2.0]


def test_chunks_are_merged_without_losing_or_doubling_notes():
    notes = [PerfNote(60, t, t + 0.3, 80) for t in np.arange(0.5, 100, 0.7)]
    overlap = 10.0
    parts = [(0.0, 40.0, [n for n in notes if n.onset < 40]),
             (30.0, 70.0, [n for n in notes if 30 <= n.onset < 70]),
             (60.0, 100.0, [n for n in notes if n.onset >= 60])]
    merged = merge_chunks(parts, overlap)
    assert [n.onset for n in merged] == [n.onset for n in notes]


class TimelineTranscriber:
    """Hears a known timeline: each call returns the notes inside the chunk it is handed."""

    name, version, sample_rate = "timeline", "timeline-1", 44100

    def __init__(self, notes, chunk_s, overlap_s):
        self.notes, self.step, self.calls = notes, chunk_s - overlap_s, 0

    def transcribe(self, audio, sr):
        start = self.calls * self.step
        self.calls += 1
        end = start + len(audio) / sr
        return Transcription(notes=[PerfNote(n.pitch, n.onset - start, n.offset - start, n.velocity)
                                    for n in self.notes if start <= n.onset < end])


def test_long_audio_is_transcribed_in_chunks_and_stitched(refs):
    notes = perform(refs["twinkle"], 55)  # sits across a chunk boundary
    audio = np.zeros(int(130 * 44100), dtype=np.float32)
    tr = TimelineTranscriber(notes, chunk_s=60, overlap_s=10)
    got = transcribe_long(audio, 44100, tr, chunk_s=60, overlap_s=10)
    assert tr.calls == 3
    assert [(n.pitch, round(n.onset, 3)) for n in got] == [(n.pitch, round(n.onset, 3)) for n in notes]


def test_annotate_class_writes_record_transcript_and_clips(tmp_path, refs, models):
    notes = perform(refs["twinkle"], 20) + perform(refs["mary"], 90, seed=3)
    wav = tmp_path / "student folder" / "class_1.wav"
    wav.parent.mkdir()
    sf.write(wav, np.zeros(int(150 * 44100), dtype=np.float32), 44100)
    tr = TimelineTranscriber(notes, chunk_s=600, overlap_s=10)
    out = tmp_path / "out"
    rec = annotate_class(wav, models, tr, out, student="student folder")
    sid = student_id("student folder")
    assert rec["student_id"] == sid and "student folder" not in json.dumps(rec)
    assert [c["song_id"] for c in rec["clips"]] == ["twinkle", "mary"]
    assert (out / sid / "class_1.json").is_file() and (out / sid / "class_1.notes.json").is_file()
    clip = rec["clips"][0]
    data, sr = sf.read(out / clip["file"])
    assert sr == 44100 and len(data) / sr == pytest.approx(clip["clip_end_s"] - clip["clip_start_s"], abs=0.01)
    assert clip["assess_start_s"] == pytest.approx(2.0, abs=0.1)
    # a second run reuses the transcript instead of transcribing again
    tr2 = TimelineTranscriber([], chunk_s=600, overlap_s=10)
    again = annotate_class(wav, models, tr2, out, student="student folder", reuse=True, cut_clips=False)
    assert tr2.calls == 0 and [c["song_id"] for c in again["clips"]] == ["twinkle", "mary"]


def test_clips_longer_than_five_minutes_are_cut_and_flagged(tmp_path, refs, models):
    cfg = SegmentConfig(max_clip_s=10.0)
    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(int(100 * 44100), dtype=np.float32), 44100)
    tr = TimelineTranscriber(perform(refs["twinkle"], 10), chunk_s=600, overlap_s=10)
    clip = annotate_class(wav, models, tr, tmp_path / "o", cfg=cfg, cut_clips=False)["clips"][0]
    assert clip["clip_end_s"] - clip["clip_start_s"] == pytest.approx(10.0) and clip["status"] == "needs_review"


def test_load_songs_from_a_folder_and_duplicate_ids(song_dir, tmp_path):
    songs = load_songs(song_dir)
    assert {"twinkle", "mary", "ode_to_joy"} <= {s.song_id for s in songs}
    with pytest.raises(ValueError):
        load_songs([song_dir / "twinkle.musicxml", (("twinkle"), song_dir / "mary.musicxml")])


def test_evaluation_against_labels(refs, models):
    notes = perform(refs["twinkle"], 100) + perform(refs["mary"], 400, seed=2) + perform(refs["au_clair"], 700, seed=3)
    rec = find_clips(notes, models)
    rec["duration_s"] = 3600
    labels = [{"song_id": "twinkle", "start_s": 100, "end_s": rec["clips"][0]["end_s"]},
              {"song_id": "ode_to_joy", "start_s": 400, "end_s": rec["clips"][1]["end_s"]},  # labelled as another song
              {"song_id": "mary", "start_s": 2000, "end_s": 2040}]  # a performance the splitter did not find
    ev = evaluate(rec, labels)
    assert (ev["found"], ev["wrong_song"], ev["missed"], ev["false_clips"]) == (1, 1, 1, 1)
    assert ev["false_clips_per_hour"] == 1.0 and ev["median_start_error_s"] < 1.0
