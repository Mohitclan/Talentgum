"""Melody isolation and the reference-fit gate, with notes patterned on real class recordings."""
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ras.core.melody import isolate_melody, is_single_line, reference_fit
from ras.core.rubric import load_rubric
from ras.core.score.reference import load_reference
from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.pipeline import assess
from ras.types import PerfNote

RUBRICS = Path(__file__).resolve().parents[1] / "data" / "rubrics"
RUBRIC = load_rubric(RUBRICS / "talentgum_criteria_v1.json")
BEAT = 0.5


def melody_of(ref, shift=0):
    return [PerfNote(int(p) + shift, float(b) * BEAT + 1.0, float(b) * BEAT + 1.0 + 0.3, 70)
            for p, b in zip(ref.notes["pitch"], ref.notes["onset_beat"])]


def with_harmonics(notes):
    """The class-call pattern: a ghost at the 3rd or 5th harmonic, as loud as the note or louder."""
    out = []
    for i, n in enumerate(notes):
        out.append(n)
        ghost = n.pitch + (19 if i % 2 else 28)
        out.append(PerfNote(ghost, n.onset + 0.01, n.onset + 0.05, n.velocity + 10))
    return out


def with_chords(notes, every=4, shift=0):
    """C-E-G sustained under the melody, as on the class 3 recording."""
    out = list(notes)
    for n in notes[::every]:
        out += [PerfNote(p + shift, n.onset - 0.03, n.onset + 1.5, 62) for p in (48, 52, 55)]
    return out


def write_silence(path):
    sf.write(path, np.random.default_rng(0).normal(0, 0.01, 44100 * 40), 44100)
    return path


@pytest.fixture(scope="module")
def mary(song_dir):
    return load_reference(song_dir / "mary.musicxml")


def test_harmonics_are_removed_and_never_the_melody(mary):
    played = melody_of(mary)
    res = isolate_melody(mary, with_harmonics(played))
    assert [n.pitch for n in res.notes] == [n.pitch for n in played]
    assert res.harmonics_removed == len(played) and res.shift == 0


def test_melody_survives_being_a_harmonic_of_the_chord_below(mary):
    # keeping "the lowest note" would delete E5, the 5th harmonic of C4
    played = melody_of(mary, shift=12)
    res = isolate_melody(mary, with_chords(played))
    assert [n.pitch for n in res.notes] == [n.pitch for n in played]
    assert res.shift == 12 and res.accompaniment_heard


def test_octave_shift_found_when_chords_sit_in_the_written_octave(mary):
    # the transposition histogram alone is fooled by chord tones that match the written notes
    played = melody_of(mary, shift=12)
    assert isolate_melody(mary, with_chords(played, every=2)).shift == 12


def test_a_real_wrong_note_is_kept(mary):
    played = melody_of(mary)
    played[5] = PerfNote(played[5].pitch + 1, played[5].onset, played[5].offset, 70)
    res = isolate_melody(mary, with_harmonics(played))
    assert res.notes[5].pitch == played[5].pitch and len(res.notes) == len(played)


def test_songs_with_chords_are_left_alone(song_dir):
    from ras.synth.songs import SONGS

    chordal = [s for s in SONGS if not is_single_line(load_reference(song_dir / f"{s}.musicxml"))]
    assert chordal
    ref = load_reference(song_dir / f"{chordal[0]}.musicxml")
    notes = with_harmonics(melody_of(ref))
    res = isolate_melody(ref, notes)
    assert not res.applied and res.notes == notes


def test_clean_playing_is_unchanged_and_scored(song_dir, mary, tmp_path):
    played = melody_of(mary)
    res = assess(song_dir / "mary.musicxml", write_silence(tmp_path / "a.wav"), RUBRIC,
                 GroundTruthTranscriber(played))
    assert res.summary["correct"] == len(played) and res.summary["extra"] == 0
    assert res.summary["reference_fit_pct"] == 100.0
    assert res.scoring["total_score"] is not None and "withheld" not in res.scoring


def test_stopping_early_is_not_a_different_song(song_dir, mary, tmp_path):
    played = melody_of(mary)[:10]
    res = assess(song_dir / "mary.musicxml", write_silence(tmp_path / "a.wav"), RUBRIC,
                 GroundTruthTranscriber(played))
    assert res.summary["reference_fit_pct"] >= 80.0 and res.scoring["total_score"] is not None


def test_a_different_song_gets_no_score_and_no_note_feedback(song_dir, tmp_path):
    played = melody_of(load_reference(song_dir / "frere_jacques.musicxml"))
    res = assess(song_dir / "mary.musicxml", write_silence(tmp_path / "a.wav"), RUBRIC,
                 GroundTruthTranscriber(played))
    assert res.scoring["total_score"] is None and res.scoring["withheld"]
    assert res.confidence["needs_review"] and "reference_mismatch" in res.confidence["reasons"]
    assert not any("bar" in line for line in res.feedback)
    assert res.scoring["criteria"]  # still there for the teacher


def test_fit_needs_more_than_a_few_chance_notes():
    class R:
        def __init__(self, i):
            self.ref_index, self.status = i, "correct"

    assert reference_fit([R(0), R(1), R(2)]) == 0.0
    assert reference_fit([R(i) for i in range(0, 20, 2)]) == pytest.approx(10 / 19)


def test_downloaded_model_files_are_readable_by_other_users(tmp_path):
    import ras.models as M

    src = tmp_path / "src.bin"
    src.write_bytes(b"weights")
    f = M.ModelFile(url=src.as_uri(), sha256=M._sha256(src), name="w.pth")
    dest = tmp_path / "out" / "w.pth"
    M._fetch(f, dest)
    assert stat.S_IMODE(dest.stat().st_mode) & 0o044 == 0o044
