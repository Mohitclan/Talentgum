import numpy as np

from ras.core.align import match_notes
from ras.core.errors import analyze
from ras.core.verify import OnsetEvidence, mark_uncertain_missing
from ras.synth.render import render_notes
from ras.types import PerfNote

SR = 44100
MELODY = [60, 64, 67, 72, 62, 65, 69, 71]


def _melody_notes():
    return [PerfNote(p, 0.5 + 0.7 * i, 0.5 + 0.7 * i + 0.55, 80) for i, p in enumerate(MELODY)]


def test_onset_evidence_separates_played_from_absent_notes():
    notes = _melody_notes()
    ev = OnsetEvidence(render_notes(notes), SR)
    played = ev.evidence_db(64, notes[1].onset, 0.15)
    absent = ev.evidence_db(66, notes[1].onset, 0.15)  # F#4 is never played
    assert played is not None and absent is not None
    assert played > 2.5 > absent


def test_low_pitches_are_not_judged():
    ev = OnsetEvidence(render_notes(_melody_notes()), SR)
    assert ev.jump_db(36, 1.0) is None


def _score(song_dir, tmp_path):
    """A reference whose notes are MELODY, as a one-hand MIDI file."""
    import pretty_midi

    from ras.core.score.reference import load_reference

    pm = pretty_midi.PrettyMIDI(initial_tempo=85.7)
    inst = pretty_midi.Instrument(program=0)
    for n in _melody_notes():
        inst.notes.append(pretty_midi.Note(80, n.pitch, n.onset, n.offset))
    pm.instruments.append(inst)
    pm.write(str(tmp_path / "m.mid"))
    return load_reference(tmp_path / "m.mid")


def test_dropped_note_with_audio_evidence_becomes_uncertain(song_dir, tmp_path):
    ref = _score(song_dir, tmp_path)
    played = _melody_notes()
    ev = OnsetEvidence(render_notes(played), SR)
    seen = [n for n in played if n.pitch != 69]  # the transcriber dropped A4, but it was played
    a = analyze(ref, seen, match_notes(ref, seen))
    assert a.n_missing == 1
    assert mark_uncertain_missing(a, ev, threshold_db=2.5) == 1
    assert (a.n_missing, a.n_uncertain, a.n_scored) == (0, 1, ref.n_notes - 1)


def test_truly_absent_note_stays_missing(song_dir, tmp_path):
    ref = _score(song_dir, tmp_path)
    played = [n for n in _melody_notes() if n.pitch != 69]  # really not played, not in the audio
    ev = OnsetEvidence(render_notes(played), SR)
    a = analyze(ref, played, match_notes(ref, played))
    assert a.n_missing == 1
    assert mark_uncertain_missing(a, ev, threshold_db=2.5) == 0
    assert a.n_missing == 1 and a.n_uncertain == 0


def test_uncertain_notes_are_not_held_against_the_student(song_dir, tmp_path):
    from ras.core.rubric import Rubric, score_analysis

    ref = _score(song_dir, tmp_path)
    played = _melody_notes()
    seen = [n for n in played if n.pitch != 69]
    a = analyze(ref, seen, match_notes(ref, seen))
    rubric = Rubric.model_validate({"rubric_id": "t", "criteria": [
        {"id": "acc", "weight": 1.0, "metric": "note_accuracy_pct"}]})
    before = score_analysis(rubric, a).criteria["acc"]["value"]
    a.mark_uncertain([i for i, r in enumerate(a.results) if r.status == "missing"])
    after = score_analysis(rubric, a).criteria["acc"]["value"]
    assert after == 100.0 and before < after


def test_parangonar_matcher_still_works(song_dir, twinkle):
    from ras.synth.performance import PerfSpec, simulate_performance

    truth = simulate_performance(twinkle, PerfSpec(seed=1, n_wrong_pitch=2))
    al = match_notes(twinkle, truth.notes, method="parangonar")
    assert al.method == "parangonar-automatic"
    assert len(al.pairs) >= 0.9 * twinkle.n_notes
    assert np.all(np.diff([r for r, _ in al.pairs]) >= 0)
