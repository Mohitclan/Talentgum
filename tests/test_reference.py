import pretty_midi

from ras.core.score.reference import load_reference


def test_musicxml_notes_measures_tempo(twinkle):
    assert twinkle.n_notes == 78  # 42 melody notes + 12 chords of 3
    assert twinkle.n_measures == 12
    assert twinkle.bpm == 100
    first = twinkle.notes[twinkle.notes["onset_beat"] == 0]
    assert sorted(first["pitch"]) == [48, 52, 55, 60]
    assert set(twinkle.notes["beat_in_measure"][twinkle.notes["measure"] == 1]) == {1.0, 2.0, 3.0, 4.0}


def test_dotted_rhythm_and_beats(song_dir):
    ode = load_reference(song_dir / "ode_to_joy.musicxml")
    assert ode.n_notes == 30
    assert 1.5 in set(ode.notes["duration_beat"])
    assert ode.notes["onset_sec"].max() > 0


def test_repeats_are_unfolded_by_default(song_dir):
    plain = load_reference(song_dir / "mary.musicxml")
    unfolded = load_reference(song_dir / "mary_repeat.musicxml")
    kept = load_reference(song_dir / "mary_repeat.musicxml", unfold_repeats=False)
    assert unfolded.n_notes == 2 * plain.n_notes
    assert kept.n_notes == plain.n_notes


def test_midi_reference(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i, p in enumerate([60, 62, 64, 65]):
        inst.notes.append(pretty_midi.Note(80, p, i * 0.5, i * 0.5 + 0.4))
    pm.instruments.append(inst)
    path = tmp_path / "ref.mid"
    pm.write(str(path))
    ref = load_reference(path)
    assert ref.n_notes == 4
    assert ref.bpm == 120
    assert list(ref.notes["pitch"]) == [60, 62, 64, 65]
    assert list(ref.notes["onset_beat"]) == [0, 1, 2, 3]


def test_sha256_changes_with_content(song_dir):
    a = load_reference(song_dir / "mary.musicxml")
    b = load_reference(song_dir / "twinkle.musicxml")
    assert a.sha256 != b.sha256 and len(a.sha256) == 64
