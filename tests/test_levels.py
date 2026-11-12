"""TalentGum's level rules: right hand only and no octave judging for beginners, and a tempo score that never
drops to zero."""

from pathlib import Path

import pytest

from ras.core.rubric import load_rubric
from ras.core.score.reference import HAND_LEFT, HAND_RIGHT, load_reference, select_hands
from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.pipeline import assess
from ras.synth.render import render_notes, write_audio
from ras.types import PerfNote

RUBRICS = Path(__file__).resolve().parents[1] / "data" / "rubrics"

TWO_HANDS = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">{measures}</part></score-partwise>"""
MEASURE = """<measure number="{n}"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type>
</time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign>
<line>4</line></clef></attributes>{tempo}{rh}<backup><duration>4</duration></backup>{lh}</measure>"""
NOTE = ("<note><pitch><step>{s}</step><octave>{o}</octave></pitch><duration>{d}</duration><voice>{v}</voice>"
        "<type>{t}</type><staff>{st}</staff></note>")


def _rh(steps):
    return "".join(NOTE.format(s=s, o=4, d=1, v=1, t="quarter", st=1) for s in steps)


def _lh(step):
    return NOTE.format(s=step, o=3, d=4, v=5, t="whole", st=2)


@pytest.fixture
def two_hand_song(tmp_path):
    bars = [("EDCD", "C"), ("EEEE", "G"), ("DDED", "G"), ("CCCC", "C")]
    xml = TWO_HANDS.format(measures="".join(
        MEASURE.format(n=i + 1, tempo='<direction><sound tempo="120"/></direction>' if i == 0 else "",
                       rh=_rh(r), lh=_lh(lh)) for i, (r, lh) in enumerate(bars)))
    p = tmp_path / "song.musicxml"
    p.write_text(xml)
    return p


def test_staves_give_hands_and_one_hand_can_be_selected(two_hand_song):
    ref = load_reference(two_hand_song)
    assert set(ref.notes["hand"]) == {HAND_RIGHT, HAND_LEFT}
    right = select_hands(ref, "right")
    assert len(right.notes) == 16 and set(right.notes["hand"]) == {HAND_RIGHT}
    assert right.notes["id"][0] == "s0" and any("right_hand_only" in w for w in right.warnings)
    assert len(select_hands(ref, "left").notes) == 4 and select_hands(ref, "both") is ref


def _play(ref, tmp_path, octave_slip_at=None, with_left_hand=False, spb=0.5):
    notes = []
    for r in ref.notes:
        if r["hand"] == HAND_LEFT and not with_left_hand:
            continue
        pitch = int(r["pitch"]) + (12 if octave_slip_at is not None and r["hand"] == HAND_RIGHT
                                   and int(r["onset_beat"]) == octave_slip_at else 0)
        on = 1.0 + float(r["onset_beat"]) * spb
        notes.append(PerfNote(pitch, on, on + float(r["duration_beat"]) * spb * 0.9, 70))
    return notes, write_audio(render_notes(notes), tmp_path / "take.wav")


def test_beginner_is_scored_on_the_right_hand_and_octaves_are_not_judged(two_hand_song, tmp_path):
    ref = load_reference(two_hand_song)
    notes, audio = _play(ref, tmp_path, octave_slip_at=5)
    beginner = load_rubric(RUBRICS / "talentgum_beginner_v1.json")
    res = assess(ref, audio, beginner, GroundTruthTranscriber(notes))
    assert res.summary["score_notes"] == 16 and res.summary["missing"] == 0  # left hand not expected
    assert res.summary["wrong_octave"] == 0 and res.summary["correct"] == 16
    assert any("octave_slips_not_counted" in w for w in res.warnings)

    advanced = load_rubric(RUBRICS / "talentgum_advanced_v1.json")
    res = assess(ref, audio, advanced, GroundTruthTranscriber(notes))
    assert res.summary["score_notes"] == 20 and res.summary["missing"] == 4  # both hands expected
    assert res.summary["wrong_octave"] == 1


def test_tempo_targets_120_or_the_score_and_never_scores_zero(two_hand_song, tmp_path):
    ref = load_reference(two_hand_song)
    rubric = load_rubric(RUBRICS / "talentgum_beginner_v1.json")
    for spb, expect in ((0.5, 100.0), (1.5, 40.0)):  # 120 bpm: full marks; 40 bpm: slow but never zero
        notes, audio = _play(ref, tmp_path, spb=spb)
        tempo = assess(ref, audio, rubric, GroundTruthTranscriber(notes)).scoring["criteria"]["tempo"]
        assert tempo["metric"] == "tempo_deviation_pct" and tempo["score"] == pytest.approx(expect, abs=1)
