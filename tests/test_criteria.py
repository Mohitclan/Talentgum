"""The client's criteria beyond right/wrong notes: score marks, order, chords, rhythm, articulation,
dynamics, pedal, second opinions, teacher profiles, rubric fallbacks, calibration, model plumbing."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ras.core import second_opinion
from ras.core.align import match_notes
from ras.core.errors import analyze
from ras.core.expression import measure
from ras.core.profile import build_profile, compare
from ras.core.rubric import Rubric, load_rubric, score_analysis
from ras.core.rubric.engine import models_needed
from ras.core.score.reference import ARTIC_ACCENT, ARTIC_LEGATO, ARTIC_STACCATO, load_reference
from ras.types import PedalEvent, PerfNote

RUBRICS = Path(__file__).resolve().parents[1] / "data" / "rubrics"

# 3 bars at 120 bpm (0.5 s per quarter): staccato C D, plain E F under p; slurred G A B C under f with
# the pedal; then a C-E-G chord and a final C.
SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>
<part id="P1">
<measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome>
</direction-type><sound tempo="120"/></direction>
<direction><direction-type><dynamics><p/></dynamics></direction-type></direction>
{n1}{n2}{n3}{n4}</measure>
<measure number="2">
<direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
<direction><direction-type><pedal type="start"/></direction-type></direction>
{n5}{n6}{n7}{n8}
<direction><direction-type><pedal type="stop"/></direction-type></direction></measure>
<measure number="3">{c1}{c2}{c3}<note><rest/><duration>2</duration><voice>1</voice></note>{n9}</measure>
</part></score-partwise>"""


def _note(step, octave, notations="", dur=1, chord=False):
    return (f"<note>{'<chord/>' if chord else ''}<pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            f"<duration>{dur}</duration><voice>1</voice>"
            f"{'<notations>' + notations + '</notations>' if notations else ''}</note>")


STAC = "<articulations><staccato/></articulations>"


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    xml = SCORE.format(
        n1=_note("C", 4, STAC), n2=_note("D", 4, STAC), n3=_note("E", 4), n4=_note("F", 4),
        n5=_note("G", 4, '<slur type="start" number="1"/><articulations><accent/></articulations>'),
        n6=_note("A", 4), n7=_note("B", 4), n8=_note("C", 5, '<slur type="stop" number="1"/>'),
        c1=_note("C", 4), c2=_note("E", 4, chord=True), c3=_note("G", 4, chord=True), n9=_note("C", 4))
    p = tmp_path_factory.mktemp("marked") / "marked.musicxml"
    p.write_text(xml)
    return load_reference(p)


def _perform(ref, *, staccato_len=0.2, legato_gap=0.0, p_vel=40, f_vel=90, swap=False, chord_drop=None,
             chord_extra=None, accent=25):
    """One note per score note at 0.5 s per beat, starting at 1.0 s."""
    notes = []
    for i, row in enumerate(ref.notes):
        on = 1.0 + 0.5 * float(row["onset_beat"])
        dur = 0.45 * float(row["duration_beat"]) / 1.0
        if row["artic"] & ARTIC_STACCATO:
            dur = staccato_len
        if row["artic"] & ARTIC_LEGATO:
            dur = 0.5 - legato_gap
        vel = p_vel if row["onset_beat"] < 4 else f_vel
        if row["artic"] & ARTIC_ACCENT:
            vel += accent
        if row["onset_beat"] >= 8:
            vel = 70
        notes.append(PerfNote(int(row["pitch"]), on, on + dur, int(vel) + (i % 3)))
    if swap:  # E and F played in the wrong order
        e, f = notes[2], notes[3]
        notes[2], notes[3] = PerfNote(f.pitch, e.onset, e.offset, e.velocity), PerfNote(e.pitch, f.onset, f.offset,
                                                                                        f.velocity)
    if chord_drop is not None:
        notes = [n for n in notes if not (n.pitch == chord_drop and abs(n.onset - 5.0) < 0.01)]
    if chord_extra is not None:
        notes.append(PerfNote(chord_extra, 5.01, 5.4, 70))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def _analyse(ref, perf, pedal=None, levels=None):
    a = analyze(ref, perf, match_notes(ref, perf))
    measure(ref, perf, a, pedal=pedal, model_levels=levels)
    return a


def test_score_marks_are_read(marked):
    n = marked.notes
    assert list(n["artic"][:2] & ARTIC_STACCATO) == [ARTIC_STACCATO] * 2
    slurred = [int(p) for p, f in zip(n["pitch"], n["artic"]) if f & ARTIC_LEGATO]
    assert slurred == [67, 69, 71]  # the slur's last note (C5) is released
    assert n["artic"][4] & ARTIC_ACCENT
    assert list(n["dyn"][:4]) == [2.0] * 4 and n["dyn"][4] == 4.0
    assert marked.pedal_spans == [(4.0, 8.0)]
    assert marked.has_articulation_marks and marked.has_dynamics_marks


def test_good_performance_follows_every_mark(marked):
    pedal = [PedalEvent(3.0, 5.0)]  # beats 4..8 at 0.5 s per beat, starting at 1.0 s
    a = _analyse(marked, _perform(marked), pedal=pedal)
    x = a.extras
    assert x["articulation"]["by_type"]["staccato"] == {"ok": 2, "checked": 2}
    assert x["articulation"]["by_type"]["legato"]["ok"] == x["articulation"]["by_type"]["legato"]["checked"] == 3
    assert x["articulation"]["by_type"]["accent"] == {"ok": 1, "checked": 1}
    assert x["dynamics"]["velocity"]["ok"] == x["dynamics"]["velocity"]["comparisons"] >= 1
    assert x["pedal"]["agreement"] > 0.9
    assert x["order"]["inversions"] == x["order"]["swaps"] == 0
    assert x["chords"]["total"] == 1 and x["chords"]["chords"][0]["correct"]
    assert x["rhythm"]["ioi_within"] == x["rhythm"]["ioi_checked"] and not x["rhythm"]["hesitations"]


def test_bad_performance_is_caught_per_criterion(marked):
    # E dropped from the chord and a far-away key added (a near one would read as a wrong chord tone)
    perf = _perform(marked, staccato_len=0.5, legato_gap=0.25, p_vel=90, f_vel=40, swap=True, chord_drop=64,
                    chord_extra=80, accent=0)
    a = _analyse(marked, perf, pedal=[])
    x = a.extras
    assert x["articulation"]["by_type"]["staccato"]["ok"] == 0
    assert x["articulation"]["by_type"]["legato"]["ok"] == 0
    assert x["articulation"]["by_type"]["accent"]["ok"] == 0
    assert x["dynamics"]["velocity"]["ok"] == 0
    assert x["pedal"]["coverage"] == 0
    assert x["order"]["inversions"] + x["order"]["swaps"] >= 1
    ch = x["chords"]["chords"][0]
    assert ch["missing"] == 1 and ch["extra"] == 1 and not ch["correct"]


def test_talentgum_rubric_scores_every_criterion(marked):
    rub = load_rubric(RUBRICS / "talentgum_criteria_v1.json")
    assert len(rub.criteria) == 22 and abs(sum(c.weight for c in rub.criteria) - 1) < 1e-9
    good = score_analysis(rub, _analyse(marked, _perform(marked), pedal=[PedalEvent(3.0, 5.0)]))
    bad = score_analysis(rub, _analyse(marked, _perform(marked, staccato_len=0.5, legato_gap=0.25, p_vel=90,
                                                       f_vel=40, swap=True, accent=0, chord_drop=64), pedal=[]))
    for cid in ("articulation", "dynamics", "sustain_pedal", "note_order", "chord_correctness"):
        assert good.criteria[cid]["status"] == "assessed", cid
        assert good.criteria[cid]["score"] > bad.criteria[cid]["score"], cid
    # no teacher profile yet: note duration falls back to the score comparison
    assert good.criteria["note_duration"]["metric"] == "duration_within_pct"
    assert good.total > bad.total


def test_rhythm_and_hesitation(marked):
    perf = _perform(marked)
    late = [PerfNote(n.pitch, n.onset + (0.6 if n.onset >= 2.99 else 0), n.offset + (0.6 if n.onset >= 2.99 else 0),
                     n.velocity) for n in perf]  # a 0.6 s hesitation before bar 2
    x = _analyse(marked, late).extras["rhythm"]
    assert len(x["hesitations"]) == 1 and x["ioi_within"] < x["ioi_checked"]


def test_flat_keyboard_uses_the_audio_dynamics_model(marked):
    perf = [PerfNote(n.pitch, n.onset, n.offset, 64) for n in _perform(marked)]  # no velocity sensitivity
    a = _analyse(marked, perf)
    rub = load_rubric(RUBRICS / "talentgum_criteria_v1.json")
    s = score_analysis(rub, a)
    assert a.velocity_flat and s.criteria["dynamics"]["status"] == "not_assessed"
    assert "dynamics" in models_needed(rub, velocity_flat=True)
    levels = {i: (2.0 if n.onset < 3.0 else 4.0) for i, n in enumerate(perf)}  # what the model would hear
    a2 = _analyse(marked, perf, levels=levels)
    s2 = score_analysis(rub, a2)
    assert s2.criteria["dynamics"]["status"] == "assessed" and s2.criteria["dynamics"]["score"] == 100


def test_not_assessed_reasons(twinkle):
    from ras.synth.performance import PerfSpec, simulate_performance

    perf = simulate_performance(twinkle, PerfSpec(seed=1)).notes
    a = _analyse(twinkle, perf)  # no pedal detection, no marks in the synthetic songs
    s = score_analysis(load_rubric(RUBRICS / "talentgum_criteria_v1.json"), a)
    assert s.criteria["sustain_pedal"]["reason"] == "transcriber_does_not_detect_pedal"
    assert s.criteria["articulation"]["reason"] == "no_articulation_marks_in_score"  # and no teacher fallback
    assert s.criteria["dynamics"]["status"] == "not_assessed"


def test_second_opinion_only_takes_claims_back(twinkle):
    from ras.synth.performance import PerfSpec, simulate_performance

    truth = simulate_performance(twinkle, PerfSpec(seed=2))
    played = truth.notes
    primary = [n for i, n in enumerate(played) if i not in (5, 9)]  # the main model dropped two notes
    primary.append(PerfNote(90, 3.0, 3.2, 60))  # and imagined one
    a = analyze(twinkle, primary, match_notes(twinkle, primary))
    assert a.n_missing == 2 and a.n_extra == 1
    out = second_opinion.apply(a, primary, played, "hft")
    assert out["claims_retracted"] == 2 and out["extras_unconfirmed"] == 1
    assert a.n_missing == 0 and a.n_uncertain == 2 and a.n_extra == 0 and a.n_unconfirmed_extra == 1

    a = analyze(twinkle, primary, match_notes(twinkle, primary))
    second_opinion.apply(a, primary, played, "hft", credit_missing=True)
    assert a.n_missing == 0 and a.n_uncertain == 0 and a.n_correct == len(twinkle.notes)


def test_teacher_profile_round_trip(marked):
    teachers = [_analyse(marked, _perform(marked, p_vel=40 + k, f_vel=90 - k)) for k in range(3)]
    prof = build_profile([t.scoring_inputs() for t in teachers], marked.sha256)
    assert prof["recordings"] == 3 and prof["tempo_bpm"][0] == pytest.approx(120, abs=2)
    like = compare(_analyse(marked, _perform(marked)).extras, prof)
    unlike = compare(_analyse(marked, _perform(marked, staccato_len=0.5, legato_gap=0.3, p_vel=90, f_vel=40)).extras,
                     prof)
    assert like["duration_within_pct"] > unlike["duration_within_pct"]
    assert like["dynamics_corr"] > 0.8 > unlike["dynamics_corr"]


def test_calibration_fits_maps_to_teacher_grades(tmp_path, twinkle):
    from ras.synth.performance import PerfSpec, simulate_performance

    rub = Rubric.model_validate({"rubric_id": "t", "criteria": [
        {"id": "acc", "weight": 1.0, "metric": "note_accuracy_pct", "map": {"from": [0, 100], "to": [0, 100]}}]})
    base = _analyse(twinkle, simulate_performance(twinkle, PerfSpec(seed=1)).notes).scoring_inputs()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(30):
        acc = float(rng.uniform(50, 100))
        res = {"scoring_inputs": {**base, "n_ref": 100, "n_correct": int(acc), "n_uncertain": 0}}
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(res))
        rows.append({"result": str(p), "acc": round(max(0.0, (acc - 50) * 2), 1)})  # teachers are stricter
    grades = tmp_path / "grades.csv"
    with grades.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["result", "acc"])
        w.writeheader()
        w.writerows(rows)
    from ras.eval.calibrate import calibrate

    new, report = calibrate(rub, grades)
    assert report["criteria"]["acc"]["mae_after"] < report["criteria"]["acc"]["mae_before"]
    assert new["criteria"][0]["map"]["type"] == "piecewise" and new["version"] == 2


def test_safe_loader_refuses_code(tmp_path):
    import os

    from ras.models.safe_load import UnsafeCheckpoint, safe_pickle_load

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(UnsafeCheckpoint):
        safe_pickle_load(p)


def test_missing_model_makes_criterion_not_assessed_not_a_crash(marked, tmp_path):
    from ras.models import is_available
    from ras.models.hub import ModelHub

    assert not is_available("dynamics")  # conftest points RAS_MODELS_DIR at an empty folder
    hub = ModelHub()
    assert hub.get("dynamics") is None and hub.errors["dynamics"] == "not_downloaded"


def test_criteria_catalogue_matches_rubric():
    from ras.core.criteria import CRITERIA
    from ras.core.rubric.engine import METRICS

    rub = load_rubric(RUBRICS / "talentgum_criteria_v1.json")
    assert [c.id for c in CRITERIA] == [c.id for c in rub.criteria]
    for c in CRITERIA:
        assert c.metric in METRICS and (c.fallback is None or c.fallback in METRICS)


def test_feedback_mentions_expression_issues(marked):
    from ras.core.feedback import make_feedback

    perf = _perform(marked, staccato_len=0.5, p_vel=90, f_vel=40, swap=True)
    text = " ".join(make_feedback(_analyse(marked, perf), marked, max_items=8))
    assert "wrong order" in text and "staccato" in text and "from p to f" in text
