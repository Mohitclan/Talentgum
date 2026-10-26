import numpy as np

from ras.core.align import estimate_transposition, match_notes
from ras.core.errors import analyze
from ras.synth.performance import PerfSpec, simulate_performance


def run(ref, spec):
    truth = simulate_performance(ref, spec)
    alignment = match_notes(ref, truth.notes)
    return truth, analyze(ref, truth.notes, alignment)


def test_perfect_performance_has_no_errors(twinkle):
    _, a = run(twinkle, PerfSpec(seed=1))
    assert (a.n_correct, a.n_wrong_pitch, a.n_wrong_octave, a.n_missing, a.n_extra) == (78, 0, 0, 0, 0)
    assert a.chords_correct == a.chords_total == 12
    assert a.pauses == []


def test_single_errors_are_classified(twinkle):
    for field, status in [("n_wrong_pitch", "wrong_pitch"), ("n_wrong_octave", "wrong_octave"),
                          ("n_missing", "missing")]:
        truth, a = run(twinkle, PerfSpec(seed=3, **{field: 1}))
        bad = [r for r in a.results if r.status not in ("correct",)]
        assert [r.status for r in bad] == [status]
        true_i = truth.ref_status.index(status)
        assert bad[0].ref_index == true_i


def test_extra_note_is_found(twinkle):
    truth, a = run(twinkle, PerfSpec(seed=5, n_extra=2))
    assert a.n_extra == 2 and a.n_correct == 78
    assert {r.perf_pitch for r in a.results if r.status == "extra"} == {e.pitch for e in truth.extras}


def test_error_detection_on_many_mistakes_is_mostly_right(frere):
    hits = total = false_alarms = 0
    for seed in range(8):
        truth, a = run(frere, PerfSpec(seed=seed, n_wrong_pitch=3, n_missing=3, n_extra=2,
                                       tempo_factor=0.7 + 0.05 * seed))
        true_bad = {i for i, s in enumerate(truth.ref_status) if s != "ok"}
        det_bad = {r.ref_index for r in a.results if r.ref_index is not None and r.status != "correct"}
        hits += len(true_bad & det_bad)
        total += len(true_bad)
        false_alarms += len(det_bad - true_bad)
    assert hits / total > 0.85
    assert false_alarms / total < 0.2


def test_slow_student_is_not_penalised_for_tempo(twinkle):
    _, a = run(twinkle, PerfSpec(seed=2, tempo_factor=0.55, jitter_ms=10))
    assert a.n_correct == 78
    assert 50 < a.mean_bpm < 60
    assert np.median(np.abs(a.onset_dev_ms)) < 30


def test_pause_is_detected_with_its_length(twinkle):
    _, a = run(twinkle, PerfSpec(seed=2, pauses=[(30, 3.0)]))
    assert len(a.pauses) == 1
    assert 2.0 < a.pauses[0]["extra_s"] < 4.0
    assert a.n_correct == 78  # a pause is not a wrong note


def test_transposition_detected_and_removed(twinkle):
    truth = simulate_performance(twinkle, PerfSpec(seed=4, transpose=2))
    shift, evidence = estimate_transposition(twinkle, truth.notes)
    assert shift == 2 and evidence > 0.15
    unshifted, _ = estimate_transposition(twinkle, simulate_performance(twinkle, PerfSpec(seed=4)).notes)
    assert unshifted == 0


def test_timing_estimate_tracks_injected_jitter(twinkle):
    truth, a = run(twinkle, PerfSpec(seed=7, jitter_ms=30))
    errs = [abs(r.onset_dev_ms - truth.jitter_ms[r.ref_index])
            for r in a.results if r.status == "correct" and r.onset_dev_ms is not None]
    assert len(errs) > 60 and np.mean(errs) < 30


def test_empty_performance_is_all_missing(twinkle):
    a = analyze(twinkle, [], match_notes(twinkle, []))
    assert a.n_missing == 78 and a.n_correct == 0 and a.n_extra == 0
