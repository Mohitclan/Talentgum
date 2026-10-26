import numpy as np

from ras.core.align import match_notes
from ras.core.errors import analyze
from ras.eval.metrics import bootstrap_ci, detection_counts, pooled_prf, prf, transcription_metrics
from ras.eval.splits import grouped_kfold, leave_one_song_out
from ras.synth.performance import PerfSpec, simulate_performance
from ras.types import PerfNote


def test_transcription_metrics_perfect_and_empty():
    notes = [PerfNote(60 + i, i * 0.5, i * 0.5 + 0.4, 70) for i in range(6)]
    m = transcription_metrics(notes, notes)
    assert m["onset_f1"] == m["onset_offset_f1"] == 1.0
    assert transcription_metrics(notes, [])["onset_f1"] == 0.0


def test_transcription_metrics_tolerance_and_pitch():
    truth = [PerfNote(60, 1.0, 1.4, 70), PerfNote(64, 2.0, 2.4, 70)]
    late = [PerfNote(60, 1.04, 1.44, 70), PerfNote(64, 2.2, 2.6, 70)]  # one within 50 ms, one not
    assert transcription_metrics(truth, late)["onset_recall"] == 0.5
    wrong = [PerfNote(61, 1.0, 1.4, 70), PerfNote(64, 2.0, 2.4, 70)]
    assert transcription_metrics(truth, wrong)["onset_precision"] == 0.5


def test_detection_counts_exact_with_perfect_notes(twinkle):
    t = simulate_performance(twinkle, PerfSpec(seed=11, n_wrong_pitch=2, n_missing=2, n_extra=2))
    a = analyze(twinkle, t.notes, match_notes(twinkle, t.notes))
    from dataclasses import asdict

    c = detection_counts([asdict(r) for r in a.results], t, twinkle)
    assert c["wrong_pitch"] == {"tp": 2, "fp": 0, "fn": 0}
    assert c["missing"] == {"tp": 2, "fp": 0, "fn": 0}
    assert c["extra"] == {"tp": 2, "fp": 0, "fn": 0}
    assert pooled_prf([c])[:2] == (1.0, 1.0)


def test_prf_edge_cases():
    assert prf(0, 0, 0)[2] != prf(0, 0, 0)[2]  # nan
    assert prf(3, 1, 1) == (0.75, 0.75, 0.75)


def test_bootstrap_ci_brackets_the_mean():
    vals = list(np.random.default_rng(0).normal(0.8, 0.05, 40))
    lo, hi = bootstrap_ci(vals, lambda v: float(np.mean(v)))
    assert lo < 0.8 < hi and hi - lo < 0.06


def test_grouped_kfold_never_splits_a_student():
    groups = [f"s{i % 7}" for i in range(40)]
    seen = set()
    for train, test in grouped_kfold(groups, k=3, seed=1):
        assert not ({groups[i] for i in train} & {groups[i] for i in test})
        seen |= set(test)
    assert seen == set(range(40))  # every recording is tested exactly once


def test_leave_one_song_out_holds_the_song_out():
    songs = ["a", "b", "c", "a", "b", "c"]
    for song, train, test in leave_one_song_out(songs):
        assert all(songs[i] != song for i in train) and all(songs[i] == song for i in test)
