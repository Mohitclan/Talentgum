"""Evaluation metrics: transcription (mir_eval), error detection, timing, bootstrap CIs."""

from __future__ import annotations

import warnings

import mir_eval
import numpy as np

from ras.core.score.reference import Reference
from ras.synth.performance import Truth
from ras.types import PerfNote

ERROR_TYPES = ("wrong_pitch", "wrong_octave", "missing", "extra")


def _arrays(notes: list[PerfNote]):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
    hz = mir_eval.util.midi_to_hz(np.array([n.pitch for n in notes], dtype=float))
    vel = np.array([n.velocity for n in notes], dtype=float)
    return iv, hz, vel


def transcription_metrics(truth: list[PerfNote], est: list[PerfNote]) -> dict:
    """Note-level P/R/F1 at the standard tolerances: onset, onset+offset, onset+velocity."""
    ri, rh, rv = _arrays(truth)
    ei, eh, ev = _arrays(est)
    out = {"n_truth": len(truth), "n_est": len(est)}
    if not len(est) or not len(truth):
        for k in ("onset", "onset_offset", "onset_velocity"):
            out[f"{k}_f1"] = 0.0
            out[f"{k}_precision"] = out[f"{k}_recall"] = 0.0
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ri, rh, ei, eh, offset_ratio=None)
        out.update(onset_precision=p, onset_recall=r, onset_f1=f)
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ri, rh, ei, eh, offset_ratio=0.2)
        out.update(onset_offset_precision=p, onset_offset_recall=r, onset_offset_f1=f)
        try:
            import mir_eval.transcription_velocity as tv

            p, r, f, _ = tv.precision_recall_f1_overlap(
                ri, rh, rv.astype(int), ei, eh, ev.astype(int), offset_ratio=None)
        except Exception:
            p = r = f = float("nan")
        out.update(onset_velocity_precision=p, onset_velocity_recall=r, onset_velocity_f1=f)
    return out


def detection_counts(result_notes: list[dict], truth: Truth, ref: Reference) -> dict:
    """Per error type tp/fp/fn.

    Repeated pitches make the exact score note ambiguous (a missing note in E E E could be
    any of the three), so a detection matches a true error of the same type and pitch within
    one beat of it.
    """
    beats, pitches = ref.notes["onset_beat"], ref.notes["pitch"]
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ERROR_TYPES}
    for typ in ("wrong_pitch", "wrong_octave", "missing"):
        true_idx = [i for i, s in enumerate(truth.ref_status) if s == typ]
        det_idx = [n["ref_index"] for n in result_notes if n["status"] == typ]
        pairs = sorted(
            ((abs(beats[i] - beats[j]), i, j) for i in true_idx for j in det_idx
             if pitches[i] == pitches[j] and abs(beats[i] - beats[j]) <= 1.01))
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i not in used_t and j not in used_d:
                used_t.add(i)
                used_d.add(j)
        counts[typ]["tp"] = len(used_t)
        counts[typ]["fn"] = len(true_idx) - len(used_t)
        counts[typ]["fp"] = len(det_idx) - len(used_d)

    det = [(n["perf_pitch"], n["perf_onset_s"]) for n in result_notes if n["status"] == "extra"]
    tru = [(e.pitch + truth.transpose, e.onset) for e in truth.extras]
    pairs = sorted(((abs(t[1] - d[1]), a, b) for a, t in enumerate(tru) for b, d in enumerate(det)
                    if t[0] == d[0] and abs(t[1] - d[1]) <= 0.35))
    ut, ud = set(), set()
    for _, a, b in pairs:
        if a not in ut and b not in ud:
            ut.add(a)
            ud.add(b)
    counts["extra"] = {"tp": len(ut), "fn": len(tru) - len(ut), "fp": len(det) - len(ud)}
    return counts


def timing_errors(result_notes: list[dict], truth: Truth) -> list[float]:
    """|estimated onset deviation - injected jitter| (ms) on notes that were truly played correctly."""
    errs = []
    for n in result_notes:
        i = n["ref_index"]
        if n["status"] == "correct" and i is not None and n["onset_dev_ms"] is not None \
                and truth.ref_status[i] == "ok":
            errs.append(abs(n["onset_dev_ms"] - float(truth.jitter_ms[i])))
    return errs


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) else float("nan")
    return p, r, f


def pooled_prf(per_rec: list[dict], typ: str | None = None):
    types = ERROR_TYPES if typ is None else (typ,)
    tp = sum(c[t]["tp"] for c in per_rec for t in types)
    fp = sum(c[t]["fp"] for c in per_rec for t in types)
    fn = sum(c[t]["fn"] for c in per_rec for t in types)
    return prf(tp, fp, fn)


def bootstrap_ci(items: list, stat, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of stat(list) resampling whole recordings (not notes)."""
    if not items:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(items), len(items))
        v = stat([items[i] for i in idx])
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))
