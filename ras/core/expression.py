"""Measures beyond right/wrong notes: order, chords, rhythm, fluency, articulation, dynamics, pedal.

Everything here is deterministic and runs on the alignment the error analysis already produced. The
results go into `Analysis.extras` (saved with the result, so a new rubric can re-score them later)
and are read by the rubric metrics in ras.core.rubric.engine.

Loudness has two possible sources:
- `velocity`: the transcriber's note velocity, only meaningful on touch-sensitive keyboards. It is
  unusable when the velocities come out flat.
- `model`: the audio dynamics model (ras.core.dynamics_model). It hears loudness, so it also works on
  keyboards without velocity.

Each loudness-based measure is reported per source, and the rubric chooses which one to use.
Thresholds are first guesses to be calibrated on teacher-graded recordings (`ras calibrate`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ras.core.errors.analyze import Analysis
from ras.core.score.reference import (
    ARTIC_ACCENT,
    ARTIC_LEGATO,
    ARTIC_STACCATISSIMO,
    ARTIC_STACCATO,
    ARTIC_TENUTO,
    Reference,
)
from ras.types import PedalEvent, PerfNote

PLAYED = ("correct", "wrong_pitch", "wrong_octave")


@dataclass
class ExpressionConfig:
    order_tolerance_s: float = 0.04  # a later score note this much before an earlier one is out of order
    chord_window_s: float = 0.08  # extra notes this close to a chord count as extra chord tones
    hesitation_ratio: float = 1.75  # an inter-onset gap this many times the local beat is a hesitation
    hesitation_min_s: float = 0.25
    rhythm_tolerance: float = 0.25  # an inter-onset ratio within +-25% counts as rhythmically right
    staccato_max_ratio: float = 0.6  # performed / written duration
    staccatissimo_max_ratio: float = 0.45
    tenuto_min_ratio: float = 0.8
    legato_max_gap_s: float = 0.08  # silence between slurred notes
    accent_velocity: float = 8.0  # louder than its neighbours by this much (MIDI velocity)
    accent_level: float = 0.3  # ... or by this much on the dynamics model's pp..ff scale
    dynamics_velocity_step: float = 5.0  # loudness change that counts as following a marking change
    dynamics_level_step: float = 0.15
    pedal_hop_s: float = 0.05


def _rolling_median(x: np.ndarray, w: int = 7) -> np.ndarray:
    if not len(x):
        return x
    h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def _groups(ref: Reference, analysis: Analysis):
    """Score onset groups with their played members and performed time (median of correct onsets)."""
    beats = ref.notes["onset_beat"]
    keys = np.round(beats, 3)
    uniq = np.unique(keys)
    members: dict[int, list] = {i: [] for i in range(len(uniq))}
    for res in analysis.results:
        if res.ref_index is not None:
            members[int(np.searchsorted(uniq, keys[res.ref_index]))].append(res)
    times = np.full(len(uniq), np.nan)
    for g, ms in members.items():
        ts = [m.perf_onset_s for m in ms if m.status == "correct" and m.perf_onset_s is not None]
        if ts:
            times[g] = float(np.median(ts))
    return uniq, members, times


def _order(uniq, members, times, cfg) -> dict:
    idx = np.flatnonzero(~np.isnan(times))
    pairs = inverted = 0
    where = []
    for a, b in zip(idx[:-1], idx[1:]):
        pairs += 1
        if times[b] < times[a] - cfg.order_tolerance_s:
            inverted += 1
            m = next((x for x in members[int(b)] if x.measure is not None), None)
            where.append({"measure": m.measure if m else None, "beat": m.beat_in_measure if m else None})
    # two neighbouring score notes played as each other's pitch: the classic swap
    wrongs = [r for ms in members.values() for r in ms if r.status in ("wrong_pitch", "wrong_octave")]
    swaps = 0
    for i, x in enumerate(wrongs):
        for y in wrongs[i + 1:]:
            if (x.perf_pitch == y.ref_pitch and y.perf_pitch == x.ref_pitch
                    and abs((x.perf_onset_s or 0) - (y.perf_onset_s or 0)) < 1.0):
                swaps += 1
                where.append({"measure": x.measure, "beat": x.beat_in_measure})
    return {"pairs": pairs, "inversions": inverted, "swaps": swaps, "where": where[:10]}


def _chords(ref, analysis, uniq, members, times, cfg) -> dict:
    extras = [r for r in analysis.results if r.status == "extra" and r.perf_onset_s is not None]
    out = []
    for g, ms in members.items():
        if len(ms) < 2:
            continue
        t = times[g]
        if np.isnan(t) and analysis.beat_to_sec is not None:
            t = analysis.beat_to_sec(float(uniq[g]))
        st = [m.status for m in ms]
        n_extra = 0 if np.isnan(t) else sum(1 for e in extras if abs(e.perf_onset_s - t) <= cfg.chord_window_s)
        played = sorted((m.perf_pitch, m.ref_pitch) for m in ms if m.status in PLAYED and m.perf_pitch is not None)
        score_pcs = {m.ref_pitch % 12 for m in ms}
        perf_pcs = {p % 12 for p, _ in played}
        bass_ref = min(m.ref_pitch for m in ms)
        inversion = bool(played) and perf_pcs == score_pcs and played[0][0] % 12 != bass_ref % 12
        out.append({
            "measure": ms[0].measure, "beat": ms[0].beat_in_measure, "tones": len(ms),
            "missing": st.count("missing"), "wrong": st.count("wrong_pitch"), "octave": st.count("wrong_octave"),
            "extra": n_extra, "voicing_error": bool(st.count("wrong_octave") or inversion),
            "correct": all(s == "correct" for s in st) and n_extra == 0,
        })
    return {"chords": out, "total": len(out), "tones": sum(c["tones"] for c in out)}


def _rhythm_and_fluency(uniq, times, analysis, cfg) -> dict:
    idx = np.flatnonzero(~np.isnan(times))
    if len(idx) < 3:
        return {"ioi_checked": 0, "ioi_within": 0, "ioi_median_dev_pct": None, "hesitations": []}
    b, t = uniq[idx], times[idx]
    dx, dy = np.diff(b), np.diff(t)
    ok = (dx > 0) & (dy > 0)
    spb = np.full(len(dx), np.nan)
    spb[ok] = _rolling_median(dy[ok] / dx[ok])
    ratio = dy / (dx * spb)
    valid = ok & np.isfinite(ratio)
    within = np.abs(np.log(np.clip(ratio[valid], 1e-3, None))) <= np.log(1 + cfg.rhythm_tolerance)
    pause_at = {round(p["at_s"], 2) for p in analysis.pauses}
    hes = []
    for k in np.flatnonzero(valid):
        extra_s = dy[k] - dx[k] * spb[k]
        if ratio[k] >= cfg.hesitation_ratio and extra_s >= cfg.hesitation_min_s and round(t[k + 1], 2) not in pause_at:
            hes.append({"at_s": round(float(t[k + 1]), 2), "extra_s": round(float(extra_s), 2)})
    dev = np.abs(ratio[valid] - 1.0) * 100 if valid.any() else np.array([])
    duration_min = max((t[-1] - t[0]) / 60.0, 1e-6)
    return {"ioi_checked": int(valid.sum()), "ioi_within": int(within.sum()),
            "ioi_median_dev_pct": float(np.median(dev)) if len(dev) else None,
            "hesitations": hes, "minutes": round(float(duration_min), 3),
            "local_spb": dict(zip([float(x) for x in b[:-1][ok]], [float(x) for x in spb[ok]]))}


def _local_spb_at(beat: float, rhythm: dict, default: float) -> float:
    m = rhythm.get("local_spb") or {}
    if not m:
        return default
    keys = np.array(list(m.keys()))
    return float(list(m.values())[int(np.argmin(np.abs(keys - beat)))])


def _loudness(analysis: Analysis, perf: list[PerfNote], model_levels: dict[int, float] | None) -> dict[str, dict]:
    """perf index -> loudness, per available source."""
    out: dict[str, dict] = {}
    if not analysis.velocity_flat:
        out["velocity"] = {i: float(n.velocity) for i, n in enumerate(perf)}
    if model_levels:
        out["model"] = {int(k): float(v) for k, v in model_levels.items() if np.isfinite(v)}
    return out


def _articulation(ref, perf, analysis, rhythm, loud, cfg) -> dict:
    notes = ref.notes
    spb_default = 60.0 / max(analysis.mean_bpm or ref.bpm, 1e-3)
    played = {r.ref_index: r for r in analysis.results if r.status == "correct" and r.perf_index is not None}
    # next score note in the same voice, for slur gaps
    order = sorted(range(len(notes)), key=lambda i: (notes["voice"][i], notes["onset_beat"][i], notes["pitch"][i]))
    nxt = {}
    for a, b in zip(order[:-1], order[1:]):
        if notes["voice"][a] == notes["voice"][b] and notes["onset_beat"][b] > notes["onset_beat"][a]:
            nxt[a] = b
    checks = {"staccato": [0, 0], "tenuto": [0, 0], "legato": [0, 0], "accent": [0, 0]}
    per_note_ratio = {}
    issues = []
    for r, res in played.items():
        p = perf[res.perf_index]
        spb = _local_spb_at(float(notes["onset_beat"][r]), rhythm, spb_default)
        written = float(notes["duration_beat"][r]) * spb
        ratio = (p.offset - p.onset) / written if written > 0 else None
        if ratio is not None:
            per_note_ratio[r] = round(float(ratio), 3)
        flags = int(notes["artic"][r])
        verdicts = []
        if flags & (ARTIC_STACCATO | ARTIC_STACCATISSIMO) and ratio is not None:
            limit = cfg.staccatissimo_max_ratio if flags & ARTIC_STACCATISSIMO else cfg.staccato_max_ratio
            verdicts.append(("staccato", ratio <= limit))
        if flags & ARTIC_TENUTO and ratio is not None:
            verdicts.append(("tenuto", ratio >= cfg.tenuto_min_ratio))
        if flags & ARTIC_LEGATO and r in nxt and nxt[r] in played:
            gap = perf[played[nxt[r]].perf_index].onset - p.offset
            verdicts.append(("legato", gap <= cfg.legato_max_gap_s))
        if flags & ARTIC_ACCENT and loud:
            src, vals = next(iter(loud.items()))
            neigh = [vals[played[k].perf_index] for k in range(max(0, r - 3), min(len(notes), r + 4))
                     if k != r and k in played and played[k].perf_index in vals]
            if neigh and res.perf_index in vals:
                step = cfg.accent_velocity if src == "velocity" else cfg.accent_level
                verdicts.append(("accent", vals[res.perf_index] - float(np.median(neigh)) >= step))
        for kind, ok in verdicts:
            checks[kind][0] += int(ok)
            checks[kind][1] += 1
            if not ok and len(issues) < 10:
                issues.append({"kind": kind, "measure": res.measure, "beat": res.beat_in_measure})
    ok = sum(v[0] for v in checks.values())
    n = sum(v[1] for v in checks.values())
    return {"checked": n, "ok": ok, "by_type": {k: {"ok": v[0], "checked": v[1]} for k, v in checks.items()},
            "issues": issues, "marks_in_score": bool(ref.has_articulation_marks), "ratio_by_note": per_note_ratio}


def _dynamics(ref, analysis, loud, cfg) -> dict:
    notes = ref.notes
    played = {r.ref_index: r.perf_index for r in analysis.results if r.status == "correct" and r.perf_index is not None}
    out = {"marks_in_score": bool(ref.has_dynamics_marks)}
    for src, vals in loud.items():
        step = cfg.dynamics_velocity_step if src == "velocity" else cfg.dynamics_level_step
        by_ref = {r: vals[p] for r, p in played.items() if p in vals}
        order = sorted(by_ref, key=lambda r: notes["onset_beat"][r])
        # sections of constant written dynamics, in score order
        sections, cur, cur_level = [], [], None
        for r in sorted(range(len(notes)), key=lambda i: notes["onset_beat"][i]):
            lv = notes["dyn"][r]
            lv = None if np.isnan(lv) else float(lv)
            if lv != cur_level and cur:
                sections.append((cur_level, cur))
                cur = []
            cur_level = lv
            if r in by_ref:
                cur.append(by_ref[r])
        if cur:
            sections.append((cur_level, cur))
        marked = [(lv, float(np.median(v)), v) for lv, v in sections if lv is not None and len(v) >= 2]
        comparisons, ok, details = 0, 0, []
        for (l1, m1, _), (l2, m2, _) in zip(marked[:-1], marked[1:]):
            if l1 == l2:
                continue
            comparisons += 1
            good = np.sign(m2 - m1) == np.sign(l2 - l1) and abs(m2 - m1) >= step
            ok += int(good)
            details.append({"from": l1, "to": l2, "change": round(m2 - m1, 2), "ok": bool(good)})
        for hp in ref.hairpins:
            inside = [r for r in order if hp["start_beat"] <= notes["onset_beat"][r] <= hp["end_beat"]]
            if len(inside) < 4:
                continue
            k = max(1, len(inside) // 3)
            change = float(np.median([by_ref[r] for r in inside[-k:]]) - np.median([by_ref[r] for r in inside[:k]]))
            good = (change >= step) if hp["kind"] == "cresc" else (change <= -step)
            comparisons += 1
            ok += int(good)
            details.append({"hairpin": hp["kind"], "change": round(change, 2), "ok": bool(good)})
        # evenness inside each section (or the whole piece when nothing is marked)
        groups = [v for _, _, v in marked] or ([list(by_ref.values())] if by_ref else [])
        stds = [float(np.std(v)) for v in groups if len(v) >= 3]
        out[src] = {"comparisons": comparisons, "ok": ok, "details": details[:20],
                    "consistency_std": float(np.mean(stds)) if stds else None,
                    "range": float(np.ptp(list(by_ref.values()))) if by_ref else None,
                    "by_note": {int(r): round(v, 3) for r, v in by_ref.items()}}
    return out


def _pedal(ref, analysis, pedal: list[PedalEvent] | None, perf: list[PerfNote], cfg) -> dict:
    if pedal is None:
        return {"supported": False, "marks_in_score": bool(ref.pedal_spans)}
    out = {"supported": True, "marks_in_score": bool(ref.pedal_spans)}
    if not perf or analysis.beat_to_sec is None:
        return out
    lo, hi = min(n.onset for n in perf), max(n.offset for n in perf)
    grid = np.arange(lo, hi, cfg.pedal_hop_s)
    did = np.zeros(len(grid), bool)
    for p in pedal:
        did |= (grid >= p.start) & (grid < p.end)
    want = np.zeros(len(grid), bool)
    for s, e in ref.pedal_spans:
        want |= (grid >= analysis.beat_to_sec(s)) & (grid < analysis.beat_to_sec(e))
    out["down_share"] = round(float(did.mean()), 3) if len(grid) else None
    if ref.pedal_spans and len(grid):
        out["agreement"] = round(float((did == want).mean()), 3)
        out["coverage"] = round(float(did[want].mean()), 3) if want.any() else None
        out["misuse"] = round(float(did[~want].mean()), 3) if (~want).any() else None
    return out


def measure(ref: Reference, perf: list[PerfNote], analysis: Analysis, pedal: list[PedalEvent] | None = None,
            model_levels: dict[int, float] | None = None, cfg: ExpressionConfig | None = None) -> dict:
    """Fill and return `analysis.extras`. `pedal` is None when the transcriber cannot detect the pedal."""
    cfg = cfg or ExpressionConfig()
    uniq, members, times = _groups(ref, analysis)
    rhythm = _rhythm_and_fluency(uniq, times, analysis, cfg)
    loud = _loudness(analysis, perf, model_levels)
    extras = {
        "order": _order(uniq, members, times, cfg),
        "chords": _chords(ref, analysis, uniq, members, times, cfg),
        "rhythm": {k: v for k, v in rhythm.items() if k != "local_spb"},
        "articulation": _articulation(ref, perf, analysis, rhythm, loud, cfg),
        "dynamics": _dynamics(ref, analysis, loud, cfg),
        "pedal": _pedal(ref, analysis, pedal, perf, cfg),
        "tempo": {"mean_bpm": analysis.mean_bpm, "reference_bpm": ref.bpm,
                  "ratio_pct": None if not analysis.mean_bpm else round(100.0 * analysis.mean_bpm / ref.bpm, 1)},
        "loudness_sources": sorted(loud),
    }
    analysis.extras = extras
    return extras
