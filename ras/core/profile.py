"""Teacher benchmark profiles: how teachers actually play a song, so students are compared with them.

The client offered 2-3 teacher recordings per song. They are assessed like any recording, then the
per-note measurements are pooled into a profile:
- how long each note is held (performed / written duration, which also captures articulation);
- how loud it is relative to the rest of the piece (z-scored per recording, per loudness source);
- the tempo.

A student is then judged against what teachers do on the same kind of keyboard and microphone,
rather than against ideal durations and loudness the instrument may not even produce. The spread
between teachers sets the tolerance, so a note the teachers play freely is judged leniently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROFILE_VERSION = 1
MIN_TOLERANCE = np.log(1.3)  # never stricter than +-30% on a duration ratio


def _by_note(d: dict | None) -> dict[int, float]:
    return {int(k): float(v) for k, v in (d or {}).items() if v is not None and np.isfinite(v)}


def _z(values: dict[int, float]) -> dict[int, float]:
    if len(values) < 3:
        return {}
    v = np.array(list(values.values()))
    sd = float(np.std(v)) or 1.0
    return {k: (x - float(np.mean(v))) / sd for k, x in values.items()}


def note_features(extras: dict) -> dict:
    """Per-note features of one assessed recording, from its scoring inputs' `extras`."""
    dyn = extras.get("dynamics", {}) or {}
    return {
        "dur_ratio": _by_note((extras.get("articulation") or {}).get("ratio_by_note")),
        "loud_z": {src: _z(_by_note((dyn.get(src) or {}).get("by_note"))) for src in ("velocity", "model")},
        "bpm": (extras.get("tempo") or {}).get("mean_bpm"),
    }


def build_profile(recordings: list[dict], reference_sha256: str) -> dict:
    """`recordings`: the `scoring_inputs` of each teacher recording's assessment."""
    feats = [note_features(r.get("extras", {})) for r in recordings]
    notes: dict[int, dict] = {}
    for f in feats:
        for r, v in f["dur_ratio"].items():
            notes.setdefault(r, {}).setdefault("dur", []).append(np.log(max(v, 1e-3)))
        for src, vals in f["loud_z"].items():
            for r, v in vals.items():
                notes.setdefault(r, {}).setdefault(src, []).append(v)
    out_notes = {}
    for r, d in notes.items():
        out_notes[str(r)] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v)] for k, v in d.items()}
    bpms = [f["bpm"] for f in feats if f["bpm"]]
    return {"version": PROFILE_VERSION, "reference_sha256": reference_sha256, "recordings": len(recordings),
            "tempo_bpm": [round(float(np.mean(bpms)), 2), round(float(np.std(bpms)), 2)] if bpms else None,
            "notes": out_notes}


def compare(extras: dict, profile: dict) -> dict:
    """Student vs teacher profile; fills the `teacher` block that the *_vs_teacher metrics read."""
    f = note_features(extras)
    prof = {int(k): v for k, v in profile.get("notes", {}).items()}
    out: dict = {"recordings": profile.get("recordings")}

    within = checked = art_ok = art_n = 0
    for r, v in f["dur_ratio"].items():
        p = prof.get(r, {}).get("dur")
        if not p:
            continue
        mean, sd = p[0], p[1]
        s = np.log(max(v, 1e-3))
        checked += 1
        within += int(abs(s - mean) <= max(2 * sd, MIN_TOLERANCE))
        # articulation as the teachers play it: clearly detached (< 70%) or clearly connected (> 90%)
        if mean < np.log(0.7) or mean > np.log(0.9):
            art_n += 1
            art_ok += int((s < np.log(0.8)) == (mean < np.log(0.7)))
    out["duration_within_pct"] = 100.0 * within / checked if checked else None
    out["articulation_within_pct"] = 100.0 * art_ok / art_n if art_n else None

    out["dynamics_corr"] = None
    for src in ("velocity", "model"):
        pairs = [(v, prof[r][src][0]) for r, v in f["loud_z"].get(src, {}).items() if src in prof.get(r, {})]
        if len(pairs) >= 6:
            a, b = np.array(pairs).T
            if np.std(a) > 0 and np.std(b) > 0:
                out["dynamics_corr"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
                out["dynamics_source"] = src
                break

    tb = profile.get("tempo_bpm")
    out["tempo_deviation_pct"] = (abs(f["bpm"] / tb[0] - 1.0) * 100.0) if tb and f["bpm"] else None
    return out


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
