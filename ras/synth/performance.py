"""Simulate a student performance of a reference, with known injected mistakes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ras.core.score.reference import Reference
from ras.types import PerfNote


@dataclass
class PerfSpec:
    tempo_factor: float = 1.0  # <1 slower than the score tempo
    drift: float = 0.06  # slow tempo wobble amplitude
    jitter_ms: float = 15.0  # random onset deviation (std)
    duration_factor: float = 0.9
    velocity_mean: float = 70.0
    velocity_std: float = 8.0
    n_wrong_pitch: int = 0
    n_wrong_octave: int = 0
    n_missing: int = 0
    n_extra: int = 0
    pauses: list[tuple[int, float]] = field(default_factory=list)  # (ref note index, seconds)
    transpose: int = 0
    start_offset_s: float = 1.0
    seed: int = 0


@dataclass
class Truth:
    """Ground truth for evaluation. Indices refer to Reference.notes rows."""

    ref_status: list[str]  # ok | wrong_pitch | wrong_octave | missing
    played_pitch: list[int | None]
    jitter_ms: np.ndarray  # injected onset deviation per reference note
    extras: list[PerfNote]
    notes: list[PerfNote]  # everything actually played (the transcription target)
    transpose: int = 0

    def counts(self) -> dict[str, int]:
        out = {"wrong_pitch": 0, "wrong_octave": 0, "missing": 0, "extra": len(self.extras)}
        for s in self.ref_status:
            if s in out:
                out[s] += 1
        return out


def simulate_performance(ref: Reference, spec: PerfSpec) -> Truth:
    rng = np.random.default_rng(spec.seed)
    notes = ref.notes
    n = len(notes)
    base_spb = 60.0 / (ref.bpm * spec.tempo_factor)
    P = 6.0

    def sec(beat: float) -> float:
        return base_spb * (beat + spec.drift * P * (1 - np.cos(beat / P)))

    def spb_at(beat: float) -> float:
        return base_spb * (1 + spec.drift * np.sin(beat / P))

    pool = list(range(n))
    rng.shuffle(pool)

    def take(k):
        out, pool[:] = pool[:k], pool[k:]
        return out

    missing = set(take(spec.n_missing))
    wrong_pitch = take(spec.n_wrong_pitch)
    wrong_oct = take(spec.n_wrong_octave)
    status = ["ok"] * n
    played: list[int | None] = [int(p) for p in notes["pitch"]]
    for i in missing:
        status[i], played[i] = "missing", None
    beats = notes["onset_beat"]

    def unambiguous(i: int, pitch: int) -> bool:
        """A wrong note that equals a nearby score pitch would legitimately look correct."""
        near = np.abs(beats - beats[i]) <= 1.5
        return not np.any(near & (notes["pitch"] == pitch))

    def pick(i: int, offsets) -> int:
        offsets = list(rng.permutation(offsets))
        for off in offsets:
            if unambiguous(i, int(notes["pitch"][i] + off)):
                return int(notes["pitch"][i] + off)
        return int(notes["pitch"][i] + offsets[0])

    for i in wrong_pitch:
        played[i] = pick(i, [-2, -1, 1, 2])
        status[i] = "wrong_pitch"
    for i in wrong_oct:
        played[i] = pick(i, [-12, 12])
        status[i] = "wrong_octave"

    pause_shift = np.zeros(n)
    for idx, secs in spec.pauses:
        pause_shift[notes["onset_beat"] >= notes["onset_beat"][idx]] += secs

    jitter = rng.normal(0.0, spec.jitter_ms, n)
    played_notes: list[PerfNote] = []
    for i in range(n):
        if played[i] is None:
            continue
        beat = float(notes["onset_beat"][i])
        onset = spec.start_offset_s + sec(beat) + pause_shift[i] + jitter[i] / 1000.0
        dur = max(0.08, float(notes["duration_beat"][i]) * spb_at(beat) * spec.duration_factor)
        vel = int(np.clip(rng.normal(spec.velocity_mean, spec.velocity_std), 20, 120))
        played_notes.append(PerfNote(played[i] + spec.transpose, onset, onset + dur, vel))

    extras: list[PerfNote] = []
    if spec.n_extra and played_notes:
        lo = min(p.onset for p in played_notes)
        hi = max(p.onset for p in played_notes)
        for _ in range(spec.n_extra):
            t = float(rng.uniform(lo, hi))
            near = min(played_notes, key=lambda p: abs(p.onset - t))
            for _ in range(20):
                pitch = int(np.clip(near.pitch + rng.integers(-7, 8), 30, 100))
                if not any(abs(p.onset - t) < 0.6 and p.pitch == pitch for p in played_notes):
                    break
            ex = PerfNote(pitch, t, t + 0.25, int(rng.integers(40, 90)))
            extras.append(ex)
            played_notes.append(ex)

    played_notes = _fix_same_pitch_overlap(played_notes)
    played_notes.sort(key=lambda p: (p.onset, p.pitch))
    return Truth(status, played, jitter, extras, played_notes, spec.transpose)


def _fix_same_pitch_overlap(notes: list[PerfNote]) -> list[PerfNote]:
    by_pitch: dict[int, list[PerfNote]] = {}
    for p in notes:
        by_pitch.setdefault(p.pitch, []).append(p)
    for lst in by_pitch.values():
        lst.sort(key=lambda p: p.onset)
        for a, b in zip(lst, lst[1:]):
            if a.offset > b.onset - 0.03:
                a.offset = max(a.onset + 0.05, b.onset - 0.03)
    return notes
