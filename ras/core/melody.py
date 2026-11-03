"""One-line songs on real recordings: find the melody among everything the transcriber heard.

Transkun is a piano model: on keyboard sounds through a class call it reports the harmonics of a note
as notes of their own, sometimes louder than the note itself, so loudness cannot tell them apart.
Some recordings also have chords sounding under the melody (a second hand, the teacher, or the
keyboard's accompaniment). Left in, these count as extra notes and fool the transposition check.

When the score has one note at a time, the score itself shows which note of a simultaneous group is
the melody. Only that note goes on to scoring; the rest are counted and reported, never silently lost.

`reference_fit` answers a different question: does this playing look like this song at all? The
pipeline trusts the song it is given, so the wrong reference would otherwise give a confident score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ras.core.align.matcher import _perf_groups, estimate_transposition, match_groups
from ras.core.score.reference import Reference
from ras.types import PerfNote

# intervals of the 2nd to 8th harmonics above a note, in semitones (nearest key)
HARMONIC_INTERVALS = frozenset({12, 19, 24, 28, 31, 34, 36})
OCTAVE_SHIFTS = (0, 12, -12, 24, -24)
MIN_SHIFT_GAIN = 1.1  # a shift must explain this many times more notes than no shift
ACCOMPANIMENT_MIN_GROUPS = 3
ACCOMPANIMENT_MIN_SHARE = 0.15


@dataclass
class MelodyResult:
    notes: list[PerfNote]  # the melody, in the performance's own pitch
    shift: int = 0
    evidence: float = 0.0
    applied: bool = False  # False: the score has chords, nothing was changed
    harmonics_removed: int = 0
    accompaniment_removed: int = 0
    accompaniment_groups: int = 0
    groups: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def accompaniment_heard(self) -> bool:
        return (self.accompaniment_groups >= ACCOMPANIMENT_MIN_GROUPS
                and self.accompaniment_groups >= ACCOMPANIMENT_MIN_SHARE * max(self.groups, 1))

    def to_dict(self) -> dict:
        return {"applied": self.applied, "harmonics_removed": self.harmonics_removed,
                "accompaniment_removed": self.accompaniment_removed,
                "accompaniment_groups": self.accompaniment_groups, "groups": self.groups,
                "accompaniment_heard": self.accompaniment_heard}


def is_single_line(ref: Reference) -> bool:
    keys = np.round(ref.notes["onset_beat"], 3)
    return len(keys) > 0 and len(np.unique(keys)) == len(keys)


def _is_harmonic(note: PerfNote, others: list[PerfNote]) -> bool:
    return any(note.pitch - o.pitch in HARMONIC_INTERVALS for o in others if o is not note)


def _shifted(notes: list[PerfNote], k: int) -> list[PerfNote]:
    return [PerfNote(n.pitch - k, n.onset, n.offset, n.velocity) for n in notes]


def _best_shift(ref: Reference, notes: list[PerfNote]) -> tuple[int, float, dict[int, object]]:
    """Which whole shift lets the most score notes be found among the notes heard."""
    hist_shift, _ = estimate_transposition(ref, notes)
    candidates = list(OCTAVE_SHIFTS) + ([hist_shift] if hist_shift not in OCTAVE_SHIFTS else [])
    found = {k: match_groups(ref, _shifted(notes, k), any_member=True) for k in candidates}
    base = len(found[0].pairs)
    best = 0
    for k in candidates:  # in order of size: a tie keeps the smaller shift
        if len(found[k].pairs) > len(found[best].pairs):
            best = k
    if best != 0 and len(found[best].pairs) < MIN_SHIFT_GAIN * max(base, 1):
        best = 0
    gain = (len(found[best].pairs) - base) / max(len(ref.notes), 1)
    return best, float(gain), found


def isolate_melody(ref: Reference, notes: list[PerfNote]) -> MelodyResult:
    if not notes or not is_single_line(ref):
        return MelodyResult(notes=list(notes))
    shift, evidence, found = _best_shift(ref, notes)
    work = _shifted(notes, shift)
    matched = {p for _, p in found[shift].pairs}
    median = float(np.median(ref.notes["pitch"]))
    res = MelodyResult(notes=[], shift=shift, evidence=evidence, applied=True)
    keep: list[int] = []
    for group in _perf_groups(work):
        res.groups += 1
        members = [work[i] for i in group]
        chosen = [i for i in group if i in matched]
        if not chosen:  # no score note here: a wrong note or a stray sound, keep one candidate
            plain = [i for i in group if not _is_harmonic(work[i], members)] or list(group)
            chosen = [min(plain, key=lambda i: abs(work[i].pitch - median))]
        keep += chosen
        lead = work[chosen[0]]
        has_lower = False
        for i in group:
            if i in chosen:
                continue
            n = work[i]
            if _is_harmonic(n, members) and n.pitch > lead.pitch:
                res.harmonics_removed += 1
                kind = "harmonic"
            else:
                res.accompaniment_removed += 1
                has_lower = has_lower or n.pitch < lead.pitch
                kind = "accompaniment"
            res.log.append({"onset_s": round(n.onset, 3), "pitch": notes[i].pitch, "kind": kind})
        res.accompaniment_groups += has_lower
    res.notes = [notes[i] for i in sorted(keep, key=lambda i: notes[i].onset)]
    return res


def reference_fit(results) -> float:
    """Of the score notes in the stretch that was played, the share found (any octave).

    Stopping early does not lower it: only the stretch between the first and last note found counts.
    """
    found = sorted(r.ref_index for r in results if r.ref_index is not None
                   and r.status in ("correct", "wrong_octave"))
    if len(found) < 4:
        return 0.0
    return len(found) / (found[-1] - found[0] + 1)
