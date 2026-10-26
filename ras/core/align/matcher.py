"""Score-to-performance note matching (Parangonar) and global transposition detection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ras.core.score.reference import Reference
from ras.types import PerfNote, perf_array


class AlignmentError(RuntimeError):
    pass


@dataclass
class Alignment:
    pairs: list[tuple[int, int]]  # (reference row, performance index), pitch-equal matches
    unmatched_ref: list[int]
    unmatched_perf: list[int]
    method: str = "parangonar-automatic"


def estimate_transposition(ref: Reference, perf: list[PerfNote], max_shift: int = 12) -> tuple[int, float]:
    """Global semitone shift of the performance relative to the score (keyboard transpose).

    Compares duration-weighted pitch histograms. Returns (shift, evidence) where evidence is
    the overlap gain of the best shift over shift 0 (0 means no shift is better).
    """
    if not len(ref.notes) or not perf:
        return 0, 0.0
    size = 128 + 2 * max_shift
    h_ref = np.zeros(size)
    h_perf = np.zeros(size)
    for p, d in zip(ref.notes["pitch"], ref.notes["duration_beat"]):
        h_ref[p + max_shift] += d
    for n in perf:
        h_perf[n.pitch + max_shift] += max(n.offset - n.onset, 0.05)
    h_ref /= h_ref.sum() or 1.0
    h_perf /= h_perf.sum() or 1.0

    def overlap(k: int) -> float:  # performance shifted down by k to compare with the score
        return float(np.minimum(h_ref, np.roll(h_perf, -k)).sum())

    scores = {k: overlap(k) for k in range(-max_shift, max_shift + 1)}
    best = max(scores, key=scores.get)
    gain = scores[best] - scores[0]
    if best == 0 or gain < 0.15:
        return 0, 0.0
    return best, float(gain)


def _perf_groups(perf: list[PerfNote], gap_s: float = 0.09) -> list[list[int]]:
    """Cluster performed notes into onsets/chords: a new group starts after a silence gap."""
    order = sorted(range(len(perf)), key=lambda i: perf[i].onset)
    groups: list[list[int]] = []
    for i in order:
        if groups and perf[i].onset - perf[groups[-1][-1]].onset <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def match_groups(ref: Reference, perf: list[PerfNote]) -> Alignment:
    """Global monotone alignment of score onset-groups to performed onset-groups.

    Needleman-Wunsch over chords: similarity is the share of shared pitches, skipping a group
    costs a fixed penalty. Being global and ordered, it cannot swap two identical bars, which a
    local matcher can do when notes in one of them are wrong.
    """
    n_ref = len(ref.notes)
    rg: list[list[int]] = []
    keys = np.round(ref.notes["onset_beat"], 3)
    for k in np.unique(keys):
        rg.append(list(np.where(keys == k)[0]))
    pg = _perf_groups(perf)
    R, Q = len(rg), len(pg)
    rsets = [set(int(ref.notes["pitch"][i]) for i in g) for g in rg]
    psets = [set(perf[i].pitch for i in g) for g in pg]
    match_hi, match_lo, gap = 2.0, -0.8, -0.6
    H = np.zeros((R + 1, Q + 1))
    H[:, 0] = np.arange(R + 1) * gap
    H[0, :] = np.arange(Q + 1) * gap
    sim = np.zeros((R, Q))
    for i in range(R):
        for j in range(Q):
            inter = len(rsets[i] & psets[j])
            sim[i, j] = inter / max(len(rsets[i]), len(psets[j]))
    score = match_lo + match_hi * sim  # sim 0 -> -0.8 (aligned but unrelated), sim 1 -> +1.2
    for i in range(1, R + 1):
        for j in range(1, Q + 1):
            H[i, j] = max(H[i - 1, j - 1] + score[i - 1, j - 1], H[i - 1, j] + gap, H[i, j - 1] + gap)
    pairs: list[tuple[int, int]] = []
    i, j = R, Q
    while i > 0 and j > 0:
        if abs(H[i, j] - (H[i - 1, j - 1] + score[i - 1, j - 1])) < 1e-9:
            if sim[i - 1, j - 1] > 0:
                used = set()
                for r in sorted(rg[i - 1]):
                    for p in sorted(pg[j - 1], key=lambda x: perf[x].onset):
                        if p not in used and perf[p].pitch == int(ref.notes["pitch"][r]):
                            pairs.append((int(r), int(p)))
                            used.add(p)
                            break
            i, j = i - 1, j - 1
        elif abs(H[i, j] - (H[i - 1, j] + gap)) < 1e-9:
            i -= 1
        else:
            j -= 1
    pairs.sort()
    mr = {r for r, _ in pairs}
    mp = {p for _, p in pairs}
    return Alignment(pairs, [r for r in range(n_ref) if r not in mr],
                     [p for p in range(len(perf)) if p not in mp], method="group-dp")


def match_notes(ref: Reference, perf: list[PerfNote], transpose: int = 0,
                method: str = "group-dp") -> Alignment:
    """Match performed notes to score notes. `transpose` semitones are removed from the performance."""
    if method == "group-dp":
        shifted = [PerfNote(n.pitch - transpose, n.onset, n.offset, n.velocity) for n in perf]
        if not len(ref.notes) or not perf:
            return Alignment([], list(range(len(ref.notes))), list(range(len(perf))), method)
        return match_groups(ref, shifted)
    import parangonar as pa

    n_ref = len(ref.notes)
    if n_ref == 0 or not perf:
        return Alignment([], list(range(n_ref)), list(range(len(perf))))
    shifted = [PerfNote(n.pitch - transpose, n.onset, n.offset, n.velocity) for n in perf]
    perf_arr = perf_array(shifted)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pa.AutomaticNoteMatcher()(ref.notes, perf_arr)
    except Exception as exc:
        raise AlignmentError(f"parangonar failed: {exc}") from exc

    pairs, matched_r, matched_p = [], set(), set()
    for item in result:
        if item.get("label") != "match":
            continue
        r = int(str(item["score_id"])[1:])
        p = int(str(item["performance_id"])[1:])
        if r in matched_r or p in matched_p:
            continue
        pairs.append((r, p))
        matched_r.add(r)
        matched_p.add(p)
    return Alignment(
        pairs=sorted(pairs),
        unmatched_ref=[i for i in range(n_ref) if i not in matched_r],
        unmatched_perf=[i for i in range(len(perf)) if i not in matched_p],
    )
