"""Deterministic error detection and timing metrics from an alignment. No learned parts.

Statuses per score note: correct | wrong_pitch | wrong_octave | missing. Performance notes
that match nothing are `extra`. Timing is measured against a *local* tempo estimated from the
matched notes, so a slower or drifting student is not penalised for tempo alone.

Known limit (Phase 1): matching is monotone, so a restart or skipped bar shows up as
missing/extra notes rather than being modelled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.isotonic import IsotonicRegression

from ras.core.align.matcher import Alignment
from ras.core.score.reference import Reference
from ras.types import PerfNote


@dataclass
class AnalysisConfig:
    pause_threshold_s: float = 1.5  # extra silence beyond the expected gap that counts as a pause
    window_groups: int = 4  # onsets on each side used for the local tempo fit
    wrong_note_window_s: float = 0.30  # how close in time a wrong note must be to its target
    max_wrong_interval: int = 7  # semitones; larger jumps are treated as missing + extra
    flat_velocity_std: float = 2.0  # below this the keyboard is treated as not velocity-sensitive
    implausible_beats: float = 0.8  # a match this many beats off its expected time is a mismatch
    min_restart_notes: int = 4  # a run of unexplained notes this long that replays the score is a restart
    restart_match_share: float = 0.6  # ...when at least this share of it matches a stretch of the score


@dataclass
class NoteResult:
    status: str
    ref_index: int | None = None
    perf_index: int | None = None
    measure: int | None = None
    beat_in_measure: float | None = None
    ref_pitch: int | None = None
    perf_pitch: int | None = None
    perf_onset_s: float | None = None
    onset_dev_ms: float | None = None
    duration_ratio: float | None = None
    expected_onset_s: float | None = None  # where the score note should have sounded
    audio_evidence_db: float | None = None  # onset jump found in the audio for a missing note


@dataclass
class Analysis:
    results: list[NoteResult]
    n_ref: int
    n_perf: int
    n_correct: int
    n_wrong_pitch: int
    n_wrong_octave: int
    n_missing: int
    n_extra: int
    n_optional_played: int
    chords_total: int
    chords_correct: int
    onset_dev_ms: np.ndarray  # per matched note with an estimate
    duration_ratio: np.ndarray
    ref_bpm: float
    mean_bpm: float | None
    tempo_cv: float | None
    pauses: list[dict] = field(default_factory=list)
    velocity_std: float | None = None
    velocity_range: float | None = None
    velocity_flat: bool = True
    notes: list[str] = field(default_factory=list)
    n_uncertain: int = 0  # reported missing, but the audio shows an onset: we cannot tell
    n_replayed: int = 0  # notes of an abandoned attempt the student then played again
    restarts: list[dict] = field(default_factory=list)
    pedal_supported: bool = False
    pedal_events: int = 0
    pedal_down_share: float | None = None  # share of the performance time with the sustain pedal down
    n_unconfirmed_extra: int = 0  # extra notes a second transcriber did not hear: not held against the student
    extras: dict = field(default_factory=dict)  # expression, chord, order and rhythm measures (ras.core.expression)
    beat_to_sec: object = None  # score beat -> performance seconds (the timing map); not serialised

    @property
    def n_scored(self) -> int:
        """Score notes whose outcome is known."""
        return self.n_ref - self.n_uncertain

    _COUNT = {"missing": "n_missing", "wrong_pitch": "n_wrong_pitch", "wrong_octave": "n_wrong_octave"}

    def mark_uncertain(self, result_indices: list[int]) -> None:
        """Missing or wrong notes whose outcome we cannot be sure of: not held against the student."""
        for i in result_indices:
            r = self.results[i]
            if r.status in self._COUNT:
                setattr(self, self._COUNT[r.status], getattr(self, self._COUNT[r.status]) - 1)
                r.status = "uncertain"
                self.n_uncertain += 1

    def mark_unconfirmed_extra(self, result_indices: list[int]) -> None:
        for i in result_indices:
            r = self.results[i]
            if r.status == "extra":
                r.status = "unconfirmed_extra"
                self.n_extra -= 1
                self.n_unconfirmed_extra += 1

    @property
    def coverage(self) -> float:
        return (self.n_correct + self.n_wrong_pitch + self.n_wrong_octave) / max(self.n_ref, 1)

    @property
    def extra_ratio(self) -> float:
        return self.n_extra / max(self.n_perf, 1)

    def n_matched_for_timing(self) -> int:
        return self.n_correct

    _SNAPSHOT = ("n_ref", "n_perf", "n_correct", "n_wrong_pitch", "n_wrong_octave", "n_missing", "n_extra",
                 "n_optional_played", "chords_total", "chords_correct", "ref_bpm", "mean_bpm", "tempo_cv", "pauses",
                 "velocity_std", "velocity_range", "velocity_flat", "n_uncertain", "n_replayed", "restarts",
                 "pedal_supported", "pedal_events", "pedal_down_share", "n_unconfirmed_extra", "extras")

    def scoring_inputs(self) -> dict:
        """Everything the rubric engine reads, so results can be re-scored later without the audio."""
        d = {k: getattr(self, k) for k in self._SNAPSHOT}
        d["onset_dev_ms"] = [round(float(x), 1) for x in self.onset_dev_ms]
        d["duration_ratio"] = [round(float(x), 3) for x in self.duration_ratio]
        return d

    @classmethod
    def from_scoring_inputs(cls, d: dict) -> "Analysis":
        kw = {k: d[k] for k in cls._SNAPSHOT if k in d}
        return cls(results=[], onset_dev_ms=np.array(d.get("onset_dev_ms", [])),
                   duration_ratio=np.array(d.get("duration_ratio", [])), **kw)

    def summary(self) -> dict:
        return {
            "score_notes": self.n_ref,
            "performed_notes": self.n_perf,
            "correct": self.n_correct,
            "wrong_pitch": self.n_wrong_pitch,
            "wrong_octave": self.n_wrong_octave,
            "missing": self.n_missing,
            "uncertain": self.n_uncertain,
            "extra": self.n_extra,
            "unconfirmed_extra": self.n_unconfirmed_extra,
            "chords_total": self.chords_total,
            "chords_correct": self.chords_correct,
            "mean_bpm": self.mean_bpm,
            "reference_bpm": self.ref_bpm,
            "tempo_cv_pct": None if self.tempo_cv is None else self.tempo_cv * 100,
            "pause_count": len(self.pauses),
            "restart_count": len(self.restarts),
            "replayed": self.n_replayed,
            "pedal_events": self.pedal_events if self.pedal_supported else None,
            "pedal_down_share": self.pedal_down_share,
            "onset_dev_median_abs_ms": (
                float(np.median(np.abs(self.onset_dev_ms))) if len(self.onset_dev_ms) else None),
            "velocity_std": self.velocity_std,
            "velocity_flat": self.velocity_flat,
        }


# ---------------------------------------------------------------------------------------------
# timing model


@dataclass
class _Timing:
    pairs: list
    gid: np.ndarray
    seg: np.ndarray
    spb: float
    slopes: np.ndarray
    preds: np.ndarray
    exp_sec: object  # callable(beat) -> seconds, or None
    mean_bpm: float | None
    tempo_cv: float | None
    pauses: list
    beat_lo: float = 0.0
    beat_hi: float = 0.0
    backward_seg: set = field(default_factory=set)  # segments that start or end with a jump back in time


def _segments(gbeat, gsec, cfg, ref_spb):
    """Typical sec/beat, pause/jump boundaries, and the segment id of every onset group."""
    spb = ref_spb
    if len(gbeat) < 2:
        return spb, np.zeros(len(gbeat), dtype=int), np.zeros(0)
    dx, dy = np.diff(gbeat), np.diff(gsec)
    good = (dx > 0) & (dy > 0)
    ratios = dy[good] / dx[good]
    ratios = ratios[ratios > 0.05]
    if len(ratios):
        spb = float(np.median(ratios))
    extra = dy - dx * spb
    seg = np.concatenate([[0], np.cumsum((extra > cfg.pause_threshold_s) | (dy <= 0))])
    return spb, seg, extra


def _local_fit(gbeat, gsec, seg, g, w):
    idx = [j for j in range(max(0, g - w), min(len(gbeat), g + w + 1)) if j != g and seg[j] == seg[g]]
    if len(idx) < 3:
        return None
    slope, intercept = np.polyfit(gbeat[idx], gsec[idx], 1)
    return slope, intercept


def _timing(ref: Reference, perf: list[PerfNote], pairs: list, cfg: AnalysisConfig) -> _Timing:
    ref_spb = 60.0 / ref.bpm
    if len(pairs) < 2:
        return _Timing(pairs, np.zeros(len(pairs), int), np.zeros(0, int), ref_spb,
                       np.zeros(0), np.zeros(0), None, None, None, [])
    rb = np.array([ref.notes["onset_beat"][r] for r, _ in pairs])
    ps = np.array([perf[p].onset for _, p in pairs])
    keys = np.round(rb, 3)
    uniq = np.unique(keys)
    gid = np.searchsorted(uniq, keys)
    gsec = np.array([np.median(ps[gid == g]) for g in range(len(uniq))])
    gbeat = uniq
    spb, seg, extra = _segments(gbeat, gsec, cfg, ref_spb)
    backward_seg = set()
    for k in np.where(np.diff(gsec) <= 0)[0]:  # group k+1 is earlier than group k: a mismatch signature
        backward_seg.update({int(seg[k]), int(seg[k + 1])})

    pauses = []
    for k in np.where(extra > cfg.pause_threshold_s)[0]:
        r_idx = pairs[int(np.where(gid == k + 1)[0][0])][0]
        pauses.append({
            "measure": int(ref.notes["measure"][r_idx]),
            "beat": float(ref.notes["beat_in_measure"][r_idx]),
            "extra_s": float(extra[k]),
            "at_s": float(gsec[k + 1]),
        })

    slopes = np.full(len(gbeat), spb)
    preds = np.full(len(gbeat), np.nan)
    fitted = []
    for g in range(len(gbeat)):
        fit = _local_fit(gbeat, gsec, seg, g, cfg.window_groups)
        if fit is not None and fit[0] > 0.05:
            slopes[g] = fit[0]
            preds[g] = fit[0] * gbeat[g] + fit[1]
            fitted.append(fit[0])
    mean_bpm = tempo_cv = None
    if fitted:
        bpms = 60.0 / np.array(fitted)
        mean_bpm = float(60.0 / np.median(fitted))
        tempo_cv = float(np.std(bpms) / np.mean(bpms))
    elif len(gbeat) >= 2:
        mean_bpm = float(60.0 / spb)

    exp_sec = None
    lo = hi = 0.0
    if len(gbeat) >= 2:
        iso = IsotonicRegression(out_of_bounds="clip").fit(gbeat, gsec)
        lo, hi = float(gbeat[0]), float(gbeat[-1])

        def edge_slope(idx):
            if len(idx) >= 3:
                slope = float(np.polyfit(gbeat[idx], gsec[idx], 1)[0])
                if 0.3 * spb < slope < 3 * spb:
                    return slope
            return spb

        head = edge_slope(list(range(0, min(6, len(gbeat)))))
        tail = edge_slope(list(range(max(0, len(gbeat) - 6), len(gbeat))))

        def exp_sec(beat, _iso=iso, _lo=lo, _hi=hi, _head=head, _tail=tail):  # noqa: E306
            if beat < _lo:
                return float(_iso.predict([_lo])[0] - (_lo - beat) * _head)
            if beat > _hi:
                return float(_iso.predict([_hi])[0] + (beat - _hi) * _tail)
            return float(_iso.predict([beat])[0])

    return _Timing(pairs, gid, seg, spb, slopes, preds, exp_sec, mean_bpm, tempo_cv, pauses,
                   lo, hi, backward_seg)


# ---------------------------------------------------------------------------------------------

def _assign(ref: Reference, perf: list[PerfNote], timing: _Timing, cfg: AnalysisConfig):
    """Globally optimal note assignment given the timing map.

    Each score note can take one performed note: same pitch near its expected time (correct),
    a nearby wrong pitch or octave (wrong), or nothing (missing). Leftover performed notes are
    extra. One assignment problem decides all of these together, so a wrong note, a missing note
    and an extra note compete fairly instead of being resolved greedily one after another.
    """
    n, m = len(ref.notes), len(perf)
    rp = ref.notes["pitch"].astype(int)
    t_exp = np.array([timing.exp_sec(float(b)) for b in ref.notes["onset_beat"]])
    pp = np.array([x.pitch for x in perf], dtype=int)
    po = np.array([x.onset for x in perf])
    # the further a note is from the part of the score we have timing evidence for, the less
    # precisely we know when it should sound, so its window grows with that distance
    b = ref.notes["onset_beat"]
    outside = np.maximum(timing.beat_lo - b, 0) + np.maximum(b - timing.beat_hi, 0)
    unc = (0.12 * outside * timing.spb)[:, None]
    window = max(cfg.wrong_note_window_s, 0.6 * timing.spb) + unc
    tol = max(0.45, cfg.implausible_beats * timing.spb) + unc
    dt = np.abs(po[None, :] - t_exp[:, None])
    window = np.broadcast_to(window, dt.shape)
    tol = np.broadcast_to(tol, dt.shape)
    diff = np.abs(pp[None, :] - rp[:, None])
    big, unmatched_cost = 1e6, 1.2
    cost = np.full((n, m), big)
    same = (diff == 0) & (dt <= tol)
    cost[same] = dt[same] / window[same]
    octave = (diff > 0) & (diff % 12 == 0) & (diff <= 24)
    wrong = (diff > 0) & ((diff <= cfg.max_wrong_interval) | octave) & (dt <= window)
    cost[wrong] = 1.2 + dt[wrong] / window[wrong] + 0.02 * np.minimum(diff[wrong], 12)
    full = np.full((n + m, n + m), big)
    full[:n, :m] = cost
    full[np.arange(n), m + np.arange(n)] = unmatched_cost
    full[n + np.arange(m), np.arange(m)] = unmatched_cost
    full[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(full)
    pairs, wrongs = [], {}
    used_r, used_p = set(), set()
    for r, c in zip(rows, cols):
        if r < n and c < m and cost[r, c] < big:
            used_r.add(int(r))
            used_p.add(int(c))
            if diff[r, c] == 0:
                pairs.append((int(r), int(c)))
            else:
                wrongs[int(r)] = (int(c), bool(octave[r, c]))
    unmatched_ref = [r for r in range(n) if r not in used_r]
    unmatched_perf = [p for p in range(m) if p not in used_p]
    return sorted(pairs), wrongs, unmatched_ref, unmatched_perf


def _local_align_matches(a: list[int], b: list[int]) -> tuple[int, int, int]:
    """Smith-Waterman on pitch sequences: (number of equal pitches in the best local alignment,
    index in `b` where that alignment starts, index in `b` where it ends)."""
    H = np.zeros((len(a) + 1, len(b) + 1))
    M = np.zeros_like(H, dtype=int)  # matches along the best path into each cell
    best, best_ij = 0.0, (0, 0)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            eq = a[i - 1] == b[j - 1]
            options = [(0.0, 0), (H[i - 1, j - 1] + (1.0 if eq else -1.0), M[i - 1, j - 1] + int(eq)),
                       (H[i - 1, j] - 1.0, M[i - 1, j]), (H[i, j - 1] - 1.0, M[i, j - 1])]
            H[i, j], M[i, j] = max(options, key=lambda o: o[0])
            if H[i, j] > best:
                best, best_ij = H[i, j], (i, j)
    i, j = best_ij
    end = j
    while i > 0 and j > 0 and H[i, j] > 0:  # walk back to the start of the local alignment
        if H[i, j] == H[i - 1, j - 1] + (1.0 if a[i - 1] == b[j - 1] else -1.0):
            i, j = i - 1, j - 1
        elif H[i, j] == H[i - 1, j] - 1.0:
            i -= 1
        else:
            j -= 1
    return int(M[best_ij]), j, end


def _find_restarts(ref: Reference, perf: list[PerfNote], extra_ps: list[int], spb: float,
                   cfg: AnalysisConfig) -> list[tuple[list[int], int]]:
    """Runs of unexplained notes that replay a stretch of the score: the student stopped and
    started again. Returns (performed note indices, score row where the replayed stretch starts)."""
    if len(extra_ps) < cfg.min_restart_notes:
        return []
    order = sorted(extra_ps, key=lambda p: (perf[p].onset, perf[p].pitch))
    runs, cur = [], [order[0]]
    for p in order[1:]:
        if perf[p].onset - perf[cur[-1]].onset <= max(1.5, 2.5 * spb):
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    score_pitches = [int(x) for x in ref.notes["pitch"]]
    found = []
    for run in runs:
        if len(run) < cfg.min_restart_notes:
            continue
        # order like the score: onset groups (chords), low to high pitch inside a chord
        ordered, group, last = [], [], None
        for p in run:
            if last is not None and perf[p].onset - last > 0.09:
                ordered += sorted(group, key=lambda q: perf[q].pitch)
                group = []
            group.append(p)
            last = perf[p].onset
        ordered += sorted(group, key=lambda q: perf[q].pitch)
        matches, start, end = _local_align_matches([perf[p].pitch for p in ordered], score_pitches)
        if matches >= cfg.min_restart_notes and matches >= cfg.restart_match_share * len(run):
            found.append((run, start, end))
    # runs that replay overlapping stretches of the score belong to one restart (the matcher may
    # have interleaved the two attempts)
    merged: list[list] = []
    for run, start, end in sorted(found, key=lambda x: x[1]):
        if merged and start <= merged[-1][2] + 4:
            merged[-1] = [merged[-1][0] + run, merged[-1][1], max(merged[-1][2], end)]
        else:
            merged.append([run, start, end])
    return [(run, start) for run, start, _ in merged]


def analyze(ref: Reference, perf: list[PerfNote], alignment: Alignment,
            cfg: AnalysisConfig | None = None) -> Analysis:
    cfg = cfg or AnalysisConfig()
    n_ref, n_perf = len(ref.notes), len(perf)
    notes_log: list[str] = []
    pairs = list(alignment.pairs)
    unmatched_ref = list(alignment.unmatched_ref)
    unmatched_perf = list(alignment.unmatched_perf)
    beats = ref.notes["onset_beat"]

    # The matcher can pair a note with an equal pitch that is far from where it should be
    # (typically a neighbouring repeated note). Demote such matches, then re-pair same-pitch
    # notes that are close in time, and repeat until stable.
    demoted_total = 0
    timing = _timing(ref, perf, pairs, cfg)
    for _ in range(4):
        if len(pairs) < 2:
            break
        tol = max(0.45, cfg.implausible_beats * timing.spb)
        demote = set()
        sizes = np.bincount(timing.seg)
        n_groups = len(timing.seg)
        for i in range(len(pairs)):
            g = timing.gid[i]
            # a tiny segment cut off by a jump BACK in time is a mismatch; one cut off only by
            # pauses (a slow start, a restart) is real playing
            island = (n_groups > 8 and len(sizes) > 1 and sizes.max() > 2 and sizes[timing.seg[g]] <= 2
                      and int(timing.seg[g]) in timing.backward_seg)
            far = (not np.isnan(timing.preds[g])
                   and abs(perf[pairs[i][1]].onset - timing.preds[g]) > tol)
            if island or far:
                demote.add(i)
        changed = False
        if demote and len(demote) <= 0.4 * len(pairs):
            for i in demote:
                unmatched_ref.append(pairs[i][0])
                unmatched_perf.append(pairs[i][1])
            pairs = [pr for i, pr in enumerate(pairs) if i not in demote]
            demoted_total += len(demote)
            changed = True
            timing = _timing(ref, perf, pairs, cfg)
        if timing.exp_sec is not None and unmatched_ref and unmatched_perf:
            window = max(0.30, 0.5 * timing.spb)
            cands = []
            for r in unmatched_ref:
                t_exp = timing.exp_sec(float(beats[r]))
                for p in unmatched_perf:
                    if perf[p].pitch == int(ref.notes["pitch"][r]):
                        dt = abs(perf[p].onset - t_exp)
                        if dt <= window:
                            cands.append((dt, r, p))
            cands.sort()
            used_r, used_p = set(), set()
            for dt, r, p in cands:
                if r not in used_r and p not in used_p:
                    used_r.add(r)
                    used_p.add(p)
                    pairs.append((r, p))
            if used_r:
                unmatched_ref = [r for r in unmatched_ref if r not in used_r]
                unmatched_perf = [p for p in unmatched_perf if p not in used_p]
                pairs.sort()
                changed = True
                timing = _timing(ref, perf, pairs, cfg)
        if not changed:
            break
    if demoted_total:
        notes_log.append(f"{demoted_total}_implausible_matches_demoted")
    if len(pairs) < 2:
        notes_log.append("too_few_matched_notes_for_timing")

    # Stage B: re-decide every note with a global assignment, then refresh the timing map and
    # repeat once so the expected times are computed from the final matches.
    statuses: dict[int, NoteResult] = {}
    used_perf: set[int] = set()
    if timing.exp_sec is not None:
        for _ in range(6):
            new_pairs, wrongs, unmatched_ref, unmatched_perf = _assign(ref, perf, timing, cfg)
            same_as_before = new_pairs == sorted(pairs)
            pairs = new_pairs
            timing_new = _timing(ref, perf, pairs, cfg)
            if timing_new.exp_sec is None:
                break
            timing = timing_new
            if same_as_before:
                break
        new_pairs, wrongs, unmatched_ref, unmatched_perf = _assign(ref, perf, timing, cfg)
        if new_pairs != sorted(pairs):
            pairs = new_pairs
            timing = _timing(ref, perf, pairs, cfg) if len(pairs) >= 2 else timing
        for r, (p, octave) in wrongs.items():
            used_perf.add(p)
            statuses[r] = NoteResult(
                "wrong_octave" if octave else "wrong_pitch", r, p, perf_pitch=perf[p].pitch,
                perf_onset_s=perf[p].onset)
        unmatched_ref = [r for r in unmatched_ref if r not in wrongs]
        unmatched_perf = [p for p in unmatched_perf if p not in used_perf]

    onset_dev: dict[int, float] = {}
    dur_ratio: dict[int, float] = {}
    for i, (r, p) in enumerate(pairs):
        if len(pairs) < 2 or i >= len(timing.gid):
            break
        g = timing.gid[i]
        if not np.isnan(timing.preds[g]):
            onset_dev[p] = float((perf[p].onset - timing.preds[g]) * 1000.0)
        dref = float(ref.notes["duration_beat"][r]) * timing.slopes[g]
        if dref > 0:
            dur_ratio[p] = float((perf[p].offset - perf[p].onset) / dref)

    exp_sec = timing.exp_sec

    # --- optional (grace) notes: a performed note that matches one is not "extra"
    optional_hit: set[int] = set()
    if len(ref.optional) and exp_sec is not None:
        for p in unmatched_perf:
            if p in used_perf:
                continue
            for o in ref.optional:
                if o["pitch"] == perf[p].pitch and abs(
                        perf[p].onset - exp_sec(float(o["onset_beat"]))) < 0.4:
                    optional_hit.add(p)
                    break

    # --- assemble per-note results
    results: list[NoteResult] = []
    for r, p in pairs:
        results.append(NoteResult(
            "correct", r, p, perf_pitch=perf[p].pitch, perf_onset_s=perf[p].onset,
            onset_dev_ms=onset_dev.get(p), duration_ratio=dur_ratio.get(p)))
    for r in unmatched_ref:
        results.append(NoteResult(
            "missing", r,
            expected_onset_s=exp_sec(float(beats[r])) if exp_sec is not None else None))
    for r, note in statuses.items():
        results.append(note)
    for p in unmatched_perf:
        if p in used_perf or p in optional_hit:
            continue
        results.append(NoteResult("extra", None, p, perf_pitch=perf[p].pitch,
                                  perf_onset_s=perf[p].onset))
    for res in results:
        if res.ref_index is not None:
            r = res.ref_index
            res.measure = int(ref.notes["measure"][r])
            res.beat_in_measure = float(ref.notes["beat_in_measure"][r])
            res.ref_pitch = int(ref.notes["pitch"][r])
    results.sort(key=lambda x: (
        beats[x.ref_index] if x.ref_index is not None else 1e9, x.perf_onset_s or 0.0))

    restarts = []
    extra_ps = [x.perf_index for x in results if x.status == "extra"]
    for run, start in _find_restarts(ref, perf, extra_ps, timing.spb, cfg):
        in_run = set(run)
        for x in results:
            if x.status == "extra" and x.perf_index in in_run:
                x.status = "replayed"
        restarts.append({"at_s": round(min(perf[p].onset for p in run), 2), "notes": len(run),
                         "from_measure": int(ref.notes["measure"][min(start, n_ref - 1)])})
    pauses = list(timing.pauses)
    if restarts:
        notes_log.append(f"{len(restarts)}_restart(s)_detected")
        # the silence around a restart is explained by the restart: do not report it twice
        starts = [r["at_s"] for r in restarts]
        pauses = [p for p in pauses if not any(p["at_s"] - p["extra_s"] - 3.0 <= t <= p["at_s"] for t in starts)]

    counts = {s: sum(1 for x in results if x.status == s)
              for s in ("correct", "wrong_pitch", "wrong_octave", "missing", "extra", "replayed")}

    # --- chords: score onset groups of >= 2 notes; correct when every tone is correct
    ok_ref = {x.ref_index for x in results if x.status == "correct"}
    keys = np.round(beats, 3)
    chords_total = chords_correct = 0
    for k in np.unique(keys):
        members = np.where(keys == k)[0]
        if len(members) >= 2:
            chords_total += 1
            chords_correct += int(all(int(m) in ok_ref for m in members))

    matched_vel = np.array([perf[p].velocity for _, p in pairs], dtype=float)
    vstd = float(np.std(matched_vel)) if len(matched_vel) else None
    vrange = float(np.ptp(matched_vel)) if len(matched_vel) else None

    return Analysis(
        results=results, n_ref=n_ref, n_perf=n_perf,
        n_correct=counts["correct"], n_wrong_pitch=counts["wrong_pitch"],
        n_wrong_octave=counts["wrong_octave"], n_missing=counts["missing"],
        n_extra=counts["extra"], n_optional_played=len(optional_hit),
        chords_total=chords_total, chords_correct=chords_correct,
        onset_dev_ms=np.array(list(onset_dev.values())),
        duration_ratio=np.array(list(dur_ratio.values())),
        ref_bpm=ref.bpm, mean_bpm=timing.mean_bpm, tempo_cv=timing.tempo_cv,
        pauses=pauses, velocity_std=vstd, velocity_range=vrange,
        velocity_flat=(vstd is None or vstd < cfg.flat_velocity_std), notes=notes_log,
        n_replayed=counts["replayed"], restarts=restarts, beat_to_sec=exp_sec,
    )
