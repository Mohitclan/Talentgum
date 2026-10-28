"""Find the song attempts in the notes transcribed from a long class recording.

Pure functions on note lists, so they are testable without audio:
1. drop isolated notes (talking and noise give few, scattered notes);
2. group the rest into playing regions separated by long silences;
3. in each region, find the best-matching stretch of any of the songs, mark it as an attempt, remove
   it and repeat, so repeated and abandoned attempts each become their own clip.

A song is matched on the steps between its melody notes (the highest note at each onset), not on exact
pitches: a wrong note breaks only a step or two, a transposed song still matches, and a student who plays
only the right hand matches as well as one who plays both. Playing the left hand alone does not match.
All thresholds are starting values (docs/CLASS_RECORDINGS.md), to be tuned on labelled classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ras.core.score.reference import Reference
from ras.types import PerfNote

from .align import LocalAlignment, local_align


@dataclass
class SegmentConfig:
    neighbour_s: float = 4.0  # a note needs `min_neighbours` other notes within this many seconds...
    min_neighbours: int = 3  # ...or it is treated as noise
    region_gap_s: float = 6.0  # silence that ends a playing region
    min_region_notes: int = 8
    group_s: float = 0.07  # onsets this close are one chord
    interval_clip: int = 12  # steps larger than an octave are clipped
    match: int = 2
    mismatch: int = -1
    gap: int = -1
    min_matches: int = 8  # matched steps for an attempt to count
    min_precision: float = 0.6  # matched steps / aligned columns
    review_matches: int = 12  # below this, a teacher should confirm
    review_precision: float = 0.75
    review_margin: float = 0.3  # best song's matches must beat the runner-up's by this share
    full_coverage: float = 0.85  # share of the song's steps an attempt covers to count as a full attempt
    clip_pad_s: float = 2.0
    max_clip_s: float = 300.0  # the assessment's limit


@dataclass
class Melody:
    """The highest note at each onset."""

    pitch: np.ndarray
    onset: np.ndarray
    offset: np.ndarray

    def __len__(self) -> int:
        return len(self.pitch)


@dataclass
class SongModel:
    song_id: str
    pitch: np.ndarray
    measure: np.ndarray
    intervals: np.ndarray


def intervals_of(pitch: np.ndarray, clip: int = 12) -> np.ndarray:
    return np.clip(np.diff(pitch.astype(np.int32)), -clip, clip).astype(np.int16)


def song_model(song_id: str, ref: Reference, cfg: SegmentConfig | None = None) -> SongModel:
    cfg = cfg or SegmentConfig()
    keys = np.round(ref.notes["onset_beat"], 3)
    pitch, measure = [], []
    for k in np.unique(keys):
        rows = np.flatnonzero(keys == k)
        pitch.append(int(ref.notes["pitch"][rows].max()))
        measure.append(int(ref.notes["measure"][rows[0]]))
    p = np.array(pitch)
    return SongModel(song_id, p, np.array(measure), intervals_of(p, cfg.interval_clip))


def melody_of_notes(notes: list[PerfNote], group_s: float = 0.07) -> Melody:
    pitch, onset, offset = [], [], []
    group: list[PerfNote] = []
    start = 0.0

    def flush():
        pitch.append(max(n.pitch for n in group))
        onset.append(min(n.onset for n in group))
        offset.append(max(n.offset for n in group))

    for n in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        if group and n.onset - start > group_s:
            flush()
            group = []
        if not group:
            start = n.onset
        group.append(n)
    if group:
        flush()
    return Melody(np.array(pitch, dtype=np.int32), np.array(onset), np.array(offset))


def drop_isolated(notes: list[PerfNote], cfg: SegmentConfig) -> list[PerfNote]:
    ns = sorted(notes, key=lambda n: (n.onset, n.pitch))
    if not ns:
        return ns
    t = np.array([n.onset for n in ns])
    near = np.searchsorted(t, t + cfg.neighbour_s, "right") - np.searchsorted(t, t - cfg.neighbour_s, "left") - 1
    return [n for n, k in zip(ns, near) if k >= cfg.min_neighbours]


def playing_regions(notes: list[PerfNote], cfg: SegmentConfig) -> list[list[PerfNote]]:
    kept = drop_isolated(notes, cfg)
    runs: list[list[PerfNote]] = []
    for n in kept:
        if runs and n.onset - runs[-1][-1].onset <= cfg.region_gap_s:
            runs[-1].append(n)
        else:
            runs.append([n])
    return [r for r in runs if len(r) >= cfg.min_region_notes]


def _align(iv: np.ndarray, song: SongModel, cfg: SegmentConfig) -> LocalAlignment:
    return local_align(iv, song.intervals, cfg.match, cfg.mismatch, cfg.gap)


def _valid(al: LocalAlignment, cfg: SegmentConfig) -> bool:
    return al.matches >= cfg.min_matches and al.precision >= cfg.min_precision


def _attempts(mel: Melody, songs: list[SongModel], cfg: SegmentConfig) -> list[dict]:
    """Repeatedly take the best-matching stretch of the melody, remove it, and look again."""
    if len(mel) < cfg.min_matches + 1:
        return []
    iv_all = intervals_of(mel.pitch, cfg.interval_clip)
    segments = [(0, len(mel))]  # note index ranges [s, e) not yet explained
    cache: dict[tuple[int, int], tuple[SongModel, LocalAlignment] | None] = {}
    found = []

    def best(seg):
        if seg not in cache:
            s, e = seg
            cand = None
            if e - s >= cfg.min_matches + 1:
                for song in songs:
                    al = _align(iv_all[s:e - 1], song, cfg)
                    if cand is None or (al.matches, al.precision) > (cand[1].matches, cand[1].precision):
                        cand = (song, al)
            cache[seg] = cand
        return cache[seg]

    while True:
        options = [(seg, best(seg)) for seg in segments]
        options = [(seg, c) for seg, c in options if c and _valid(c[1], cfg)]
        if not options:
            break
        seg, (song, al) = max(options, key=lambda o: (o[1][1].matches, o[1][1].precision))
        s, e = seg
        first, last = s + al.a0, s + al.a1  # melody notes of the attempt, inclusive
        segments.remove(seg)
        segments += [(s, first), (last + 1, e)]
        found.append(_describe(mel, iv_all, songs, song, al, first, last, cfg))
    found.sort(key=lambda c: c["start_s"])
    return found


def _describe(mel, iv_all, songs, song, al, first, last, cfg) -> dict:
    # how clearly this song beats the others on exactly this stretch
    span = iv_all[first:last]
    rivals = [(_align(span, o, cfg).matches, o.song_id) for o in songs if o is not song]
    runner_up, runner_id = max(rivals, default=(0, None))
    margin = (al.matches - runner_up) / al.matches
    last_song_note = min(al.b1, len(song.pitch) - 1)
    coverage = (al.b1 - al.b0) / max(len(song.intervals), 1)
    reasons = []
    if al.matches < cfg.review_matches:
        reasons.append("few_notes_matched")
    if al.precision < cfg.review_precision:
        reasons.append("many_wrong_steps")
    if margin < cfg.review_margin:
        reasons.append(f"could_be_{runner_id}")
    return {
        "song_id": song.song_id,
        "start_s": round(float(mel.onset[first]), 2),
        "end_s": round(float(mel.offset[last]), 2),
        "match_score": round(al.precision, 3),
        "matched_steps": al.matches,
        "song_steps": int(len(song.intervals)),
        "coverage": "full" if coverage >= cfg.full_coverage else "partial",
        "coverage_pct": round(100 * coverage, 1),
        "song_bars": [int(song.measure[min(al.b0, len(song.pitch) - 1)]), int(song.measure[last_song_note])],
        "margin": round(float(margin), 3),
        "status": "needs_review" if reasons else "auto",
        "review_reasons": reasons,
        "reviewer": None,
    }


def find_clips(notes: list[PerfNote], songs: list[SongModel], cfg: SegmentConfig | None = None) -> dict:
    """Playing regions and song attempts in one class recording's transcribed notes."""
    cfg = cfg or SegmentConfig()
    regions, clips = [], []
    for run in playing_regions(notes, cfg):
        attempts = _attempts(melody_of_notes(run, cfg.group_s), songs, cfg)
        regions.append({"start_s": round(min(n.onset for n in run), 2),
                        "end_s": round(max(n.offset for n in run), 2),
                        "notes": len(run), "clips": len(attempts)})
        clips += attempts
    clips.sort(key=lambda c: c["start_s"])
    return {"playing_regions": regions, "clips": clips, "config": asdict(cfg)}
