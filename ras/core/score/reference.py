"""Load the reference (MusicXML/MXL primary, MIDI alternative) into one note table."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REF_DTYPE = [
    ("id", "U16"),
    ("pitch", "i4"),
    ("onset_beat", "f8"),  # quarter-note units
    ("duration_beat", "f8"),
    ("onset_sec", "f8"),
    ("duration_sec", "f8"),
    ("measure", "i4"),
    ("beat_in_measure", "f8"),  # 1-based, quarter-note units
    ("voice", "i4"),
]
DEFAULT_BPM = 100.0


@dataclass
class Reference:
    notes: np.ndarray  # REF_DTYPE, sorted by onset then pitch; excludes optional notes
    optional: np.ndarray  # same dtype: grace notes etc. (never penalised, never required)
    bpm: float
    sha256: str
    source: str
    n_measures: int
    warnings: list[str] = field(default_factory=list)

    @property
    def n_notes(self) -> int:
        return len(self.notes)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finish(rows, bpm, sha, source, warns) -> Reference:
    arr = np.array(rows, dtype=REF_DTYPE) if rows else np.zeros(0, dtype=REF_DTYPE)
    optional_mask = arr["duration_beat"] <= 1e-6
    notes, optional = arr[~optional_mask], arr[optional_mask]
    order = np.lexsort((notes["pitch"], notes["onset_beat"]))
    notes = notes[order]
    notes["id"] = [f"s{i}" for i in range(len(notes))]
    if len(optional):
        optional["id"] = [f"o{i}" for i in range(len(optional))]
    n_measures = int(notes["measure"].max()) if len(notes) else 0
    return Reference(notes, optional, bpm, sha, source, n_measures, warns)


def _quarter_bpm(tempo) -> float:
    """Tempo in quarter notes per minute, whatever note value the score's metronome mark uses."""
    mpq = getattr(tempo, "microseconds_per_quarter", None)
    return 60e6 / mpq if mpq else float(tempo.bpm)


def load_musicxml(path: str | Path, unfold_repeats: bool = True, bpm: float | None = None) -> Reference:
    import partitura as pt

    path = Path(path)
    warns: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(path))
        part = pt.score.merge_parts(score.parts) if len(score.parts) > 1 else score.parts[0]
        if unfold_repeats:
            try:
                part = pt.score.unfold_part_maximal(part)
            except Exception as exc:  # unusual repeat structures: keep the written order
                warns.append(f"repeat_unfold_failed: {exc}")
        na = part.note_array(include_metrical_position=True)
        tempi = list(part.iter_all(pt.score.Tempo))
        measures = part.measure_number_map(na["onset_div"]) if len(na) else []

    # tempo map: (start in quarters, seconds per quarter), applied piecewise to every note
    if bpm is not None or not tempi:
        if bpm is None:
            warns.append(f"no_tempo_in_score_using_{DEFAULT_BPM:g}_bpm")
            bpm = DEFAULT_BPM
        segments = [(0.0, 60.0 / bpm)]
    else:
        changes = sorted({float(part.quarter_map(t.start.t)): 60.0 / _quarter_bpm(t) for t in tempi}.items())
        segments = [(0.0, changes[0][1])] + [c for c in changes if c[0] > 0]
        bpm = 60.0 / segments[0][1]
        if len(segments) > 1:
            warns.append(f"{len(segments) - 1}_tempo_change(s)_applied")

    def to_sec(q: float) -> float:
        total = 0.0
        for k, (q0, spq) in enumerate(segments):
            q1 = segments[k + 1][0] if k + 1 < len(segments) else float("inf")
            if q <= q0:
                break
            total += (min(q, q1) - q0) * spq
        return total

    with_dur = na[na["duration_quarter"] > 0]
    divs = float(np.median(with_dur["duration_div"] / with_dur["duration_quarter"])) if len(with_dur) else 1.0

    rows = []
    for i, n in enumerate(na):
        on, du = float(n["onset_quarter"]), float(n["duration_quarter"])
        rows.append((
            "", int(n["pitch"]), on, du, to_sec(on), to_sec(on + du) - to_sec(on),
            int(measures[i]), float(n["rel_onset_div"]) / divs + 1.0, int(n["voice"]),
        ))
    ref = _finish(rows, bpm, _sha256(path), "musicxml", warns)
    if len(ref.optional):
        ref.warnings.append(f"{len(ref.optional)}_optional_notes_ignored")
    return ref


def load_midi(path: str | Path, bpm: float | None = None) -> Reference:
    import pretty_midi

    path = Path(path)
    pm = pretty_midi.PrettyMIDI(str(path))
    warns: list[str] = []
    tempi = pm.get_tempo_changes()[1]
    if bpm is None:
        bpm = float(tempi[0]) if len(tempi) else DEFAULT_BPM
    if len(tempi) > 1:
        warns.append(f"{len(tempi) - 1}_tempo_change(s)_applied")
    downbeats = pm.get_downbeats()

    def beats(t: float) -> float:  # quarter notes from ticks, so tempo changes are respected
        return pm.time_to_tick(t) / pm.resolution

    notes = sorted((n for inst in pm.instruments if not inst.is_drum for n in inst.notes),
                   key=lambda n: (n.start, n.pitch))
    rows = []
    for n in notes:
        m = int(np.searchsorted(downbeats, n.start + 1e-6, side="right"))
        m_start = downbeats[m - 1] if m >= 1 else 0.0
        rows.append((
            "", n.pitch, beats(n.start), beats(n.end) - beats(n.start), n.start, n.end - n.start,
            max(m, 1), beats(n.start) - beats(m_start) + 1.0, 1,
        ))
    return _finish(rows, bpm, _sha256(path), "midi", warns)


def load_reference(path: str | Path, unfold_repeats: bool = True, bpm: float | None = None) -> Reference:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".mid", ".midi"):
        return load_midi(path, bpm=bpm)
    if suffix in (".xml", ".musicxml", ".mxl"):
        return load_musicxml(path, unfold_repeats=unfold_repeats, bpm=bpm)
    raise ValueError(f"unsupported reference format: {suffix}")
