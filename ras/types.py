"""Plain data types shared by all stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PIANO_MIN, PIANO_MAX = 21, 108  # A0..C8


@dataclass
class PerfNote:
    pitch: int
    onset: float
    offset: float
    velocity: int = 64


@dataclass
class PedalEvent:
    start: float
    end: float
    value: int = 127


@dataclass
class Transcription:
    notes: list[PerfNote]
    pedal: list[PedalEvent] = field(default_factory=list)
    transcriber: str = ""
    version: str = ""
    elapsed_s: float = 0.0
    supports_pedal: bool = False
    supports_velocity: bool = True


PERF_DTYPE = [
    ("pitch", "i4"),
    ("onset_sec", "f8"),
    ("duration_sec", "f8"),
    ("velocity", "i4"),
    ("track", "i4"),
    ("channel", "i4"),
    ("id", "U16"),
]


def perf_array(notes: list[PerfNote]) -> np.ndarray:
    """Structured array in the layout parangonar expects for a performance."""
    arr = np.zeros(len(notes), dtype=PERF_DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (n.pitch, n.onset, max(n.offset - n.onset, 1e-3), n.velocity, 0, 0, f"p{i}")
    return arr
