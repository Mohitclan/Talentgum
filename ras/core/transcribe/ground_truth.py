"""A 'transcriber' that returns known notes. Isolates downstream stages from model error.

Used by tests and by the benchmark to measure the upper bound of the alignment and
error-detection stages when transcription is perfect.
"""

from __future__ import annotations

import numpy as np

from ras.types import PerfNote, Transcription


class GroundTruthTranscriber:
    name = "ground-truth"
    sample_rate = 44100
    version = "ground-truth"

    def __init__(self, notes: list[PerfNote] | None = None):
        self.notes = notes or []

    def set_notes(self, notes: list[PerfNote]) -> None:
        self.notes = notes

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        return Transcription(
            notes=[PerfNote(n.pitch, n.onset, n.offset, n.velocity) for n in self.notes],
            transcriber=self.name, version=self.version,
        )
