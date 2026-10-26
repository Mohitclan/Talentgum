"""Transcriber interface: audio in, note events out. Everything downstream depends only on this."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ras.types import PIANO_MAX, PIANO_MIN, PerfNote, Transcription


class TranscriberUnavailable(RuntimeError):
    """The model or its dependency is not installed."""


class Transcriber(Protocol):
    name: str
    sample_rate: int

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription: ...


def clean_notes(notes: list[PerfNote], min_duration_s: float = 0.03) -> list[PerfNote]:
    """Normalise raw model output: keep the piano range, drop blips, sort."""
    out = [
        n for n in notes
        if PIANO_MIN <= n.pitch <= PIANO_MAX and (n.offset - n.onset) >= min_duration_s
    ]
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def list_transcribers() -> list[str]:
    return ["transkun", "basic-pitch", "ground-truth"]


def get_transcriber(name: str, **kwargs) -> Transcriber:
    if name == "transkun":
        from .transkun_adapter import TranskunTranscriber

        return TranskunTranscriber(**kwargs)
    if name == "basic-pitch":
        from .basicpitch_adapter import BasicPitchTranscriber

        return BasicPitchTranscriber(**kwargs)
    if name == "ground-truth":
        from .ground_truth import GroundTruthTranscriber

        return GroundTruthTranscriber(**kwargs)
    raise ValueError(f"unknown transcriber {name!r}; available: {list_transcribers()}")
