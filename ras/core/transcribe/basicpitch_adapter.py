"""Basic Pitch (Spotify, Apache-2.0): the cheap lower-bound baseline for the benchmark."""

from __future__ import annotations

import tempfile
import time

import numpy as np
import soundfile as sf

from ras.types import PerfNote, Transcription

from .base import TranscriberUnavailable, clean_notes


class BasicPitchTranscriber:
    name = "basic-pitch"
    sample_rate = 22050
    version = "basic-pitch"

    def __init__(self):
        try:
            import basic_pitch  # noqa: F401
        except ImportError as exc:
            raise TranscriberUnavailable("pip install basic-pitch to benchmark this model") from exc

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        from basic_pitch.inference import predict

        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr)
            _, midi, _ = predict(tmp.name)
        notes = [PerfNote(n.pitch, n.start, n.end, n.velocity)
                 for inst in midi.instruments for n in inst.notes]
        return Transcription(
            notes=clean_notes(notes), transcriber=self.name, version=self.version,
            elapsed_s=time.perf_counter() - t0, supports_velocity=False,
        )
