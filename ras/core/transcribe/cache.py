"""Disk cache for transcriptions, keyed by audio content and model version.

Transcription is the slow step. Caching it means scoring rules can change and recordings can be
re-assessed without re-running the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ras.types import PedalEvent, PerfNote, Transcription


class CachedTranscriber:
    def __init__(self, inner, cache_dir: str | Path):
        self.inner = inner
        self.name = inner.name
        self.sample_rate = getattr(inner, "sample_rate", 44100)
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0

    def _key(self, audio: np.ndarray, sr: int) -> Path:
        h = hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        h.update(f"{sr}|{self.name}|{getattr(self.inner, 'version', '')}".encode())
        return self.dir / f"{h.hexdigest()[:32]}.json"

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        path = self._key(audio, sr)
        if path.exists():
            d = json.loads(path.read_text())
            self.hits += 1
            return Transcription(
                notes=[PerfNote(**n) for n in d["notes"]], pedal=[PedalEvent(**p) for p in d["pedal"]],
                transcriber=d["transcriber"], version=d["version"], elapsed_s=d["elapsed_s"],
                supports_pedal=d["supports_pedal"], supports_velocity=d["supports_velocity"])
        out = self.inner.transcribe(audio, sr)
        path.write_text(json.dumps({
            "notes": [vars(n) for n in out.notes], "pedal": [vars(p) for p in out.pedal],
            "transcriber": out.transcriber, "version": out.version, "elapsed_s": out.elapsed_s,
            "supports_pedal": out.supports_pedal, "supports_velocity": out.supports_velocity}))
        return out
