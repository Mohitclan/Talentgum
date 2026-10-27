"""Transkun V2 (MIT) piano transcription, on CPU or a CUDA GPU (`RAS_DEVICE`, default auto)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from ras.types import PedalEvent, PerfNote, Transcription

from .base import TranscriberUnavailable, clean_notes


class TranskunTranscriber:
    name = "transkun"

    def __init__(self, device: str | None = None, weight: str | None = None, conf: str | None = None,
                 num_threads: int | None = None):
        try:
            import torch
            import transkun
        except ImportError as exc:  # pragma: no cover
            raise TranscriberUnavailable("install the extra: pip install -e '.[transkun]'") from exc
        pre = Path(transkun.__file__).parent / "pretrained"
        weight = weight or os.environ.get("RAS_TRANSKUN_WEIGHT")  # a fine-tuned model (ras finetune export)
        self._weight = Path(weight) if weight else pre / "2.0.pt"
        self._conf = Path(conf) if conf else pre / "2.0.conf"
        device = device or os.environ.get("RAS_DEVICE", "auto")  # cpu | cuda | cuda:N | auto
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise TranscriberUnavailable(f"RAS_DEVICE={device} but no CUDA GPU is visible to PyTorch")
        self.device = self._device = device
        self._num_threads = num_threads
        self._model = None
        self.sample_rate = 44100
        self.version = f"transkun-{self._weight.name}-{self._sha(self._weight)}"

    @staticmethod
    def _sha(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]

    def _load(self):
        import moduleconf
        import torch

        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        torch.set_grad_enabled(False)
        manager = moduleconf.parseFromFile(str(self._conf))
        model = manager["Model"].module.TransKun(conf=manager["Model"].config).to(self._device)
        ckpt = torch.load(str(self._weight), map_location=self._device, weights_only=True)
        key = "best_state_dict" if "best_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[key], strict=False)
        model.eval()
        self._model = model
        self.sample_rate = int(model.fs)

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        import torch

        if self._model is None:
            self._load()
        if sr != self.sample_rate:
            import soxr

            audio = soxr.resample(audio, sr, self.sample_rate)
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).reshape(-1, 1)
        events = self._model.transcribe(x.to(self._device), discardSecondHalf=False)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()  # so elapsed_s is the real GPU time
        notes, pedal = [], []
        for e in events:
            if e.pitch > 0:
                notes.append(PerfNote(int(e.pitch), float(e.start), float(e.end), int(e.velocity)))
            elif -e.pitch == 64:  # sustain pedal control change
                pedal.append(PedalEvent(float(e.start), float(e.end), int(e.velocity)))
        return Transcription(
            notes=clean_notes(notes), pedal=pedal, transcriber=self.name, version=self.version,
            elapsed_s=time.perf_counter() - t0, supports_pedal=True, supports_velocity=True,
        )
