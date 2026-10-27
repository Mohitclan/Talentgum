"""Kong et al. high-resolution piano transcription with pedals (ByteDance, MIT): pedal and velocity cross-check.

Uses the model and post-processor of the `piano_transcription_inference` package (the 'models' extra)
but not its `PianoTranscription` class, which fully unpickles the checkpoint and shells out to wget.
Weights come from `ras models download kong` and are loaded with weights_only=True.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ras.types import PedalEvent, PerfNote, Transcription

from .base import TranscriberUnavailable, clean_notes

SR, FPS, N_NOTE = 16000, 100, 88
SEGMENT = SR * 10


class KongTranscriber:
    name = "kong"

    def __init__(self, device: str | None = None, weights: str | None = None, batch: int = 4):
        try:
            import piano_transcription_inference  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise TranscriberUnavailable("install the extra: pip install -e '.[models]'") from exc
        from ras.models import is_available, model_path

        if weights is None and not is_available("kong"):
            raise TranscriberUnavailable("Kong weights missing: run `ras models download kong`")
        self._weights = weights or str(model_path("kong"))
        self._device = device or os.environ.get("RAS_DEVICE", "auto")
        self._batch = batch
        self._model = None
        self.sample_rate = SR
        self.version = "kong-note_pedal-crnn-0.9677"

    def _load(self):
        import torch
        from piano_transcription_inference.models import Note_pedal

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Note_pedal(frames_per_second=FPS, classes_num=N_NOTE)
        ckpt = torch.load(self._weights, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=False)
        self._model = model.to(self._device).eval()
        self.device = self._device

    @staticmethod
    def _enframe(x: np.ndarray) -> np.ndarray:
        """Half-overlapping 10 s segments, as upstream PianoTranscription.enframe."""
        pad = int(np.ceil(len(x) / SEGMENT)) * SEGMENT - len(x)
        x = np.concatenate([x, np.zeros(pad, np.float32)])
        return np.stack([x[p:p + SEGMENT] for p in range(0, len(x) - SEGMENT + 1, SEGMENT // 2)])

    @staticmethod
    def _deframe(x: np.ndarray) -> np.ndarray:
        """Keep the middle half of each overlapping segment (upstream PianoTranscription.deframe)."""
        if x.shape[0] == 1:
            return x[0]
        x = x[:, :-1, :]
        n = x.shape[1]
        parts = [x[0, : int(n * 0.75)]] + [x[i, int(n * 0.25): int(n * 0.75)] for i in range(1, len(x) - 1)]
        return np.concatenate(parts + [x[-1, int(n * 0.25):]], axis=0)

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        import torch
        from piano_transcription_inference.utilities import RegressionPostProcessor

        if self._model is None:
            self._load()
        if sr != SR:
            import soxr

            audio = soxr.resample(audio, sr, SR)
        t0 = time.perf_counter()
        n_frames = int(len(audio) / SR * FPS)
        segments = self._enframe(np.asarray(audio, np.float32))
        out: dict[str, list] = {}
        with torch.no_grad():
            for b in range(0, len(segments), self._batch):
                res = self._model(torch.from_numpy(segments[b:b + self._batch]).to(self._device))
                for k, v in res.items():
                    out.setdefault(k, []).append(v.cpu().numpy())
        output = {k: self._deframe(np.concatenate(v))[:n_frames] for k, v in out.items()}
        post = RegressionPostProcessor(FPS, classes_num=N_NOTE, onset_threshold=0.3, offset_threshold=0.3,
                                       frame_threshold=0.1, pedal_offset_threshold=0.2)
        note_events, pedal_events = post.output_dict_to_midi_events(output)
        notes = [PerfNote(int(e["midi_note"]), float(e["onset_time"]), float(e["offset_time"]), int(e["velocity"]))
                 for e in note_events]
        pedal = [PedalEvent(float(e["onset_time"]), float(e["offset_time"])) for e in pedal_events or []]
        return Transcription(notes=clean_notes(notes), pedal=pedal, transcriber=self.name, version=self.version,
                             elapsed_s=time.perf_counter() - t0, supports_pedal=True, supports_velocity=True)
