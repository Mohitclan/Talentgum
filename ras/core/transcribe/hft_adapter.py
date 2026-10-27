"""hFT-Transformer (Sony, MIT) piano transcription: the second opinion for missing and wrong notes.

Feature extraction, windowing and note decoding follow upstream model/amt.py (AMT.wav2feature,
AMT.transcript, AMT.mpe2note) with the second-stage ("2nd", time-attention) outputs, which the paper
reports as the final result. Weights come from `ras models download hft`.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ras.types import PerfNote, Transcription

from .base import TranscriberUnavailable, clean_notes

SR, HOP, N_FFT, N_MELS = 16000, 256, 2048, 256
MARGIN, N_FRAME, NOTE_MIN, N_NOTE = 32, 128, 21, 88
LOG_OFFSET = 1e-8
MIN_VALUE = float(np.log(LOG_OFFSET))  # padding value of the log-mel input


def _peaks(x: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    """Local maxima of one pitch's activation above `threshold`, with sub-frame time (frames)."""
    out = []
    n = len(x)
    for i in np.flatnonzero(x >= threshold):
        j = i - 1
        while j >= 0 and x[j] == x[i]:
            j -= 1
        k = i + 1
        while k < n and x[k] == x[i]:
            k += 1
        if (j >= 0 and x[j] > x[i]) or (k < n and x[k] > x[i]):
            continue
        if 0 < i < n - 1 and x[i - 1] != x[i + 1]:
            lo, hi = x[i - 1], x[i + 1]
            shift = 0.5 * (hi - lo) / (x[i] - min(lo, hi)) if x[i] > min(lo, hi) else 0.0
            out.append((int(i), i + shift))
        else:
            out.append((int(i), float(i)))
    return out


def decode_notes(onset: np.ndarray, offset: np.ndarray, mpe: np.ndarray, velocity: np.ndarray,
                 th_onset: float = 0.5, th_offset: float = 0.5, th_mpe: float = 0.5) -> list[PerfNote]:
    """Upstream mpe2note with mode_offset='shorter' and mode_velocity='ignore_zero'."""
    hop_s = HOP / SR
    notes: list[PerfNote] = []
    n_frames = len(mpe)
    for j in range(N_NOTE):
        ons = _peaks(onset[:, j], th_onset)
        offs = _peaks(offset[:, j], th_offset)
        pitch_notes: list[PerfNote] = []
        for k, (loc_on, t_on) in enumerate(ons):
            if k + 1 < len(ons):
                loc_next, t_next = ons[k + 1][0], ons[k + 1][1] * hop_s
            else:
                loc_next, t_next = n_frames, (n_frames - 1) * hop_s
            loc_off, t_off, has_off = loc_on + 1, None, False
            for lo, to in offs:
                if lo > loc_on:
                    loc_off, t_off, has_off = lo, to * hop_s, True
                    break
            if loc_off > loc_next:
                loc_off, t_off = loc_next, t_next
            below = np.flatnonzero(mpe[loc_on + 1:loc_next, j] < th_mpe)
            has_mpe = len(below) > 0
            loc_mpe = loc_on + 1 + int(below[0]) if has_mpe else loc_on + 1
            t_mpe = loc_mpe * hop_s
            if not has_off and not has_mpe:
                end = t_next
            elif has_off and not has_mpe:
                end = t_off
            elif has_mpe and not has_off:
                end = t_mpe
            else:
                end = t_off if loc_off <= loc_mpe else t_mpe
            vel = int(velocity[loc_on, j])
            if vel <= 0:
                continue
            start = t_on * hop_s
            if pitch_notes and pitch_notes[-1].offset > start:
                pitch_notes[-1].offset = start
            pitch_notes.append(PerfNote(NOTE_MIN + j, float(start), float(max(end, start + hop_s)), vel))
        notes += pitch_notes
    return notes


class HftTranscriber:
    name = "hft"

    def __init__(self, device: str | None = None, weights: str | None = None, batch: int = 4):
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise TranscriberUnavailable("install the extra: pip install -e '.[transkun]'") from exc
        from ras.models import is_available, model_path

        if weights is None and not is_available("hft"):
            raise TranscriberUnavailable("hFT-Transformer weights missing: run `ras models download hft`")
        self._weights = weights or str(model_path("hft"))
        self._device = device or os.environ.get("RAS_DEVICE", "auto")
        self._batch = batch
        self._model = None
        self.sample_rate = SR
        self.version = "hft-maestro-v3-016_003"

    def _load(self):
        import torch

        from ras.models.vendor.hft_model import build

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self._weights, map_location="cpu", weights_only=True)
        model = build(ckpt["params"], self._device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        self._model = model.to(self._device).eval()
        self.device = self._device

    def _features(self, audio: np.ndarray, sr: int):
        import torch
        import torchaudio

        if sr != SR:
            import soxr

            audio = soxr.resample(audio, sr, SR)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                                   pad_mode="constant", n_mels=N_MELS, norm="slaney")
        spec = mel(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)))
        return torch.log(spec + LOG_OFFSET).T  # [frames, mels]

    def activations(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Frame-wise onset/offset/frame/velocity for each of the 88 keys (second stage)."""
        import torch

        if self._model is None:
            self._load()
        feat = self._features(audio, sr).numpy()
        n = feat.shape[0]
        pad_end = int(np.ceil(n / N_FRAME) * N_FRAME) - n
        x = np.concatenate([np.full((MARGIN, N_MELS), MIN_VALUE, np.float32), feat,
                            np.full((pad_end + MARGIN, N_MELS), MIN_VALUE, np.float32)])
        windows = [x[i:i + MARGIN + N_FRAME + MARGIN].T for i in range(0, n, N_FRAME)]
        outs = {k: [] for k in ("onset", "offset", "mpe", "velocity")}
        with torch.no_grad():
            for b in range(0, len(windows), self._batch):
                batch = torch.from_numpy(np.stack(windows[b:b + self._batch])).to(self._device)
                _, _, _, _, _, on, off, mpe, vel = self._model(batch)
                outs["onset"].append(on.cpu().numpy().reshape(-1, N_NOTE))
                outs["offset"].append(off.cpu().numpy().reshape(-1, N_NOTE))
                outs["mpe"].append(mpe.cpu().numpy().reshape(-1, N_NOTE))
                outs["velocity"].append(vel.argmax(-1).cpu().numpy().reshape(-1, N_NOTE))
        return {k: np.concatenate(v)[:n] for k, v in outs.items()}

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        t0 = time.perf_counter()
        a = self.activations(audio, sr)
        notes = decode_notes(a["onset"], a["offset"], a["mpe"], a["velocity"])
        return Transcription(notes=clean_notes(notes), transcriber=self.name, version=self.version,
                             elapsed_s=time.perf_counter() - t0, supports_pedal=False, supports_velocity=True)
