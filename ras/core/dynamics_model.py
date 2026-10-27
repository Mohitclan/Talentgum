"""Audio dynamics estimation (He et al., ICASSP 2026, MIT): pp/p/mf/f/ff over time.

The model predicts a dynamics class every 20 ms from Bark-scale loudness. RAS reads it at each
performed note, so dynamics can be judged even on keyboards whose transcribed velocities are flat
(the model hears loudness, not key velocity). Upstream validation F1 is 0.53 on 5 levels, so its
output is used for direction of change (louder/softer than the previous marking), not for
absolute levels. Weights come from `ras models download dynamics`.
"""

from __future__ import annotations

import os

import numpy as np

from ras.models.vendor.pde_model import DYNAMIC_LABELS_5

# 1 = pp ... 5 = ff on the model's 5-level scale; 0 is "no marking" (blank)
LEVEL_OF_MARKING = {"ppp": 1, "pp": 1, "p": 2, "mp": 2.5, "mf": 3, "f": 4, "ff": 5, "fff": 5}


class DynamicsUnavailable(RuntimeError):
    pass


class DynamicsEstimator:
    def __init__(self, device: str | None = None, weights: str | None = None):
        from ras.models import is_available, model_path

        if weights is None and not is_available("dynamics"):
            raise DynamicsUnavailable("dynamics weights missing: run `ras models download dynamics`")
        self._weights = weights or str(model_path("dynamics"))
        self._device = device or os.environ.get("RAS_DEVICE", "auto")
        self._model = None
        self.version = "pde-multitask-sone-fold0-e96"

    def _load(self):
        import torch

        from ras.models.vendor.pde_model import MultiTaskCNN

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self._weights, map_location="cpu", weights_only=True)
        self._cfg = ckpt["cfg"]
        model = MultiTaskCNN(self._cfg)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        self._model = model.to(self._device).eval()

    def frame_levels(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """Expected dynamics level per frame on a 1 (pp) .. 5 (ff) scale, and frames per second.

        The level is the probability-weighted mean over the five markings, ignoring 'blank', so it
        changes smoothly and can be compared between notes.
        """
        import torch

        if self._model is None:
            self._load()
        feat = self._cfg["feature"]
        fs, fps, seg_s = int(feat["sample_rate"]), int(feat["frames_per_second"]), float(feat["segment_seconds"])
        if sr != fs:
            import soxr

            audio = soxr.resample(audio, sr, fs)
        audio = np.asarray(audio, np.float32)
        seg = int(seg_s * fs)
        n_frames = int(round(len(audio) / fs * fps)) + 1
        probs = []
        with torch.no_grad():
            for s in range(0, max(len(audio), 1), seg):
                chunk = audio[s:s + seg]
                chunk = np.pad(chunk, (0, seg - len(chunk)))
                logits = self._model(torch.from_numpy(chunk)[None].to(self._device))["dynamic_output"][0]
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        p = np.concatenate(probs)[:n_frames]
        if len(p) < n_frames:
            p = np.pad(p, ((0, n_frames - len(p)), (0, 0)), mode="edge")
        marked = p[:, 1:]
        level = (marked * np.arange(1, marked.shape[1] + 1)).sum(1) / np.clip(marked.sum(1), 1e-6, None)
        return level, float(fps)

    def levels_at(self, audio: np.ndarray, sr: int, times_s: list[float], offset_s: float = 0.0) -> list[float]:
        """Model level just after each time (seconds on the original recording's clock)."""
        level, fps = self.frame_levels(audio, sr)
        out = []
        for t in times_s:
            i = int(round((t - offset_s) * fps))
            out.append(float(np.mean(level[max(i, 0):max(i, 0) + 5])) if len(level) else float("nan"))
        return out


LABELS = DYNAMIC_LABELS_5
