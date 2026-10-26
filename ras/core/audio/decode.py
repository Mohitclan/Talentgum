"""Decode, quality gate and level handling. No ML here."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


class DecodeError(RuntimeError):
    pass


def decode_audio(path: str | Path, sr: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at `sr`."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(proc.stderr.decode(errors="replace").strip() or "empty output")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


@dataclass
class QualityReport:
    duration_s: float
    peak: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    snr_est_db: float
    ok: bool = True
    reject_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 1.0  # 0..1, feeds the confidence estimate

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityLimits:
    min_duration_s: float = 1.0
    max_duration_s: float = 300.0
    max_clipping_ratio: float = 0.05
    max_silence_ratio: float = 0.97
    min_rms_dbfs: float = -65.0
    warn_clipping_ratio: float = 0.002
    warn_snr_db: float = 12.0


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([np.sqrt(np.mean(x**2) + 1e-12)])
    return np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)


def assess_quality(x: np.ndarray, sr: int, limits: QualityLimits | None = None) -> QualityReport:
    limits = limits or QualityLimits()
    dur = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x**2) + 1e-12)) if len(x) else 0.0
    rms_db = 20 * np.log10(rms + 1e-12)
    clip = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    frames = _frame_rms(x, int(0.05 * sr))
    frames_db = 20 * np.log10(frames)
    silence = float(np.mean(frames_db < -50.0))
    # crude SNR proxy: loud frames vs the quietest frames (noise floor)
    snr = float(np.percentile(frames_db, 90) - np.percentile(frames_db, 10))

    rep = QualityReport(dur, peak, rms_db, clip, silence, snr)
    if dur < limits.min_duration_s:
        rep.ok, rep.reject_reason = False, "too_short"
    elif dur > limits.max_duration_s:
        rep.ok, rep.reject_reason = False, "too_long"
    elif rms_db < limits.min_rms_dbfs:
        rep.ok, rep.reject_reason = False, "too_quiet"
    elif clip > limits.max_clipping_ratio:
        rep.ok, rep.reject_reason = False, "clipping"
    elif silence > limits.max_silence_ratio:
        rep.ok, rep.reject_reason = False, "mostly_silence"

    score = 1.0
    if clip > limits.warn_clipping_ratio:
        rep.warnings.append("some_clipping")
        score -= min(0.4, clip * 20)
    if snr < limits.warn_snr_db:
        rep.warnings.append("low_dynamic_range_or_noisy")
        score -= 0.2
    if silence > 0.8:
        rep.warnings.append("mostly_silence")
        score -= 0.2
    rep.quality_score = float(np.clip(score, 0.0, 1.0))
    return rep


def normalise_level(x: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Boost quiet recordings so the transcriber sees a usable level.

    Only boosts (never attenuates); gain is capped at +30 dB.
    """
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 1e-6 or peak >= target_peak * 0.5:
        return x
    gain = min(target_peak / peak, 10 ** (30 / 20))
    return (x * gain).astype(np.float32)


def crop_from_start(x: np.ndarray, sr: int, start_s: float | None, margin_s: float = 1.0):
    """Drop audio before (start_s - margin). Returns (cropped, offset_s)."""
    if start_s is None or start_s <= margin_s:
        return x, 0.0
    off = start_s - margin_s
    return x[int(off * sr):], float(off)
