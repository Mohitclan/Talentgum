"""Render notes to audio with a small additive piano synth, then degrade like a phone recording.

This is deliberately simple. Real keyboards sound different, so scores from synthetic audio
say nothing about accuracy on TalentGum recordings; they are for pipeline testing only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from ras.types import PerfNote

SR = 44100


def _render_note(pitch: int, dur: float, vel: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    f0 = 440.0 * 2 ** ((pitch - 69) / 12) * (1 + rng.normal(0, 0.0004))
    tau0 = float(np.interp(pitch, [21, 60, 108], [3.0, 1.6, 0.5]))
    length = dur + 0.35
    t = np.arange(int(length * sr)) / sr
    out = np.zeros_like(t)
    B = 0.0002
    for k in range(1, 13):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1.0 / k**1.1) * (1 + 0.3 * np.sin(k + pitch))
        out += amp * np.exp(-t / (tau0 / (1 + 0.55 * (k - 1)))) * np.sin(
            2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    env = np.minimum(1.0, t / 0.004)
    gate = np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.06))  # key released: damper falls
    out *= env * gate
    # short hammer noise burst
    nb = int(0.015 * sr)
    out[:nb] += rng.normal(0, 0.05, nb) * np.linspace(1, 0, nb)
    return out * (vel / 127.0) ** 1.3


def render_notes(notes: list[PerfNote], sr: int = SR, seed: int = 0, tail_s: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = max((n.offset for n in notes), default=1.0) + 0.4 + tail_s
    y = np.zeros(int(total * sr))
    for n in notes:
        seg = _render_note(n.pitch, max(n.offset - n.onset, 0.05), n.velocity, sr, rng)
        i = int(n.onset * sr)
        y[i:i + len(seg)] += seg[: len(y) - i]
    peak = np.max(np.abs(y)) or 1.0
    return (0.5 * y / peak).astype(np.float32)


def degrade(
    y: np.ndarray,
    sr: int = SR,
    rt60: float = 0.0,
    snr_db: float | None = None,
    lowpass_hz: float | None = None,
    gain_db: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Room reverb, background noise, band-limiting and gain, approximating a phone recording."""
    rng = np.random.default_rng(seed)
    x = y.astype(np.float64)
    if rt60 > 0:
        n = int(rt60 * sr)
        t = np.arange(n) / sr
        ir = rng.normal(0, 1, n) * np.exp(-6.9 * t / rt60)
        ir[0] = 1.0
        wet = fftconvolve(x, ir / np.sqrt(np.sum(ir**2)))[: len(x)]
        x = 0.6 * x + 0.4 * wet * (np.std(x) / (np.std(wet) + 1e-9))
    if lowpass_hz:
        x = sosfilt(butter(4, lowpass_hz, fs=sr, output="sos"), x)
    if snr_db is not None:
        p = np.mean(x**2) + 1e-12
        noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))
        x = x + noise
    x *= 10 ** (gain_db / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def write_audio(y: np.ndarray, path: str | Path, sr: int = SR) -> Path:
    """Write wav, or AAC m4a (the format the client uses) when the suffix is .m4a."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".wav":
        sf.write(str(path), y, sr)
        return path
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), y, sr)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:a", "aac", "-b:a", "64k", str(path)],
        check=True,
    )
    tmp.unlink()
    return path
