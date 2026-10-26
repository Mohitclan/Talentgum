import numpy as np
import pytest

from ras.core.audio.decode import (
    DecodeError,
    assess_quality,
    crop_from_start,
    decode_audio,
    normalise_level,
)
from ras.synth.render import degrade, render_notes, write_audio
from ras.types import PerfNote

SR = 44100


def test_decode_m4a_roundtrip(tmp_path):
    y = render_notes([PerfNote(60, 0.2, 1.0, 80), PerfNote(64, 1.0, 1.8, 80)])
    path = write_audio(y, tmp_path / "a.m4a")
    x = decode_audio(path, SR)
    assert abs(len(x) / SR - len(y) / SR) < 0.1
    assert x.dtype == np.float32 and np.max(np.abs(x)) > 0.1


def test_decode_failure_is_explicit(tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(DecodeError):
        decode_audio(bad, SR)


def test_quality_gate_rejects_bad_input():
    assert assess_quality(np.zeros(SR * 5, dtype=np.float32), SR).reject_reason in (
        "too_quiet", "mostly_silence")
    assert assess_quality(np.zeros(SR // 4, dtype=np.float32), SR).reject_reason == "too_short"
    clipped = np.clip(np.sin(np.linspace(0, 2000, SR * 3)) * 5, -1, 1).astype(np.float32)
    assert assess_quality(clipped, SR).reject_reason == "clipping"


def test_quality_gate_accepts_piano():
    y = render_notes([PerfNote(60 + i, i * 0.5, i * 0.5 + 0.4, 80) for i in range(8)])
    rep = assess_quality(degrade(y, rt60=0.3, snr_db=30), SR)
    assert rep.ok and rep.quality_score > 0.7


def test_normalise_only_boosts_quiet_audio():
    quiet = (np.sin(np.linspace(0, 300, SR)) * 0.05).astype(np.float32)
    assert abs(np.max(np.abs(normalise_level(quiet))) - 0.9) < 0.01
    very_quiet = (np.sin(np.linspace(0, 300, SR)) * 0.001).astype(np.float32)
    assert abs(np.max(np.abs(normalise_level(very_quiet))) - 0.001 * 10 ** 1.5) < 1e-3  # +30 dB cap
    loud = (np.sin(np.linspace(0, 300, SR)) * 0.8).astype(np.float32)
    assert normalise_level(loud) is loud


def test_crop_keeps_margin():
    x = np.arange(SR * 10, dtype=np.float32)
    cropped, offset = crop_from_start(x, SR, 4.0, margin_s=1.0)
    assert offset == 3.0 and len(cropped) == SR * 7
    same, off0 = crop_from_start(x, SR, 0.5)
    assert off0 == 0.0 and same is x
