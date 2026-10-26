"""Needs the real model (slow on CPU). Run with: pytest -m slow"""

import pytest

from ras.eval.metrics import transcription_metrics
from ras.synth.render import degrade, render_notes
from ras.types import PerfNote

pytestmark = pytest.mark.slow


def test_transkun_transcribes_synthetic_piano():
    pytest.importorskip("transkun")
    from ras.core.transcribe import get_transcriber

    truth = [PerfNote(p, 0.5 + 0.6 * i, 0.5 + 0.6 * i + 0.5, 80)
             for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])]
    audio = degrade(render_notes(truth), rt60=0.3, snr_db=30)
    out = get_transcriber("transkun").transcribe(audio, 44100)
    assert out.supports_pedal and out.version.startswith("transkun-")
    m = transcription_metrics(truth, out.notes)
    assert m["onset_f1"] > 0.8, m
