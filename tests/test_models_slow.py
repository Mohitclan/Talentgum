"""The downloaded open-source models on real audio (slow on CPU). Run with: pytest -m slow

Skipped unless `ras models download` has put the weights in RAS_MODELS_DIR (default ./models)."""

import pytest

from ras.eval.metrics import transcription_metrics
from ras.models import is_available
from ras.synth.render import degrade, render_notes
from ras.types import PerfNote

pytestmark = pytest.mark.slow

TRUTH = [PerfNote(p, 0.5 + 0.6 * i, 0.5 + 0.6 * i + 0.5, 50 + 8 * i)
         for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])]


def _audio():
    return degrade(render_notes(TRUTH), rt60=0.3, snr_db=30)


@pytest.mark.parametrize("name", ["hft", "kong"])
def test_second_transcribers(name):
    if not is_available(name):
        pytest.skip(f"run `ras models download {name}`")
    from ras.core.transcribe import get_transcriber

    out = get_transcriber(name).transcribe(_audio(), 44100)
    m = transcription_metrics(TRUTH, out.notes)
    # Kong misses the upper notes of this simple synthetic timbre once reverb and noise are added (it finds
    # them on the clean render); it is therefore only the pedal source, not the default second opinion
    assert m["onset_precision"] > 0.85 and m["onset_f1"] > (0.8 if name == "hft" else 0.7), m
    assert out.supports_pedal == (name == "kong")


def test_dynamics_model_hears_louder_playing():
    if not is_available("dynamics"):
        pytest.skip("run `ras models download dynamics`")
    import numpy as np

    from ras.core.dynamics_model import DynamicsEstimator

    soft = [PerfNote(n.pitch, n.onset, n.offset, 30) for n in TRUTH]
    loud = [PerfNote(n.pitch, n.onset + 6.0, n.offset + 6.0, 120) for n in TRUTH]
    audio = render_notes(soft + loud)
    lv = DynamicsEstimator().levels_at(audio, 44100, [n.onset for n in soft + loud])
    assert np.mean(lv[8:]) > np.mean(lv[:8])
