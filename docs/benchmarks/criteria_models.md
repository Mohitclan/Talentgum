# Client criteria and open-source models: measurements (synthetic audio)

Measured on 2026-10-27, Intel Mac CPU. All audio is synthetic (simple additive piano synth); real
TalentGum recordings are still needed.

## Second opinion (hFT-Transformer) on error detection

The first 16 recordings of `data/synth` were assessed with Transkun V2, once without and once with hFT as
the second opinion (policy `uncertain`). Error claims are scored against the injected mistakes
(`ras.eval.metrics.detection_counts`).

| | precision | recall | F1 |
|---|---|---|---|
| Transkun alone | 0.725 | 0.899 | 0.803 |
| Transkun + hFT second opinion | **0.906** | 0.891 | **0.898** |

By error type (precision / recall):

| | wrong pitch | wrong octave | missing | extra |
|---|---|---|---|---|
| alone | 0.922 / 0.959 | 1.000 / 0.750 | 0.623 / 0.935 | 0.588 / 0.769 |
| with hFT | 0.922 / 0.959 | 1.000 / 0.750 | **0.894** / 0.913 | **0.870** / 0.769 |

The gain comes where Transkun was weak: notes it dropped (false "missing") and notes it imagined
(false "extra"). One recording out of 16 went to review in the first run and two in the second, because
withdrawn claims count as uncertain. Cost: hFT takes about as long as Transkun on CPU (35 s for a 21 s
recording), so a recording takes roughly twice as long.

Example, recording `rec_001` (21 s, 4 wrong notes and 1 wrong octave injected):
- Transkun alone reported those 5 errors plus 8 false "missing" notes.
- hFT heard all 8 of those notes, so they were withdrawn. The final result has exactly the 5 injected
  errors.

## Models on a short synthetic clip (`pytest -m slow tests/test_models_slow.py`)

| Model | Result |
|---|---|
| hFT-Transformer | onset F1 > 0.8, precision 0.89 (one extra note) |
| Kong | Misses the upper notes of the simple synth once reverb and noise are added (onset F1 0.77, precision 1.0). Finds them on the clean render. It is used as the pedal source, not as the default second opinion. It reported 14 pedal events on a synthetic recording that has no pedal: calibrate before trusting its pedal on real keyboards. |
| Dynamics model | Rates a passage played at velocity 120 louder than the same passage at velocity 30 |

## Not measured yet

- Articulation, dynamics and pedal criteria on real recordings. They are covered by unit tests on a
  marked score, but have no real-audio accuracy number.
- Teacher-profile comparisons on real teacher recordings.
- Whether `second_opinion_policy: credit` (count hFT-confirmed notes as correct) is safe. That needs
  teacher-reviewed recordings.
