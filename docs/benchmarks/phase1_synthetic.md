# RAS benchmark

Dataset: `data/synth` (30 synthetic recordings). Synthetic audio only: these numbers test the pipeline and say nothing about accuracy on TalentGum keyboards or phones.

## Transcription (note-level, mir_eval)

| model | onset F1 (95% CI) | onset+offset F1 | onset+velocity F1 | CPU s per audio min | peak RAM GB | failures |
|---|---|---|---|---|---|---|
| ground-truth | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.0 | 0.21 | 0 |
| transkun | 0.964 [0.946, 0.979] | 0.905 | 0.400 | 57.5 | 1.47 | 0 |
| basic-pitch | 0.851 [0.814, 0.890] | 0.586 | 0.321 | 2.4 | 0.34 | 0 |

## Error detection (pooled over recordings; precision matters most)

| model | precision (CI) | recall (CI) | F1 | timing MAE ms | review rate |
|---|---|---|---|---|---|
| ground-truth | 0.955 [0.930, 0.979] | 0.921 [0.879, 0.961] | 0.938 | 23.9 | 0.00 |
| transkun | 0.704 [0.621, 0.782] | 0.886 [0.821, 0.942] | 0.784 | 25.7 | 0.00 |
| basic-pitch | 0.239 [0.176, 0.326] | 0.851 [0.804, 0.905] | 0.373 | 39.1 | 0.37 |

## Per error type (precision / recall)

| model | wrong_pitch | wrong_octave | missing | extra |
|---|---|---|---|---|
| ground-truth | 0.93 / 0.98 | 1.00 / 1.00 | 0.99 / 0.91 | 0.93 / 0.82 |
| transkun | 0.92 / 0.95 | 1.00 / 0.60 | 0.62 / 0.94 | 0.54 / 0.78 |
| basic-pitch | 0.88 / 0.91 | 0.70 / 0.93 | 0.86 / 0.77 | 0.07 / 0.86 |

## Onset F1 by recording condition (domain-shift check)

| model | clean | noisy_quiet | phone | room |
|---|---|---|---|---|
| ground-truth | 1.000 | 1.000 | 1.000 | 1.000 |
| transkun | 0.924 | 0.981 | 0.987 | 0.984 |
| basic-pitch | 0.789 | 0.918 | 0.864 | 0.874 |

## Not benchmarked

- **hft-transformer**: research code (shell scripts, corpus generation); needs a wrapper before it can be benchmarked. Slot reserved behind the Transcriber interface.

`ground-truth` feeds the known notes to the pipeline: it is the upper bound for alignment and error detection when transcription is perfect.

## Reading these numbers (Phase 1, synthetic audio)

* Perfect transcription (`ground-truth`) gives error-detection precision 0.955 / recall 0.921, so the
  alignment and error-detection stages themselves are sound.
* Transkun reaches onset F1 0.964 but error-detection precision drops to 0.704. The loss comes from
  transcription misses: on `ode_to_joy` and `mary` (long runs of repeated same-pitch notes) it drops
  re-struck notes on this synthetic audio (precision 1.0, recall about 0.7), which shows up as false
  "missing" notes. Whether real keyboards behave the same is unknown until real recordings are measured.
* The review flag stayed off for those recordings, so the confidence heuristic does not yet catch this.
  Both points are Phase 2 work (real data, calibration on teacher-reviewed recordings).
* Basic Pitch is a lower bound only: many ghost notes (extra-note precision 0.07).
* Transkun needs about 57 CPU-seconds per audio minute on this Intel laptop (about 1 s per audio second),
  1.5 GB peak RAM. Measure again on the target CPU before fixing a latency target.
