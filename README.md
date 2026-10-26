# Recording Assessment System (RAS)

Score-aware assessment of keyboard performances: a student recording is compared with the reference
score and turned into note-level results, a rubric score, confidence and encouraging feedback.

> **Status: Phases 1 and 2 built (core engine, then local API, worker and review UI).** Runs fully on a
> laptop, no cloud, no GPU. Phase 3 is AWS/GPU readiness, on hold. See [docs/PHASES.md](docs/PHASES.md).

## How it works

```
reference (MXL / MIDI) ──► partitura/pretty_midi ──► score note table ─────────────┐
                                                                                    ▼
recording (m4a, wav…) ─► FFmpeg decode ─► quality gate ─► crop at start ─► Transkun V2 ─► performed notes
                                                                                    │
                           transposition check ─► group-level DP alignment ─► timing map
                                                                                    ▼
                    global note assignment ─► correct / wrong pitch / wrong octave / missing / extra
                                                                                    ▼
                    timing, tempo, pauses, chords ─► rubric engine ─► score, confidence, feedback ─► JSON
```

* One learned component: **Transkun V2** (pretrained piano transcription, MIT). Everything else is
  deterministic and explainable note by note. No training in Phase 1.
* The reference is MusicXML/MXL (repeats unfolded, ties merged, grace notes optional) or MIDI.
* The rubric is data (`data/rubrics/*.json`). A criterion that cannot be measured is reported as
  `not_assessed`, never guessed.
* Every result records the model, reference hash, rubric and code versions.

## Setup

Python 3.12 is required (PyTorch/Transkun do not support 3.13+ yet). FFmpeg must be on the PATH.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,transkun,service]"
```

On an Intel Mac, `librosa` needs an older `llvmlite`; if the install fails building it, run
`pip install --only-binary=llvmlite "llvmlite<0.45" "numba<0.62"` first.

Optional, for the Basic Pitch baseline. Its normal install pins old numpy/TensorFlow that do not build on
Python 3.12 (and TensorFlow is not needed): install it without dependencies and use its bundled ONNX model.

```bash
.venv/bin/pip install --no-deps basic-pitch
.venv/bin/pip install --only-binary=:all: "numpy<2" onnxruntime resampy
```

## Use

```bash
# 1. make synthetic songs, mistaken performances, AAC audio and ground truth
.venv/bin/ras synth --out data/synth --n 30

# 2. assess one recording
.venv/bin/ras assess data/synth/refs/twinkle.musicxml data/synth/audio/rec_000.m4a \
    --start-s 2.0 --out data/runs/result.json --midi-out data/runs/performance.mid

# 3. compare transcribers (note F1, error-detection precision/recall, CPU time, peak RAM)
.venv/bin/ras benchmark --data data/synth --models ground-truth,transkun
```

`ground-truth` feeds the known notes into the pipeline: it measures the alignment and
error-detection stages alone, with perfect transcription.

## Run the service (Phase 2)

```bash
.venv/bin/ras serve --data-dir data/service      # API + web UI + a worker thread: http://127.0.0.1:8000/ui/
.venv/bin/ras ingest client/manifest.json         # bulk-load songs, rubric and recordings, queued for the worker
.venv/bin/ras evaluate                            # results vs teacher reviews + a recommended confidence threshold
```

* Web UI: add songs (MusicXML/MIDI) and a rubric, upload a recording, see the score, feedback and a
  per-bar view coloured by error; teachers fix notes and give scores in the **Review queue**.
* API under `/v1` (OpenAPI at `/docs`): files, songs and reference versions, rubric versions,
  assessments (202 + `Idempotency-Key`), notes, reviews, review queue, signed webhooks, RFC 9457 errors.
* Jobs run on a DB-backed queue with retries (3 attempts, then `failed`), stuck-job recovery and
  idempotent submission. Set `RAS_API_KEY` to require an `X-API-Key` header.
* Local stand-ins sit behind small interfaces: filesystem for S3, SQLite for RDS (set `RAS_DB_URL` for
  PostgreSQL), DB queue for SQS.
* Results that claim most of the piece is wrong, or have low confidence, go to the review queue instead
  of being final. Teacher corrections are stored and are the ground truth `ras evaluate` uses.

## Tests

```bash
.venv/bin/pytest            # fast suite, no model needed
.venv/bin/pytest -m slow    # also runs the real Transkun model
```

## Layout

```
ras/core/audio       decode, quality gate, level, crop
ras/core/score       MusicXML/MIDI reference loader
ras/core/transcribe  Transcriber interface + Transkun / Basic Pitch / ground-truth adapters
ras/core/align       transposition detection, group-level DP matcher (and Parangonar)
ras/core/errors      note assignment, error classes, timing, tempo, pauses, chords
ras/core/rubric      rubric model and scoring engine
ras/core/confidence  confidence and "needs review"
ras/core/feedback    template feedback
ras/pipeline.py      end-to-end assess()
ras/service          API (FastAPI), worker, queue, DB models, storage, ingest, evaluation, web UI
ras/synth            songs, MusicXML writer, performance simulator, audio renderer, dataset
ras/eval             metrics, grouped splits, benchmark harness
ras/tools/cli.py     `ras` command
data/rubrics         rubric JSON (dev placeholder only)
```

## Phase 1 results (synthetic, 30 recordings)

| | note onset F1 | error detection precision / recall | CPU s per audio min |
|---|---|---|---|
| ground-truth notes | 1.000 | 0.955 / 0.921 | n/a |
| Transkun V2 | 0.964 | 0.704 / 0.886 | 57.5 |
| Basic Pitch | 0.851 | 0.239 / 0.851 | 2.4 |

Full report and interpretation: [docs/benchmarks/phase1_synthetic.md](docs/benchmarks/phase1_synthetic.md).

## Honest limits of Phase 1

* **All accuracy numbers are on synthetic audio** (a simple additive piano synth). They test the
  pipeline; they say nothing about TalentGum's keyboards, phones or rooms. The real measurement
  starts when the client data and teacher-corrected annotations arrive (Phase 2 tooling).
* The bundled rubric is a **development placeholder**; weights and thresholds are not the client's.
* **Confidence is a heuristic**, to be calibrated on teacher-reviewed recordings.
* **Synthetic audio is a brittle test of Transkun.** Small changes to the synth moved Transkun's
  error-detection precision between 0.70 and 0.34 (see
  [docs/benchmarks/phase1_closeout.md](docs/benchmarks/phase1_closeout.md)). Real recordings are needed.
* The audio onset check for "missing" notes exists but is **off by default**: on synthetic audio it did not
  separate dropped notes from truly absent ones. Evaluate it on real recordings before enabling.
* **Restarts and skipped bars** are not modelled; they surface as missing/extra notes.
  A pause is detected, restarts are not.
* hFT-Transformer is not benchmarked (research code, needs a wrapper).
* Transkun runs at roughly real time or slower on this Intel laptop CPU; see the benchmark report
  for measured numbers before choosing a latency target.
