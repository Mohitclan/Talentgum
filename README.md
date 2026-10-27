# Recording Assessment System (RAS)

Score-aware assessment of keyboard performances: a student recording is compared with the reference
score and turned into note-level results, a rubric score, confidence and encouraging feedback.

> **Status: Phases 1–3 built.**
> * Phase 1: the core engine.
> * Phase 2: the local API, worker and review UI.
> * Phase 3: the AWS deployment (Docker image, S3/SQS/PostgreSQL backends, Terraform), tested locally.
>
> Nothing is deployed to AWS yet. Without Docker it all still runs on a laptop with no cloud and no GPU.
> See [docs/PHASES.md](docs/PHASES.md) and [docs/DEPLOY.md](docs/DEPLOY.md).

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
scripts/setup.sh
```

The script creates `.venv`, installs everything (`.[dev,transkun,service]` plus the optional Basic Pitch
baseline through its bundled ONNX model) and runs the tests. It handles three macOS quirks: a Homebrew
Python whose `venv` cannot install pip, the missing Intel-Mac wheel for the newest `llvmlite`, and
Basic Pitch's old numpy/TensorFlow pins that do not build on Python 3.12.

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
  idempotent submission.
* **Tenants:** each client organisation is a tenant with its own API keys (`X-API-Key`, stored only as
  hashes). Songs, rubrics, recordings, results, reviews, audit entries and storage prefixes are isolated
  per tenant. With no keys created the API runs open as the `local` tenant, which is convenient for
  development: `ras tenant-create talentgum` prints the first key and switches the server to keys.
  Rotate with `ras key-create` / `ras key-revoke`.
* **Rubric test:** `POST /v1/rubrics/test` (and a button in the UI) re-scores past results with a candidate
  rubric and shows the change before it is saved. Results keep their scoring inputs for this.
* **Privacy:** `DELETE /v1/students/{id}` erases a student's recordings, results and reviews;
  `ras purge-audio --older-than-days N` enforces audio retention (results are kept); every recording read,
  review and erasure is in `GET /v1/audit`.
* Storage, database and queue sit behind small interfaces: files, SQLite and table polling locally;
  S3, PostgreSQL and SQS on AWS (environment variables, see [docs/DEPLOY.md](docs/DEPLOY.md)).
* Results that claim most of the piece is wrong, or have low confidence, go to the review queue instead
  of being final. Teacher corrections are stored and are the ground truth `ras evaluate` uses.

## Run the AWS shape locally (Phase 3)

```bash
docker compose up -d --build        # PostgreSQL, S3 (SeaweedFS), SQS (ElasticMQ), API on :8088, a worker
```

AWS deployment, GPU benchmarking and the licence register: [docs/DEPLOY.md](docs/DEPLOY.md).

## Tests

```bash
.venv/bin/pytest                    # fast suite, no model needed
.venv/bin/pytest -m slow            # also runs the real Transkun model
docker compose run --rm tests       # cloud backends: PostgreSQL, S3, SQS (stack must be up)
```

## Layout

```
ras/core/audio       decode, quality gate, level, crop
ras/core/score       MusicXML/MIDI reference loader
ras/core/transcribe  Transcriber interface + Transkun / Basic Pitch / ground-truth adapters
ras/core/align       transposition detection, group-level DP matcher (and Parangonar)
ras/core/errors      note assignment, error classes, timing, tempo, pauses, restarts, chords
ras/core/rubric      rubric model and scoring engine
ras/core/confidence  confidence and "needs review"
ras/core/feedback    template feedback
ras/pipeline.py      end-to-end assess()
ras/service          API (FastAPI), worker, queue, DB models, storage, ingest, evaluation, web UI
ras/synth            songs, MusicXML writer, performance simulator, audio renderer, dataset
ras/eval             metrics, grouped splits, benchmark harness
ras/tools/cli.py     `ras` command (ras/tools/sbom.py: licence register)
data/rubrics         rubric JSON (dev placeholder only)
Dockerfile, docker-compose.yml   one image for API/worker/admin; the local AWS-shaped stack
infra/terraform      AWS: ECS, RDS, S3, SQS, ALB, alarms, budget, GPU workers (+ offline plan tests)
scripts/             setup.sh, smoke_stack.py (end-to-end check of a running deployment)
```

## Phase 1 results (synthetic, 30 recordings)

| | note onset F1 | error detection precision / recall | CPU s per audio min |
|---|---|---|---|
| ground-truth notes | 1.000 | 0.955 / 0.921 | n/a |
| Transkun V2 | 0.964 | 0.704 / 0.886 | 57.5 |
| Basic Pitch | 0.851 | 0.239 / 0.851 | 2.4 |

Full report and interpretation: [docs/benchmarks/phase1_synthetic.md](docs/benchmarks/phase1_synthetic.md).

## Honest limits

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
* **Restarts** are detected (an abandoned attempt the student plays again is labelled `replayed`, not
  extra; 17 of 20 simulated restarts found, 98.6% of notes still judged correctly). Very short abandoned
  fragments (about 5 notes) can be missed. Skipped bars show as missing notes.
* hFT-Transformer is not benchmarked (research code, needs a wrapper).
* Transkun runs at roughly real time or slower on this Intel laptop CPU; see the benchmark report
  for measured numbers before choosing a latency target.
