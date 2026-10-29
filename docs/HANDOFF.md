# Handoff: where the project stands

Written 2026-10-29 so a new person or a new Claude account can carry on without the old chat.
Read this first, then [PHASES.md](PHASES.md), [CRITERIA.md](CRITERIA.md) and [CLASS_RECORDINGS.md](CLASS_RECORDINGS.md).

## What this is

The Recording Assessment System (RAS) for TalentGum. A student keyboard recording is compared with the
reference score (MXL/MIDI) and turned into note-level results, a rubric score, confidence and positive
feedback. Repo: https://github.com/Mohitclan/Talentgum (branch `main`).

## What is built

| Part | Status |
|---|---|
| Phase 1: engine (decode, quality gate, Transkun V2, alignment, error detection, rubric, feedback) | Built, tested |
| Phase 2: REST API, worker with retries, review UI, tenants and API keys, privacy (erasure, retention, audit) | Built, tested locally |
| Phase 3: Docker image, S3/PostgreSQL/SQS backends, Terraform, GPU worker setup | Built, tested locally. **Not deployed to AWS** |
| All 22 client criteria scored by `data/rubrics/talentgum_criteria_v1.json` | Built. Weights are first guesses |
| Open-source models for the weak criteria (hFT, Kong, dynamics) | Built, downloadable with checksums (`ras models download`) |
| Class-recording splitter (`ras segment`, `ras segment-eval`) | Built. **Tested on synthetic classes only** |

## Open-source models (4, all MIT)

| Model | Used for | Criteria |
|---|---|---|
| Transkun V2 | main transcription | all 22 |
| hFT-Transformer (Sony) | second opinion, can only withdraw error claims | correct, incorrect, missed, extra notes |
| piano-dynamic-estimation | loudness from audio (weak: F1 0.53, direction of change only) | dynamics, dynamic consistency |
| Kong et al. (ByteDance) | pedal | sustain/pedal |

The other 15 criteria use Transkun plus our own deterministic code (alignment, error detection, rubric
engine). Third-party checkpoints load through an allow-list loader (`ras/models/safe_load.py`).

## What the client told us (reply of 28 Oct 2026)

- Five launch songs: Twinkle Twinkle Little Star, Mary Had a Little Lamb, Happy Birthday, Believer, Ode to
  Joy. About 50 students per song.
- Training data is **online class recordings**: M4A/AAC, 45 min average, 60 max, one folder per student,
  talking and noise, several attempts or two songs per recording. No start times, no quality labels.
- Production input is different: about 20 recordings a day, 2-3 min each, 5 min maximum.
- Reference is MXL and MIDI only. Teachers sometimes play wrong notes, so teacher recordings are never
  note truth. No existing rubric. All 22 criteria required. Tone must be positive.
- Students play right hand, left hand or both. No paired audio and MIDI exists.
- AWS, Docker, REST. Corrected MIDI may be kept for training; teachers verify jointly with us.
- Only the Ode to Joy MXL and MIDI have been received (notes only, right hand, no marks).

## Honest limits

- **Every accuracy number is from synthetic audio.** Nothing is measured on real TalentGum recordings.
- The rubric weights and confidence are heuristics until calibrated on teacher marks.
- The assessment judges at most 5 minutes (`max_duration_s=300`), one performance, 15 minute job timeout,
  50 MB upload cap. Class recordings must go through `ras segment` first.
- The splitter has not seen real class audio. Its thresholds are starting values.
- Dynamics, articulation and pedal are judged only when the score carries those marks, or a teacher
  benchmark exists. Dynamics also needs a touch-sensitive keyboard or the weak audio model.

## Still needed from the client

1. MXL/MIDI of Twinkle, Mary, Happy Birthday and Believer.
2. Three or four real class recordings to prototype on, and about five with a teacher's start and end times
   for each song.
3. Decisions: the hands rule (what to score when only one hand is played), what to do when the teacher
   plays in class, keyboard types, parent permission to use recordings for training, retention period,
   AWS region and account, volume of human verification, acceptance criteria.
4. The GPU's type and how long we can use it.

The full question list is the Claude Docs file https://claude.ai/artifact/DopkKPuerj5395D7rvgNBr
(sections "What to ask first", "Answers received", "Still open after the reply"). It lives in the old
account, so the open items are repeated above.

## Next steps, in order

1. Get the missing songs and real classes, then run
   `ras segment classes/ --songs songs/ --out data/classes` and score it with
   `ras segment-eval annotation.json labels.json`. Tune the thresholds in `ras/segment/find.py`
   (`SegmentConfig`).
2. Time one real class on the GPU.
3. Build the teacher review screen for clips (the current screen works on notes, not clips).
4. Add a "score only the part played" mode, so stopped or one-handed attempts are scored fairly.
5. First accuracy test of `assess` on real clips; teachers verify a sample; calibrate with `ras calibrate`.
6. Only if errors repeat on their keyboards: fine-tune Transkun (`ras finetune prepare/export`, CUDA only;
   see [DEPLOY.md](DEPLOY.md) section 4).
7. Deploy to AWS (Terraform in `infra/terraform`, steps in [DEPLOY.md](DEPLOY.md)).

## How to run

```bash
scripts/setup.sh                                   # creates .venv, installs, runs tests (Python 3.12, FFmpeg needed)
.venv/bin/pytest                                   # fast suite, no model needed (119 tests)
.venv/bin/pytest -m slow                           # also runs Transkun and the downloaded models
.venv/bin/ras models download                      # hFT, Kong, dynamics (checksummed)
.venv/bin/ras criteria                             # the 22 criteria and what scores each
.venv/bin/ras assess ref.musicxml take.m4a --rubric data/rubrics/talentgum_criteria_v1.json
.venv/bin/ras serve --data-dir data/service        # API + UI at http://127.0.0.1:8000/ui/
docker compose up -d --build                       # the AWS-shaped stack locally
```

## Where things are

| Path | What |
|---|---|
| `ras/pipeline.py` | end-to-end `assess()` |
| `ras/core/` | audio, score, transcribe adapters, alignment, errors, expression, rubric, feedback |
| `ras/core/criteria.py` | the 22 criteria and their models |
| `ras/models/` | model registry, checksummed download, allow-list loader, vendored model code |
| `ras/segment/` | class splitter: alignment, region finding, song identification, clips, evaluation |
| `ras/service/` | API, worker, queue, database, storage, review UI |
| `ras/eval/` | metrics, benchmark, calibration, fine-tuning data |
| `data/rubrics/` | rubric JSON files |
| `infra/terraform/` | AWS setup as code |
| `docs/` | PHASES, CRITERIA, CLASS_RECORDINGS, DEPLOY, benchmarks, compliance/DECISIONS |

## Working agreements

- Commit and push only when the user asks.
- Commits are authored as Mohit kumar `<192804849+Mohitclan@users.noreply.github.com>`, with the Claude
  co-author trailer.
- Model and data downloads need the user's explicit approval.
- The user explains this project to non-technical people: use simple words, and tick/cross tables when asked.
- Prefer simple single-purpose shell commands.
