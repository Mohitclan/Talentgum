# Delivery phases

## Phase 1: Core engine (CLI, local): built, gaps closed out (see benchmarks/phase1_closeout.md)
Second close-out round: restart detection (`replayed` notes, restart events, `restart_count` metric),
MusicXML and MIDI tempo changes applied, sustain-pedal use measured (`pedal_use_pct`), a too-late start
timestamp falls back to the full recording, and a setup script tested on a clean copy.
Setup, synthetic data tool, the full pipeline as a CLI, and a benchmark harness.
Done when injected errors are detected at the expected precision/recall on synthetic data and the
model comparison is measured locally. See `README.md` and the latest `data/runs/*/report.md`.

## Phase 2: Service and review (API, worker, UI, local): built
* `/v1` API: files, songs with immutable reference versions, rubric versions, assessments (202,
  `Idempotency-Key`), notes, reviews, review queue, signed webhooks, RFC 9457 problem details, optional API key.
* DB-backed job queue: claim, 3 attempts then `failed`, stuck-job recovery, deterministic outcomes
  (rejected recording) not retried. Local stand-ins behind interfaces: filesystem for S3, SQLite for RDS
  (PostgreSQL via `RAS_DB_URL`), DB queue for SQS.
* Web UI (no build step): assess a recording, score view coloured by error, review queue with note
  correction and teacher scores, history, songs and rubrics.
* Review storage as ground truth; `ras ingest` (bulk load client data) and `ras evaluate` (error-detection
  precision/recall vs teachers, score agreement, recommended confidence threshold).
* Tenants with per-tenant hashed API keys, isolation of every table and storage prefix, key rotation
  and revocation.
* Rubric test (re-score past results with a candidate rubric), student erasure, audio retention purge,
  audit log of recording access, reviews and erasures.
* Not done: confidence calibration on real reviews (the tool exists, the data does not), user accounts and
  roles for teachers (all callers of a tenant are equal), S3/SQS adapters and PostgreSQL in practice (Phase 3),
  model training endpoints (no model is trained in this design).

## Phase 3: AWS and GPU readiness: built and tested locally; not deployed (no AWS account or GPU yet)
Runbook: [DEPLOY.md](DEPLOY.md).
* Cloud backends behind the existing interfaces:
  * S3 object store with a local read cache, audio objects tagged for lifecycle rules.
  * SQS as a wake-up signal; the database stays the queue of record.
  * PostgreSQL with schema creation that is safe when the API and workers start together.
  * `RAS_DEVICE` (cpu, cuda or auto) for Transkun.
* One Docker image (CPU, or CUDA through a build argument) for the API, workers and admin tasks. Runs as
  non-root with a health check. Workers stop cleanly on SIGTERM.
* `docker compose` runs the AWS shape locally: PostgreSQL, SeaweedFS (S3) and ElasticMQ (SQS). Integration
  tests run against it, and a smoke script assesses a recording end to end with Transkun.
* Terraform (`infra/terraform`):
  * Data and network: VPC inputs, ALB with HTTPS, S3, SQS, RDS PostgreSQL, ECR.
  * Compute: ECS Fargate for the API and CPU workers, optional EC2 GPU workers that scale from zero,
    queue-based autoscaling, and a daily retention purge.
  * Monitoring and access: CloudWatch alarms and dashboard, budget alerts, least-privilege IAM.
  * Checks: `validate` passes and offline `terraform test` (mocked providers) covers the CPU staging and
    GPU production plans plus the safety preconditions.
* Hardening:
  * Webhooks to internal addresses are refused.
  * A non-local deployment refuses development defaults.
  * JSON logs, plus CloudWatch EMF job metrics (count, seconds, real-time factor).
* GPU benchmarking: the benchmark reports device and peak GPU memory. The run itself needs a GPU.
* Licence register and CycloneDX SBOM (`ras sbom`), generated from the shipped image.
* CI workflow: lint, tests, Terraform checks, and the compose integration tests.
* Not done, needs an account or a decision:
  * The actual AWS apply.
  * The GPU benchmark run.
  * Database migrations tooling. The schema is created on start; add Alembic before the first schema
    change on a live database.
  * Fine-tuning runs. The tooling exists (`ras finetune`); training needs a GPU and real recordings.

## Client criteria coverage and open-source models: built, measured on synthetic audio
Guide: [CRITERIA.md](CRITERIA.md).
* Every criterion on TalentGum's requirements form (section 6, 22 items) has a metric and a place in
  `data/rubrics/talentgum_criteria_v1.json`. `ras criteria` prints the map.
* New deterministic measures: note order, per-chord missing/extra/wrong/voicing, rhythm ratios,
  hesitations and a fluency index, and tempo against the target. Articulation, dynamics and pedal are
  checked against the MusicXML marks, which the reference loader now reads (articulations, slurs,
  dynamics, hairpins, pedal).
* Open-source models, downloaded with checksums and loaded through an allow-list:
  * hFT-Transformer as a second transcriber that withdraws doubtful error claims.
  * Kong's model as a pedal source.
  * An audio dynamics model for keyboards without velocity.

  Each is called only when a rubric criterion needs it, and a missing model makes its criterion "not
  assessed".
* Rubric fallbacks (teacher benchmark first, score second) and a reason for every criterion that is not
  assessed.
* Teacher benchmark profiles: `ras profile build`, and `POST /v1/songs/{id}/teacher-profile`, which the
  worker then uses automatically.
* Learning from client data without training a network: `ras calibrate` fits thresholds and weights to
  teacher grades. `ras finetune prepare/export` builds a Transkun fine-tuning set by aligning teacher
  recordings to the score (there is no MIDI), plus a start checkpoint; training needs a GPU.
* Not done, needs client data or a decision: see the end of [CRITERIA.md](CRITERIA.md).

## Class-recording annotation: first version built, tested on synthetic classes
TalentGum's data is 40-60 minute online classes. `ras segment` cuts each one into one clip per song attempt
(transcribe in chunks, find the playing, identify the song among the five, split repeated and abandoned
attempts) and writes an annotation JSON, the transcript and the clips; `ras segment-eval` scores it against a
teacher's labels. On a rendered 8 minute synthetic class it found 6 of 6 attempts with no false clips.
Method, output format, trial and open items: [CLASS_RECORDINGS.md](CLASS_RECORDINGS.md).
* Not done: the teacher review screen for clips, a "score only the part played" mode for partial attempts,
  a rule for one-handed playing in the assessment, and any test on real classes.
