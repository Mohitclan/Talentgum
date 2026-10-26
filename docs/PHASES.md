# Delivery phases

## Phase 1: Core engine (CLI, local): built, gaps closed out (see benchmarks/phase1_closeout.md)
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
* Not done: restart-aware matching, confidence calibration on real reviews (the tool exists, the data does not),
  multi-tenant isolation, authentication beyond a shared API key, S3/SQS adapters (Phase 3).

## Phase 3: AWS and GPU readiness: on hold until an AWS account and GPU exist
Dockerfiles, S3/SQS adapters, ECS/Fargate definitions, CloudWatch/budget alerts, GPU benchmarking and
conditional fine-tuning (only if Phase 1/2 show systematic transcription errors), licence and SBOM register.
