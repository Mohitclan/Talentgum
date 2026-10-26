# Delivery phases

## Phase 1: Core engine (CLI, local): built
Setup, synthetic data tool, the full pipeline as a CLI, and a benchmark harness.
Done when injected errors are detected at the expected precision/recall on synthetic data and the
model comparison is measured locally. See `README.md` and the latest `data/runs/*/report.md`.

## Phase 2: Service and review (API, worker, UI, local): next
* `/v1` API: uploads, songs, rubrics, assessments; async jobs with retries, dead-letter state and
  idempotency keys; every result records model, rubric, reference and code versions.
* Local stand-ins for AWS behind interfaces: filesystem for S3, SQLite/Postgres for RDS,
  DB-backed queue for SQS.
* Teacher review endpoint, corrected-note storage, evaluation grouped by student and
  leave-one-song-out (the splits already exist in `ras/eval/splits.py`).
* Minimal web UI: upload a take, view score, per-note results, score coloured by error.
* Loading the real client data and producing the Research Plan's metrics.
* Confidence calibration on teacher-reviewed recordings; restart-aware matching.

## Phase 3: AWS and GPU readiness: on hold until an AWS account and GPU exist
Dockerfiles, S3/SQS adapters, ECS/Fargate definitions, CloudWatch/budget alerts, GPU benchmarking and
conditional fine-tuning (only if Phase 1/2 show systematic transcription errors), licence and SBOM register.
