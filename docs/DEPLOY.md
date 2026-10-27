# Deployment (Phase 3: AWS and GPU readiness)

Everything here is built and tested locally. **Nothing has been deployed to AWS**: no account exists yet.
What runs on AWS is the same image and the same code paths that the local stack exercises.

```
              HTTPS                       ┌──────────── private subnets ─────────────┐
 client ───► ALB ───► API (Fargate ×2) ───┤  RDS PostgreSQL 16 (encrypted, TLS only)  │
                         │   │            │  S3 bucket (private, SSE, TLS only)       │
                         │   └─ SQS ─────►│  Workers (Fargate CPU, or EC2 GPU)        │
                         │  wake-up       │  Admin task (daily retention purge)       │
                         └────────────────┴───────────────────────────────────────────┘
 CloudWatch: JSON logs, EMF job metrics, alarms → SNS email, dashboard; AWS Budgets → email
```

| Local (Phases 1–2) | AWS (Phase 3) | Switch |
|---|---|---|
| SQLite | RDS PostgreSQL | `RAS_DB_URL` or `RAS_DB_HOST` + secret |
| files under `data/service/objects` | S3 bucket | `RAS_STORAGE=s3`, `RAS_S3_BUCKET` |
| workers poll the table | SQS long-poll wake-ups | `RAS_QUEUE=sqs`, `RAS_SQS_QUEUE_URL` |
| CPU | CPU or CUDA | `RAS_DEVICE=cpu|cuda|auto` |

The assessments table stays the job queue of record. An SQS message only says "there may be work": a
lost or duplicated message can neither lose nor double-run an assessment, and a crashed worker's job is
recovered after the visibility timeout (15 min) exactly as before.

## 1. Run the AWS shape locally (Docker)

PostgreSQL stands in for RDS, SeaweedFS for S3, ElasticMQ for SQS.

```bash
docker compose up -d --build
```

UI and API: http://127.0.0.1:8088/ui/. One worker runs Transkun on CPU.

```bash
.venv/bin/python scripts/smoke_stack.py --url http://127.0.0.1:8088 --recording rec_001
```

```bash
docker compose run --rm tests
```

```bash
docker compose down -v
```

`docker compose run --rm tests` runs the integration tests against their own database, bucket and queue:
concurrent schema creation, S3 objects and tags, an assessment on all three backends, erasure removing the
S3 object, and three workers racing for six jobs (each job claimed exactly once).

Measured on 2026-10-27, Intel Mac with Docker Desktop:
- **Integration tests:** 4 of 4 pass.
- **Full suite in the Linux image:** 88 passed.
- **Smoke test** (real Transkun in the worker container, synthetic recordings):

| run | audio | job time | real-time factor |
|---|---|---|---|
| first job (includes model load) | 21 s | 34.6 s | 1.62 |
| warm worker | 24 s | 29.3 s | 1.22 |

- **First job's detection:** all 4 injected wrong notes and the 1 wrong octave were found, and the result
  was routed to review.
- **Graceful stop:** `docker compose stop worker` finished in 20 s (the SQS long poll) and logged
  "worker stopped".
- **Running cost:** the real-time factor above 1 means a CPU worker of this size handles about one audio
  minute per wall minute. Size `worker_max` for the peak submission rate, then re-measure on Fargate.

## 2. Deploy to AWS (when the account exists)

Prerequisites, once per account:
1. Region: `ap-south-1` (Mumbai) by default. Confirm where TalentGum needs student data to live.
2. A VPC with two public and two private subnets; private subnets need NAT or VPC endpoints.
3. An ACM certificate for the API hostname (production refuses to plan without one).
4. An S3 bucket for Terraform state.
5. Activate the `Environment` cost allocation tag in Billing, so the budget filters correctly.

Build and push the image (tags are immutable; use date and commit):

```bash
TAG=$(date +%Y-%m-%d)-$(git rev-parse --short HEAD)
```

```bash
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com
```

```bash
docker build --platform linux/amd64 -t <ecr-repo>:$TAG . && docker push <ecr-repo>:$TAG
```

The ECR repository is created by Terraform. On the very first run, apply once with
`-target=aws_ecr_repository.ras`, push, then apply everything.

```bash
cd infra/terraform && cp staging.tfvars.example staging.tfvars
```

```bash
terraform init -backend-config="bucket=<state-bucket>" -backend-config="key=ras/staging.tfstate" -backend-config="region=ap-south-1"
```

```bash
terraform apply -var-file=staging.tfvars
```

After the first apply:
- Create the client tenant with the `admin_run_task` output command (`ras tenant-create talentgum`). The
  API key is printed once, in the admin task's CloudWatch log group `/ras/<env>/admin`. Copy it, then
  delete that log stream.
- Read the webhook signing secret once (`aws secretsmanager get-secret-value --secret-id <webhook_secret_arn>`)
  and give it to the system that receives webhooks.
- Confirm the SNS subscription email.
- Run `scripts/smoke_stack.py --url <api_url> --api-key <key>`.

What Terraform creates:
- **Storage:** an S3 bucket with public access blocked, encryption, TLS only and no versioning (so an
  erasure really deletes). Lifecycle backstop at retention + 30 days.
- **Queue:** an SQS queue.
- **Database:** RDS PostgreSQL 16 with encryption and forced TLS. The master password is kept and rotated
  by RDS in Secrets Manager.
- **Images:** an ECR repository with scan on push and immutable tags.
- **Compute:** an ECS cluster with Container Insights, an ALB, and the API service with circuit-breaker
  rollback.
- **Workers:** autoscaled on SQS backlog (min 1, so stuck-job recovery always runs).
- **Retention:** a daily EventBridge purge task.
- **Monitoring:**
  - Six alarms: 5xx, unhealthy API, jobs waiting over 15 min, dead-lettered jobs, database CPU, database
    storage.
  - A dashboard.
  - A monthly budget with alerts at 80% and 100% actual and 100% forecast.

Security, in brief:
- IAM is least privilege: one bucket and one queue, and the two secrets are read only at task start.
- Task containers cannot reach the instance metadata service: `ECS_AWSVPC_BLOCK_IMDS` on GPU hosts,
  plus IMDSv2 with hop limit 1.
- Webhooks are refused to private, loopback or link-local addresses, and must use https outside local
  development.
- A non-local deployment refuses to start with the development webhook secret, SQLite or local storage.

## 3. GPU: benchmark first, then decide

The CPU benchmark (Phase 1): Transkun took 57.5 CPU-seconds per audio minute, with 1.47 GB peak RAM.
Switch `worker_mode = "gpu"` only if a measured GPU run gives a cost or latency win:

1. Build the CUDA image:

   ```bash
   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t ras-gpu .
   ```

2. On one `g4dn.xlarge` (ECS GPU AMI or Deep Learning AMI), with a copy of `data/synth` (or real
   recordings with ground truth):

   ```bash
   docker run --gpus all -e RAS_DEVICE=cuda -v $PWD/data:/app/data ras-gpu ras benchmark --data data/synth --models transkun
   ```

   The report now shows device, transcription seconds per audio minute and peak GPU memory.

3. Cost per audio hour = instance price per hour × (transcription seconds per audio minute ÷ 60). Compare it
   with the Fargate CPU worker (2 vCPU, 4 GB) at its measured speed. Choose GPU if it is cheaper at
   expected volume, or if the turnaround target needs it.

## 4. Fine-tuning: tooling ready, run only when the data says so

Fine-tuning Transkun is worth it only if teacher reviews (`ras evaluate`) show a **systematic**
transcription error on real recordings, for example one keyboard model's timbre or a phone
microphone. Isolated mistakes do not count. Prerequisites:
- At least a few hours of teacher or reviewed recordings.
- A held-out set of students (`ras.eval.splits`).
- A GPU.

There is no MIDI for these recordings, so `ras finetune prepare items.json --out data/finetune` builds
labels another way:
- Each recording is aligned to its score.
- A teacher recording is taken as a correct performance; a reviewed recording uses the teacher's
  corrections.
- It writes Transkun's dataset files and a start checkpoint holding the released V2 weights, then
  prints the training command (`python -m transkun.train ... --max_lr 2e-5 --augment`).

Then:
1. Run `ras finetune export checkpoint.pt --out transkun_talentgum.pt`.
2. Benchmark it against the released model on held-out students.
3. Only if it is better, set `RAS_TRANSKUN_WEIGHT=/path/transkun_talentgum.pt` on the workers.

## 5. Licence register and SBOM

`ras sbom` writes a CycloneDX 1.5 SBOM and a licence table. It flags copyleft or unknown licences for a
decision. Run it inside the image that ships:

```bash
docker run --rm -v $PWD/docs/compliance:/out ras:local ras sbom --name image --out-dir /out
```

See `docs/compliance/` for the current register and the decisions taken.
