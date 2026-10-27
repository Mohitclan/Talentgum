variable "region" {
  description = "AWS region. Keep student recordings in the client's required jurisdiction."
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "staging or production (production turns on deletion protection and Multi-AZ defaults)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production"
  }
}

# ---- network: an existing VPC. Private subnets need outbound access (NAT or VPC endpoints) for ECR,
# S3, SQS, Secrets Manager and CloudWatch Logs.
variable "vpc_id" { type = string }
variable "public_subnet_ids" {
  description = "For the load balancer (at least two availability zones)"
  type        = list(string)
}
variable "private_subnet_ids" {
  description = "For ECS tasks, GPU instances and the database (at least two availability zones)"
  type        = list(string)
}
variable "allowed_cidrs" {
  description = "Who may reach the API. Narrow this to the client's systems if possible."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Empty means plain HTTP (staging tests only)."
  type        = string
  default     = ""
}

# ---- application
variable "image_tag" {
  description = "Tag of the CPU image in ECR (built from the repository Dockerfile)"
  type        = string
}
variable "gpu_image_tag" {
  description = "Tag of the CUDA image (Dockerfile with TORCH_INDEX=.../cu121); used when worker_mode = gpu"
  type        = string
  default     = ""
}
variable "transcriber" {
  type    = string
  default = "transkun"
}
variable "api_desired_count" {
  type    = number
  default = 2
}
variable "api_cpu" {
  type    = number
  default = 512
}
variable "api_memory" {
  type    = number
  default = 1024
}

# ---- workers
variable "worker_mode" {
  description = "cpu: Fargate tasks. gpu: EC2 GPU instances (only after the GPU benchmark shows it pays off)."
  type        = string
  default     = "cpu"
  validation {
    condition     = contains(["cpu", "gpu"], var.worker_mode)
    error_message = "worker_mode must be cpu or gpu"
  }
}
variable "worker_cpu" {
  description = "Fargate vCPU units for a CPU worker (Transkun measured ~1 CPU-minute per audio minute)"
  type        = number
  default     = 2048
}
variable "worker_memory" {
  description = "MiB; Transkun peaked at 1.5 GB in the Phase 1 benchmark"
  type        = number
  default     = 4096
}
variable "worker_min" {
  description = "Keep at least one worker so stuck-job recovery always runs"
  type        = number
  default     = 1
}
variable "worker_max" {
  type    = number
  default = 4
}
variable "gpu_instance_type" {
  type    = string
  default = "g4dn.xlarge"
}
variable "gpu_ami_ssm_parameter" {
  type    = string
  default = "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
}

# ---- data
variable "db_instance_class" {
  type    = string
  default = "db.t4g.small"
}
variable "db_multi_az" {
  type    = bool
  default = null # null: on in production, off in staging
}
variable "db_allocated_storage_gb" {
  type    = number
  default = 20
}
variable "audio_retention_days" {
  description = "Raw recordings are deleted this many days after assessment (results and reviews are kept)"
  type        = number
  default     = 90
}

# ---- cost and alerts
variable "alert_email" {
  description = "Receives alarms and budget warnings (must confirm the SNS subscription email)"
  type        = string
  default     = ""
}
variable "monthly_budget_usd" {
  type    = number
  default = 300
}
variable "log_retention_days" {
  type    = number
  default = 30
}
