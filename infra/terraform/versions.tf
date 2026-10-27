terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # State: use an S3 backend per environment, e.g.
  #   terraform init -backend-config="bucket=<state-bucket>" -backend-config="key=ras/staging.tfstate" \
  #                  -backend-config="region=ap-south-1" -backend-config="use_lockfile=true"
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ras", Environment = var.environment, ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name       = "ras-${var.environment}"
  account_id = data.aws_caller_identity.current.account_id
  production = var.environment == "production"
  image      = "${aws_ecr_repository.ras.repository_url}:${var.image_tag}"
}
