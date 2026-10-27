# Recordings and reference files. Private, encrypted, TLS only. No versioning: an erasure request must
# really remove the bytes. The application purges recordings after the retention period; the lifecycle
# rule is a backstop 30 days later for anything it missed (objects are tagged kind=audio on upload).

resource "aws_s3_bucket" "objects" {
  bucket        = "${local.name}-objects-${local.account_id}"
  force_destroy = !local.production
}

resource "aws_s3_bucket_public_access_block" "objects" {
  bucket                  = aws_s3_bucket.objects.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "objects" {
  bucket = aws_s3_bucket.objects.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "objects" {
  bucket = aws_s3_bucket.objects.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "objects" {
  bucket = aws_s3_bucket.objects.id
  rule {
    id     = "expire-recordings-backstop"
    status = "Enabled"
    filter {
      tag {
        key   = "kind"
        value = "audio"
      }
    }
    expiration {
      days = var.audio_retention_days + 30
    }
  }
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

data "aws_iam_policy_document" "objects_tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.objects.arn, "${aws_s3_bucket.objects.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "objects" {
  bucket     = aws_s3_bucket.objects.id
  policy     = data.aws_iam_policy_document.objects_tls_only.json
  depends_on = [aws_s3_bucket_public_access_block.objects]
}

# ---- job signals. The database is the job queue of record; messages only wake workers and give the
# autoscaler a backlog to act on, so a short retention is enough.
resource "aws_sqs_queue" "jobs" {
  name                       = "${local.name}-jobs"
  message_retention_seconds  = 86400
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 30
  sqs_managed_sse_enabled    = true
}

# ---- container images
resource "aws_ecr_repository" "ras" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "ras" {
  repository = aws_ecr_repository.ras.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "keep the last 30 images"
    selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 }
    action       = { type = "expire" }
  }] })
}

# ---- webhook signing secret (shared with the client's webhook receiver)
resource "random_password" "webhook" {
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "webhook" {
  name                    = "${local.name}/webhook-secret"
  recovery_window_in_days = local.production ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "webhook" {
  secret_id     = aws_secretsmanager_secret.webhook.id
  secret_string = random_password.webhook.result
}
