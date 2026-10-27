# Offline plan tests with mocked providers (no AWS account): `terraform test`

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "111122223333" }
  }
  mock_data "aws_ssm_parameter" {
    defaults = { value = "ami-0123456789abcdef0" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
}

mock_provider "random" {}

variables {
  environment        = "staging"
  vpc_id             = "vpc-0123456789abcdef0"
  public_subnet_ids  = ["subnet-a", "subnet-b"]
  private_subnet_ids = ["subnet-c", "subnet-d"]
  image_tag          = "test"
  alert_email        = "ops@example.com"
}

run "cpu_staging" {
  command = plan

  assert {
    condition     = aws_ecs_service.worker.launch_type == "FARGATE"
    error_message = "CPU workers run on Fargate"
  }
  assert {
    condition     = length(aws_autoscaling_group.gpu) == 0 && length(aws_ecs_task_definition.worker_gpu) == 0
    error_message = "no GPU resources in cpu mode"
  }
  assert {
    condition     = length(aws_lb_listener.http) == 1 && length(aws_lb_listener.https) == 0
    error_message = "staging without a certificate serves plain HTTP"
  }
  assert {
    condition     = aws_db_instance.db.deletion_protection == false && aws_db_instance.db.multi_az == false
    error_message = "staging database is disposable and single-AZ"
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.objects.block_public_acls && aws_s3_bucket_public_access_block.objects.restrict_public_buckets
    error_message = "recordings bucket must block public access"
  }
  assert {
    condition     = aws_s3_bucket_lifecycle_configuration.objects.rule[0].expiration[0].days == 120
    error_message = "lifecycle backstop is retention + 30 days"
  }
  assert {
    condition     = length(aws_cloudwatch_metric_alarm.ops) == 6 && length(aws_budgets_budget.monthly.notification) == 3
    error_message = "alarms and budget notifications"
  }
}

run "gpu_production" {
  command = plan
  variables {
    environment     = "production"
    worker_mode     = "gpu"
    gpu_image_tag   = "test-cu121"
    certificate_arn = "arn:aws:acm:ap-south-1:111122223333:certificate/abc"
  }

  assert {
    condition     = length(aws_autoscaling_group.gpu) == 1 && aws_autoscaling_group.gpu[0].min_size == 0
    error_message = "GPU instances scale from zero"
  }
  assert {
    condition     = length(aws_ecs_service.worker.capacity_provider_strategy) == 1
    error_message = "GPU workers use the GPU capacity provider"
  }
  assert {
    condition     = length(aws_lb_listener.https) == 1 && length(aws_lb_listener.http) == 0
    error_message = "production serves HTTPS only"
  }
  assert {
    condition     = aws_db_instance.db.deletion_protection && aws_db_instance.db.multi_az
    error_message = "production database is protected and Multi-AZ"
  }
}

run "production_requires_https" {
  command = plan
  variables {
    environment = "production"
  }
  expect_failures = [aws_lb.api]
}

run "gpu_requires_gpu_image" {
  command = plan
  variables {
    worker_mode = "gpu"
  }
  expect_failures = [aws_ecs_task_definition.worker_gpu]
}
