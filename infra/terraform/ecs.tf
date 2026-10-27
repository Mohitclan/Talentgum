# One image, three roles: API (Fargate, behind the load balancer), workers (Fargate CPU, or EC2 GPU when
# worker_mode = "gpu") and an admin task (daily retention purge; one-off tenant/key commands).

resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  for_each          = toset(["api", "worker", "admin"])
  name              = "/ras/${var.environment}/${each.key}"
  retention_in_days = var.log_retention_days
}

locals {
  app_env = [
    { name = "RAS_ENV", value = var.environment },
    { name = "RAS_STORAGE", value = "s3" },
    { name = "RAS_S3_BUCKET", value = aws_s3_bucket.objects.bucket },
    { name = "RAS_QUEUE", value = "sqs" },
    { name = "RAS_SQS_QUEUE_URL", value = aws_sqs_queue.jobs.url },
    { name = "RAS_DB_HOST", value = aws_db_instance.db.address },
    { name = "RAS_DB_NAME", value = "ras" },
    { name = "RAS_DB_USER", value = "ras" },
    { name = "RAS_DB_SSLMODE", value = "require" },
    { name = "RAS_LOG_FORMAT", value = "json" },
    { name = "RAS_TRANSCRIBER", value = var.transcriber },
    { name = "AWS_DEFAULT_REGION", value = var.region },
  ]
  app_secrets = [
    { name = "RAS_DB_PASSWORD", valueFrom = "${aws_db_instance.db.master_user_secret[0].secret_arn}:password::" },
    { name = "RAS_WEBHOOK_SECRET", valueFrom = aws_secretsmanager_secret.webhook.arn },
  ]
  logs = { for k, g in aws_cloudwatch_log_group.app : k => {
    logDriver = "awslogs"
    options   = { "awslogs-group" = g.name, "awslogs-region" = var.region, "awslogs-stream-prefix" = k }
  } }
  network = {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = false
  }
}

# ---- API
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name             = "api"
    image            = local.image
    essential        = true
    command          = ["ras", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-with-worker"]
    portMappings     = [{ containerPort = 8000, protocol = "tcp" }]
    environment      = local.app_env
    secrets          = local.app_secrets
    logConfiguration = local.logs["api"]
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_service" "api" {
  name                              = "api"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = var.api_desired_count
  launch_type                       = "FARGATE"
  enable_execute_command            = true
  health_check_grace_period_seconds = 60
  network_configuration {
    subnets          = local.network.subnets
    security_groups  = local.network.security_groups
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  depends_on = [aws_lb_listener.http, aws_lb_listener.https]
}

# ---- workers
resource "aws_ecs_task_definition" "worker_cpu" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  ephemeral_storage {
    size_in_gib = 30
  }
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name             = "worker"
    image            = local.image
    essential        = true
    command          = ["ras", "worker"]
    stopTimeout      = 120 # finish the current recording on scale-in
    environment      = concat(local.app_env, [{ name = "RAS_DEVICE", value = "cpu" }])
    secrets          = local.app_secrets
    logConfiguration = local.logs["worker"]
  }])
}

resource "aws_ecs_task_definition" "worker_gpu" {
  count                    = var.worker_mode == "gpu" ? 1 : 0
  family                   = "${local.name}-worker-gpu"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  cpu                      = 3584
  memory                   = 14336
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name                 = "worker"
    image                = "${aws_ecr_repository.ras.repository_url}:${var.gpu_image_tag}"
    essential            = true
    command              = ["ras", "worker"]
    stopTimeout          = 120
    resourceRequirements = [{ type = "GPU", value = "1" }]
    environment          = concat(local.app_env, [{ name = "RAS_DEVICE", value = "cuda" }])
    secrets              = local.app_secrets
    logConfiguration     = local.logs["worker"]
  }])
  lifecycle {
    precondition {
      condition     = var.gpu_image_tag != ""
      error_message = "worker_mode = gpu needs gpu_image_tag (an image built with the CUDA torch wheels)"
    }
  }
}

resource "aws_ecs_service" "worker" {
  name                   = "worker"
  cluster                = aws_ecs_cluster.main.id
  task_definition        = var.worker_mode == "gpu" ? aws_ecs_task_definition.worker_gpu[0].arn : aws_ecs_task_definition.worker_cpu.arn
  desired_count          = var.worker_min
  launch_type            = var.worker_mode == "cpu" ? "FARGATE" : null
  enable_execute_command = true
  network_configuration {
    subnets          = local.network.subnets
    security_groups  = local.network.security_groups
    assign_public_ip = false
  }
  dynamic "capacity_provider_strategy" {
    for_each = var.worker_mode == "gpu" ? [1] : []
    content {
      capacity_provider = aws_ecs_capacity_provider.gpu[0].name
      weight            = 1
    }
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  lifecycle {
    ignore_changes = [desired_count] # owned by autoscaling
  }
  depends_on = [aws_ecs_cluster_capacity_providers.main]
}

# scale workers on the backlog of job signals
resource "aws_appautoscaling_target" "worker" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.worker_min
  max_capacity       = var.worker_max
}

resource "aws_appautoscaling_policy" "worker_out" {
  name               = "${local.name}-worker-out"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 120
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_lower_bound = 0
      metric_interval_upper_bound = 10
      scaling_adjustment          = 1
    }
    step_adjustment {
      metric_interval_lower_bound = 10
      scaling_adjustment          = 2
    }
  }
}

resource "aws_appautoscaling_policy" "worker_in" {
  name               = "${local.name}-worker-in"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 300
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "backlog_high" {
  alarm_name          = "${local.name}-backlog-high"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.jobs.name }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  alarm_actions       = [aws_appautoscaling_policy.worker_out.arn]
}

resource "aws_cloudwatch_metric_alarm" "backlog_empty" {
  alarm_name          = "${local.name}-backlog-empty"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.jobs.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 3
  threshold           = 0
  comparison_operator = "LessThanOrEqualToThreshold"
  alarm_actions       = [aws_appautoscaling_policy.worker_in.arn]
}

# ---- admin task: daily retention purge, and one-off commands via `aws ecs run-task` overrides
resource "aws_ecs_task_definition" "admin" {
  family                   = "${local.name}-admin"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name             = "admin"
    image            = local.image
    essential        = true
    command          = ["ras", "--help"]
    environment      = local.app_env
    secrets          = local.app_secrets
    logConfiguration = local.logs["admin"]
  }])
}

resource "aws_cloudwatch_event_rule" "purge" {
  name                = "${local.name}-retention-purge"
  description         = "Delete recordings older than ${var.audio_retention_days} days"
  schedule_expression = "cron(30 21 * * ? *)" # 03:00 IST
}

resource "aws_cloudwatch_event_target" "purge" {
  rule     = aws_cloudwatch_event_rule.purge.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.events.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.admin.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = local.network.subnets
      security_groups  = local.network.security_groups
      assign_public_ip = false
    }
  }
  input = jsonencode({ containerOverrides = [{
    name    = "admin"
    command = ["ras", "purge-audio", "--older-than-days", tostring(var.audio_retention_days)]
  }] })
}
