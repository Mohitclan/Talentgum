# Alarms go to one SNS topic (email). Job metrics come from the worker's Embedded Metric Format log lines
# (namespace RAS: Jobs, JobSeconds, RealTimeFactor by Status), so no agent or extra API calls are needed.

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

locals {
  alarms = {
    api-5xx = {
      namespace = "AWS/ApplicationELB", metric = "HTTPCode_Target_5XX_Count", stat = "Sum", period = 300,
      threshold = 5, op = "GreaterThanThreshold", dims = { LoadBalancer = aws_lb.api.arn_suffix },
      what      = "API returned more than 5 server errors in 5 minutes"
    }
    api-unhealthy = {
      namespace = "AWS/ApplicationELB", metric = "UnHealthyHostCount", stat = "Maximum", period = 60,
      threshold = 0, op = "GreaterThanThreshold",
      dims      = { LoadBalancer = aws_lb.api.arn_suffix, TargetGroup = aws_lb_target_group.api.arn_suffix },
      what      = "An API task is failing its health check"
    }
    backlog-old = {
      namespace = "AWS/SQS", metric = "ApproximateAgeOfOldestMessage", stat = "Maximum", period = 300,
      threshold = 900, op = "GreaterThanThreshold", dims = { QueueName = aws_sqs_queue.jobs.name },
      what      = "Assessments have waited more than 15 minutes for a worker"
    }
    jobs-failed = {
      namespace = "RAS", metric = "Jobs", stat = "Sum", period = 900, threshold = 0,
      op        = "GreaterThanThreshold", dims = { Status = "failed" },
      what      = "An assessment failed after all retries (dead-lettered)"
    }
    db-cpu = {
      namespace = "AWS/RDS", metric = "CPUUtilization", stat = "Average", period = 300, threshold = 80,
      op        = "GreaterThanThreshold", dims = { DBInstanceIdentifier = aws_db_instance.db.identifier },
      what      = "Database CPU above 80%"
    }
    db-storage = {
      namespace = "AWS/RDS", metric = "FreeStorageSpace", stat = "Minimum", period = 300,
      threshold = 2147483648, op = "LessThanThreshold",
      dims      = { DBInstanceIdentifier = aws_db_instance.db.identifier }, what = "Database has under 2 GB free"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "ops" {
  for_each            = local.alarms
  alarm_name          = "${local.name}-${each.key}"
  alarm_description   = each.value.what
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  dimensions          = each.value.dims
  statistic           = each.value.stat
  period              = each.value.period
  evaluation_periods  = 1
  threshold           = each.value.threshold
  comparison_operator = each.value.op
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Environment$%s", var.environment)] # activate the Environment cost allocation tag
  }
  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [
      { type = "ACTUAL", pct = 80 }, { type = "ACTUAL", pct = 100 }, { type = "FORECASTED", pct = 100 }
    ]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value.pct
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value.type
      subscriber_email_addresses = [var.alert_email]
    }
  }
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = local.name
  dashboard_body = jsonencode({ widgets = [
    { type = "metric", x = 0, y = 0, width = 12, height = 6, properties = {
      title   = "Assessments by outcome", region = var.region, stat = "Sum", period = 300, view = "timeSeries",
      metrics = [for s in ["completed", "review_required", "rejected", "failed"] : ["RAS", "Jobs", "Status", s]]
    } },
    { type = "metric", x = 12, y = 0, width = 12, height = 6, properties = {
      title = "Seconds per job (p50, p90) and real-time factor", region = var.region, period = 300,
      metrics = [["RAS", "JobSeconds", { stat = "p50" }], ["RAS", "JobSeconds", { stat = "p90" }],
      ["RAS", "RealTimeFactor", { stat = "Average", yAxis = "right" }]]
    } },
    { type = "metric", x = 0, y = 6, width = 12, height = 6, properties = {
      title = "Backlog", region = var.region, stat = "Maximum", period = 60,
      metrics = [["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", aws_sqs_queue.jobs.name],
      ["AWS/SQS", "ApproximateAgeOfOldestMessage", "QueueName", aws_sqs_queue.jobs.name, { yAxis = "right" }]]
    } },
    { type = "metric", x = 12, y = 6, width = 12, height = 6, properties = {
      title = "API", region = var.region, stat = "Sum", period = 300,
      metrics = [["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.api.arn_suffix],
      ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", aws_lb.api.arn_suffix]]
    } },
  ] })
}
