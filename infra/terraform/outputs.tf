output "api_url" {
  value = "${var.certificate_arn == "" ? "http" : "https"}://${aws_lb.api.dns_name}"
}

output "ecr_repository" {
  value = aws_ecr_repository.ras.repository_url
}

output "bucket" {
  value = aws_s3_bucket.objects.bucket
}

output "queue_url" {
  value = aws_sqs_queue.jobs.url
}

output "db_endpoint" {
  value = aws_db_instance.db.address
}

output "cluster" {
  value = aws_ecs_cluster.main.name
}

output "webhook_secret_arn" {
  description = "Read it once with `aws secretsmanager get-secret-value` and give it to the webhook receiver"
  value       = aws_secretsmanager_secret.webhook.arn
}

output "admin_run_task" {
  description = "Template for one-off admin commands, e.g. creating a tenant"
  value = join(" ", [
    "aws ecs run-task --cluster ${aws_ecs_cluster.main.name} --launch-type FARGATE",
    "--task-definition ${aws_ecs_task_definition.admin.family}",
    "--network-configuration 'awsvpcConfiguration={subnets=[${join(",", var.private_subnet_ids)}],securityGroups=[${aws_security_group.app.id}]}'",
    "--overrides '{\"containerOverrides\":[{\"name\":\"admin\",\"command\":[\"ras\",\"tenant-create\",\"talentgum\"]}]}'",
  ])
}
