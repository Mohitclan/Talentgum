# PostgreSQL on RDS. The master password is generated and rotated by RDS in Secrets Manager; ECS injects
# it into the containers, so it never appears in Terraform state or task definitions.

resource "aws_db_subnet_group" "db" {
  name       = local.name
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_parameter_group" "db" {
  name   = "${local.name}-pg16"
  family = "postgres16"
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
  parameter {
    name  = "log_min_duration_statement"
    value = "1000" # log queries slower than 1 s
  }
}

resource "aws_db_instance" "db" {
  identifier                      = local.name
  engine                          = "postgres"
  engine_version                  = "16"
  instance_class                  = var.db_instance_class
  allocated_storage               = var.db_allocated_storage_gb
  max_allocated_storage           = var.db_allocated_storage_gb * 5
  storage_type                    = "gp3"
  storage_encrypted               = true
  db_name                         = "ras"
  username                        = "ras"
  manage_master_user_password     = true
  db_subnet_group_name            = aws_db_subnet_group.db.name
  vpc_security_group_ids          = [aws_security_group.db.id]
  parameter_group_name            = aws_db_parameter_group.db.name
  multi_az                        = coalesce(var.db_multi_az, local.production)
  publicly_accessible             = false
  backup_retention_period         = local.production ? 14 : 3
  deletion_protection             = local.production
  skip_final_snapshot             = !local.production
  final_snapshot_identifier       = local.production ? "${local.name}-final" : null
  auto_minor_version_upgrade      = true
  copy_tags_to_snapshot           = true
  enabled_cloudwatch_logs_exports = ["postgresql"]
}
