# GPU workers (worker_mode = "gpu"): an Auto Scaling group of ECS GPU-optimised instances, scaled by ECS
# from zero. Only switch on after the GPU benchmark (docs/DEPLOY.md) shows a cost or latency win.

data "aws_ssm_parameter" "gpu_ami" {
  count = var.worker_mode == "gpu" ? 1 : 0
  name  = var.gpu_ami_ssm_parameter
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu_instance" {
  count              = var.worker_mode == "gpu" ? 1 : 0
  name               = "${local.name}-gpu-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "gpu_instance" {
  for_each = var.worker_mode == "gpu" ? toset([
    "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
  ]) : toset([])
  role       = aws_iam_role.gpu_instance[0].name
  policy_arn = each.value
}

resource "aws_iam_instance_profile" "gpu" {
  count = var.worker_mode == "gpu" ? 1 : 0
  name  = "${local.name}-gpu"
  role  = aws_iam_role.gpu_instance[0].name
}

resource "aws_launch_template" "gpu" {
  count                  = var.worker_mode == "gpu" ? 1 : 0
  name_prefix            = "${local.name}-gpu-"
  image_id               = data.aws_ssm_parameter.gpu_ami[0].value
  instance_type          = var.gpu_instance_type
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile {
    arn = aws_iam_instance_profile.gpu[0].arn
  }
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 80
      volume_type = "gp3"
      encrypted   = true
    }
  }
  user_data = base64encode(<<-EOT
    #!/bin/bash
    cat >> /etc/ecs/ecs.config <<CFG
    ECS_CLUSTER=${aws_ecs_cluster.main.name}
    ECS_ENABLE_GPU_SUPPORT=true
    ECS_AWSVPC_BLOCK_IMDS=true
    ECS_ENABLE_SPOT_INSTANCE_DRAINING=true
    CFG
  EOT
  )
}

resource "aws_autoscaling_group" "gpu" {
  count                 = var.worker_mode == "gpu" ? 1 : 0
  name_prefix           = "${local.name}-gpu-"
  min_size              = 0
  max_size              = var.worker_max
  vpc_zone_identifier   = var.private_subnet_ids
  protect_from_scale_in = true
  launch_template {
    id      = aws_launch_template.gpu[0].id
    version = "$Latest"
  }
  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }
  lifecycle {
    ignore_changes = [desired_capacity]
  }
}

resource "aws_ecs_capacity_provider" "gpu" {
  count = var.worker_mode == "gpu" ? 1 : 0
  name  = "${local.name}-gpu"
  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu[0].arn
    managed_termination_protection = "ENABLED"
    managed_scaling {
      status          = "ENABLED"
      target_capacity = 100
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = concat(["FARGATE"], var.worker_mode == "gpu" ? [aws_ecs_capacity_provider.gpu[0].name] : [])
}
