# Phase 8: refresh pipeline IaC — standard AWS provider, deployed against the
# floci local AWS emulator (port 4566). Same shape a real account would take;
# the only emulator-specific part is the endpoints block.
# Env: TF_ENDPOINT_HOST (default localhost:4566).

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40"
    }
  }
}


variable "endpoint_host" {
  description = "floci endpoint host:port override; empty = localhost:4566."
  type        = string
  default     = ""
}
variable "image_uri" {
  description = "Local container image URI for the Lambda (built from pipeline/Dockerfile.lambda)."
  type        = string
  default     = "workbench-refresh-lambda:latest"
}

locals {
  host    = var.endpoint_host != "" ? var.endpoint_host : "localhost:4566"
  region  = "us-east-1"
  account = "000000000000"
}

provider "aws" {
  access_key = "test"
  secret_key = "test"
  region     = local.region

  s3_use_path_style = true

  endpoints {
    s3     = "http://${local.host}"
    lambda = "http://${local.host}"
    iam    = "http://${local.host}"
    logs   = "http://${local.host}"
  }

  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
}

resource "aws_s3_bucket" "snapshots" {
  bucket        = "workbench-gated-snapshots"
  force_destroy = true
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "refresh" {
  name               = "workbench-refresh-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy" "s3_put" {
  name = "put-snapshots"
  role = aws_iam_role.refresh.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
      Resource = ["${aws_s3_bucket.snapshots.arn}", "${aws_s3_bucket.snapshots.arn}/*"]
    }]
  })
}

resource "aws_lambda_function" "refresh" {
  function_name = "workbench-refresh"
  package_type  = "Image"
  image_uri     = var.image_uri
  role          = aws_iam_role.refresh.arn
  timeout       = 900
  memory_size   = 1024

  environment {
    variables = {
      REFRESH_BUCKET = aws_s3_bucket.snapshots.id
    }
  }
}

output "bucket" {
  value = aws_s3_bucket.snapshots.id
}

output "function_name" {
  value = aws_lambda_function.refresh.function_name
}
