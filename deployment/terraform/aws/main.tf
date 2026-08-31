# PyGeoVision — AWS EKS deployment
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    helm = { source = "hashicorp/helm", version = "~> 2.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "pygeovision-${var.environment}"
  cluster_version = "1.29"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  eks_managed_node_groups = {
    cpu = {
      instance_types = ["c5.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
    gpu = {
      instance_types = ["g4dn.xlarge"]
      min_size       = 0
      max_size       = 5
      desired_size   = 0
      ami_type       = "AL2_x86_64_GPU"
    }
  }
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  name    = "pygeovision-${var.environment}"
  cidr    = "10.0.0.0/16"
  azs     = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  enable_nat_gateway = true
}

# S3 bucket for satellite data cache
resource "aws_s3_bucket" "pgv_data" {
  bucket = "pygeovision-data-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "pgv_data" {
  bucket = aws_s3_bucket.pgv_data.id
  versioning_configuration { status = "Enabled" }
}

data "aws_caller_identity" "current" {}

# Helm release
resource "helm_release" "pygeovision" {
  name       = "pygeovision"
  repository = "https://appiahkubis14.github.io/pygeovision-helm"
  chart      = "pygeovision"
  version    = "2.1.7"
  namespace  = "pygeovision"
  create_namespace = true

  values = [templatefile("${path.module}/values.yaml", {
    environment = var.environment
    s3_bucket   = aws_s3_bucket.pgv_data.id
  })]

  set_sensitive {
    name  = "secrets.anthropicApiKey"
    value = var.anthropic_api_key
  }
}
