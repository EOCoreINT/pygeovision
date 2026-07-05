output "eks_cluster_endpoint"   { value = module.eks.cluster_endpoint }
output "eks_cluster_name"       { value = module.eks.cluster_name }
output "s3_data_bucket"         { value = aws_s3_bucket.pgv_data.id }
output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
