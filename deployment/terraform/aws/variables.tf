variable "aws_region"       { default = "us-east-1" }
variable "environment"      { default = "production" }
variable "anthropic_api_key" {
  description = "Anthropic API key for GeoAgent LLM planning"
  sensitive   = true
  default     = ""
}
