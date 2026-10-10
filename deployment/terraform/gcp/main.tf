# PyGeoVision — GCP GKE deployment
terraform {
  required_version = ">= 1.5"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
    helm   = { source = "hashicorp/helm",   version = "~> 2.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "pgv" {
  name     = "pygeovision-${var.environment}"
  location = var.region

  node_pool {
    name       = "cpu-pool"
    node_count = 3
    node_config {
      machine_type = "n2-standard-8"
      oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    }
    autoscaling {
      min_node_count = 2
      max_node_count = 10
    }
  }
}

resource "google_storage_bucket" "pgv_data" {
  name     = "pygeovision-data-${var.project_id}-${var.environment}"
  location = var.region
  versioning { enabled = true }
}
