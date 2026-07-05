# PyGeoVision — Azure AKS deployment
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 3.0" }
    helm    = { source = "hashicorp/helm",    version = "~> 2.0" }
  }
}

provider "azurerm" { features {} }

resource "azurerm_resource_group" "pgv" {
  name     = "pygeovision-${var.environment}"
  location = var.location
}

resource "azurerm_kubernetes_cluster" "pgv" {
  name                = "pygeovision-${var.environment}"
  location            = azurerm_resource_group.pgv.location
  resource_group_name = azurerm_resource_group.pgv.name
  dns_prefix          = "pygeovision"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"
    enable_auto_scaling = true
    min_count  = 2
    max_count  = 10
  }

  identity { type = "SystemAssigned" }
}

resource "azurerm_storage_account" "pgv" {
  name                     = "pgvdata${var.environment}"
  resource_group_name      = azurerm_resource_group.pgv.name
  location                 = azurerm_resource_group.pgv.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}
