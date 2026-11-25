terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.14"
    }
    # Used where azurerm lacks a resource (Azure Managed Redis).
    azapi = {
      source  = "Azure/azapi"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state in Azure Storage with blob-lease locking (no DynamoDB; that is AWS).
  # Values are supplied at init time: terraform init -backend-config=envs/dev.backend.hcl
  backend "azurerm" {
    use_azuread_auth = true
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
  storage_use_azuread = true
}

provider "azapi" {}
