terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Deliberately LOCAL state: this configuration creates the remote state storage, so it cannot
  # use it. Run it once as an Owner and keep terraform.tfstate safe.
}

provider "azurerm" {
  features {}
  # The state account has shared keys disabled, so the provider must use Entra for the data plane.
  storage_use_azuread = true
}

provider "azuread" {}
