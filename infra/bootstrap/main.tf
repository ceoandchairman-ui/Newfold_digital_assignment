# One-time bootstrap (run by an Owner, locally or from Cloud Shell). It creates what the pipeline
# needs BEFORE it can run at all:
#   1. remote Terraform state storage (Entra auth only, versioned),
#   2. one shared container registry (dev and prod pull the same signed image),
#   3. two GitHub OIDC identities: `plan` (read-only) and `apply` (can deploy), with no secrets.

data "azurerm_client_config" "current" {}
data "azurerm_subscription" "current" {}

resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
}

locals {
  tags = merge({ app = "support-assistant", managed_by = "terraform-bootstrap" }, var.tags)

  # GitHub environment name -> which identity it signs in as. plan-* environments are read-only.
  federated = {
    "plan-dev"         = { app = "plan", environment = "plan-dev" }
    "plan-prod"        = { app = "plan", environment = "plan-prod" }
    "apply-dev"        = { app = "apply", environment = "dev" }
    "apply-production" = { app = "apply", environment = "production" }
  }
}

# --- Remote state ------------------------------------------------------------------------------
resource "azurerm_resource_group" "state" {
  name     = "rg-${var.prefix}-tfstate"
  location = var.location
  tags     = local.tags
}

resource "azurerm_storage_account" "state" {
  name                            = "st${var.prefix}tf${random_string.suffix.result}"
  resource_group_name             = azurerm_resource_group.state.name
  location                        = azurerm_resource_group.state.location
  account_tier                    = "Standard"
  account_replication_type        = "GRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false # Entra only: there is no key to leak
  default_to_oauth_authentication = true
  tags                            = local.tags

  blob_properties {
    versioning_enabled = true # every state version is recoverable

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 30
    }
  }
}

resource "azurerm_storage_container" "state" {
  name                  = "tfstate"
  storage_account_id    = azurerm_storage_account.state.id
  container_access_type = "private"
}

# --- Shared container registry ------------------------------------------------------------------
resource "azurerm_resource_group" "shared" {
  name     = "rg-${var.prefix}-shared"
  location = var.location
  tags     = local.tags
}

resource "azurerm_container_registry" "shared" {
  name                = "acr${var.prefix}${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.shared.name
  location            = azurerm_resource_group.shared.location
  sku                 = "Standard"
  admin_enabled       = false
  tags                = local.tags
}

# --- GitHub OIDC identities (no stored secrets) --------------------------------------------------
resource "azuread_application" "github" {
  for_each     = toset(["plan", "apply"])
  display_name = "gh-${var.prefix}-${each.key}"
}

resource "azuread_service_principal" "github" {
  for_each  = azuread_application.github
  client_id = each.value.client_id
}

# Trust ONLY tokens GitHub issues for this repository and one named environment.
resource "azuread_application_federated_identity_credential" "github" {
  for_each       = local.federated
  application_id = azuread_application.github[each.value.app].id
  display_name   = each.key
  issuer         = "https://token.actions.githubusercontent.com"
  audiences      = ["api://AzureADTokenExchange"]
  subject        = "repo:${var.github_repository}:environment:${each.value.environment}"
}

# plan: can read everything and the state, and nothing else.
resource "azurerm_role_assignment" "plan_reader" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "Reader"
  principal_id         = azuread_service_principal.github["plan"].object_id
}

resource "azurerm_role_assignment" "plan_state" {
  scope                = azurerm_storage_account.state.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azuread_service_principal.github["plan"].object_id
}

# apply: builds and pushes the image, deploys infrastructure, and grants roles to the workload
# identities it creates (that needs User Access Administrator, which Contributor does not include).
resource "azurerm_role_assignment" "apply_contributor" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "Contributor"
  principal_id         = azuread_service_principal.github["apply"].object_id
}

resource "azurerm_role_assignment" "apply_access_admin" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "User Access Administrator"
  principal_id         = azuread_service_principal.github["apply"].object_id
}

resource "azurerm_role_assignment" "apply_state" {
  scope                = azurerm_storage_account.state.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azuread_service_principal.github["apply"].object_id
}
