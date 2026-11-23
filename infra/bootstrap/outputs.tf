# Copy these into GitHub (Settings > Environments) and into the main Terraform's tfvars/backend files.

output "github_variables" {
  description = "Variables for the GitHub workflows. The two client ids differ by environment (plan-prod reads; build and production apply)."
  value = {
    AZURE_TENANT_ID                      = data.azurerm_client_config.current.tenant_id
    AZURE_SUBSCRIPTION_ID                = data.azurerm_client_config.current.subscription_id
    ACR_NAME                             = azurerm_container_registry.shared.name
    "plan-prod: AZURE_CLIENT_ID"         = azuread_application.github["plan"].client_id
    "build, production: AZURE_CLIENT_ID" = azuread_application.github["apply"].client_id
  }
}

output "backend_config" {
  description = "Contents for infra/terraform/envs/prod.backend.hcl"
  value = {
    resource_group_name  = azurerm_resource_group.state.name
    storage_account_name = azurerm_storage_account.state.name
    container_name       = azurerm_storage_container.state.name
  }
}

output "main_tfvars" {
  description = "Values for infra/terraform/envs/*.tfvars"
  value = {
    acr_name                = azurerm_container_registry.shared.name
    acr_resource_group_name = azurerm_resource_group.shared.name
  }
}
