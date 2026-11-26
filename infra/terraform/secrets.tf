# Secrets and database identity wiring: everything the pods need at start-up, created by Terraform
# so nothing is typed in by hand.
#
# No passwords exist anywhere: Postgres and Redis use Entra tokens (db.py, redis_factory.py). The
# secrets below hold only addresses (host, database, port, TLS).

# --- Key Vault CSI driver: ONE identity (the AKS addon's) reads the vault for all pods ---------
resource "azurerm_role_assignment" "csi_kv_secrets_user" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_kubernetes_cluster.main.key_vault_secrets_provider[0].secret_identity[0].object_id
}

# The key that hashes session and customer ids in logs (logs.py). Every service must use the same
# one, or the same conversation could not be followed across them. Changing it breaks the link
# between older and newer log lines. Like any generated secret, it is also held in Terraform state.
resource "random_password" "log_hash_key" {
  length  = 48
  special = false
}

# --- Secrets, created through the ARM control plane ----------------------------------------------
# The vault has no public network access, so a CI runner cannot reach its data plane. The ARM
# secrets resource works without it. The deploying identity needs "Key Vault Contributor".
locals {
  kv_secrets = merge(
    {
      "database-url" = "postgresql://${azurerm_postgresql_flexible_server.main.fqdn}:5432/support?sslmode=require"
    },
    {
      "appinsights-connection-string" = azurerm_application_insights.main.connection_string
    },
    {
      "log-hash-key" = random_password.log_hash_key.result
    },
    {
      "redis-url"            = "rediss://${azapi_resource.redis["cache"].output.properties.hostName}:10000/0"
      "checkpoint-redis-url" = "rediss://${azapi_resource.redis["checkpoint"].output.properties.hostName}:10000/0"
      "tool-cache-redis-url" = "rediss://${azapi_resource.redis["toolcache"].output.properties.hostName}:10000/0"
    },
  )
}

resource "azapi_resource" "kv_secret" {
  for_each = local.kv_secrets

  type      = "Microsoft.KeyVault/vaults/secrets@2023-07-01"
  name      = each.key
  parent_id = azurerm_key_vault.main.id

  # sensitive_body keeps values (the App Insights key) out of plan output and state diffs.
  sensitive_body = {
    properties = {
      value = each.value
    }
  }
}

# --- Postgres: the bootstrap identity can create roles for the other workload identities ----------
resource "azurerm_postgresql_flexible_server_active_directory_administrator" "bootstrap" {
  server_name         = azurerm_postgresql_flexible_server.main.name
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = azurerm_user_assigned_identity.workload["db-bootstrap"].principal_id
  principal_name      = azurerm_user_assigned_identity.workload["db-bootstrap"].name
  principal_type      = "ServicePrincipal"

  # Entra administrators are created one at a time.
  depends_on = [azurerm_postgresql_flexible_server_active_directory_administrator.admin]
}
