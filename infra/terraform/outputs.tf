output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "aks_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "acr_login_server" {
  value = data.azurerm_container_registry.main.login_server
}

output "postgres_fqdn" {
  value = azurerm_postgresql_flexible_server.main.fqdn
}

# Host per purpose: checkpoint, cache, toolcache. Port is 10000 on every cluster.
output "redis_hosts" {
  value = { for k, r in azapi_resource.redis : k => r.output.properties.hostName }
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "servicebus_namespace" {
  value = azurerm_servicebus_namespace.main.name
}

# Client IDs are injected into each ServiceAccount's workload-identity annotation by Helm.
output "workload_identity_client_ids" {
  value = { for k, v in azurerm_user_assigned_identity.workload : k => v.client_id }
}

output "content_safety_endpoint" {
  value = azurerm_cognitive_account.content_safety.endpoint
}

# Object ids are the Redis usernames under Entra auth (REDIS_ENTRA_OBJECT_ID per service).
output "workload_principal_ids" {
  value = { for k, v in azurerm_user_assigned_identity.workload : k => v.principal_id }
}

output "grafana_name" {
  value = azurerm_dashboard_grafana.main.name
}

# --- Consumed by scripts/helm_values_from_terraform.py (the deploy pipeline) ------------------
output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "tenant_id" {
  value = data.azurerm_client_config.current.tenant_id
}

# The Key Vault CSI addon's identity: the single identity that reads secrets for every pod.
output "csi_client_id" {
  value = azurerm_kubernetes_cluster.main.key_vault_secrets_provider[0].secret_identity[0].client_id
}

# Identity names are `id-<identity_prefix>-<workload>`; Postgres role names are the same strings.
output "identity_prefix" {
  value = local.name
}

# --- Document landing zone: consumed by scripts/helm_values_from_terraform.py ------------------
output "docs_storage_account" {
  value = azurerm_storage_account.docs.name
}

output "docs_container" {
  value = azurerm_storage_container.knowledge_base.name
}

output "uploads_storage_account" {
  value = azurerm_storage_account.uploads.name
}

output "uploads_container" {
  value = azurerm_storage_container.customer_uploads.name
}
