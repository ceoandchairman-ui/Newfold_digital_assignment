output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "aks_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
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

# The deploy pipeline stores this as the `otel-azuremonitor` Kubernetes secret for the collector.
output "appinsights_connection_string" {
  value     = azurerm_application_insights.main.connection_string
  sensitive = true
}

output "grafana_name" {
  value = azurerm_dashboard_grafana.main.name
}
