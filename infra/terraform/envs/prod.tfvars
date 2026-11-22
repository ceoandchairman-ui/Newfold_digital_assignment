# Production (E10-14). Differences from dev are deliberate: HA, larger SKUs, uptime SLA, edge on.
env      = "prod"
location = "eastus2" # placeholder until D15 is decided; must match dev's region family for quota

aks_sku_tier     = "Standard" # uptime SLA
aks_node_vm_size = "Standard_D8s_v5"
aks_min_nodes    = 3
aks_max_nodes    = 10

pg_sku_name          = "GP_Standard_D4ds_v5"
pg_storage_mb        = 131072
pg_high_availability = true                                   # zone-redundant
pg_admin_object_id   = "00000000-0000-0000-0000-000000000000" # replace: Entra group object ID
pg_admin_name        = "support-db-admins"

redis_checkpoint_sku = "Balanced_B5" # NoEviction: size with headroom (alert at 75 %)
redis_cache_sku      = "Balanced_B1"
redis_toolcache_sku  = "Balanced_B1"

enable_edge      = true
edge_origin_host = "api.example.invalid" # replace: the backend ingress hostname

alert_emails = {} # replace: on-call recipients

openai_deployments = {
  "chat-director"       = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 150 }
  "chat-troubleshooter" = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 150 }
  "chat-small"          = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 150 }
  "chat-evaluator"      = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 50 }
}

# From infra/bootstrap outputs.
acr_name                = "acrsupportppiia"
acr_resource_group_name = "rg-support-shared"
