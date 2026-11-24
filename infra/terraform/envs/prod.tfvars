# Production: the only environment (there is no dev). Sized to this subscription's current limits,
# NOT to the original production design. Each "raise" note below needs a quota increase or a
# region with usable availability zones first; none of them needs a code change.
env      = "prod"
location = "eastus2" # placeholder until D15 is decided

# The Postgres Entra administrator is NOT set here (the repository is public). The pipeline passes
# TF_VAR_pg_admin_object_id / _name / _type from the PG_ADMIN_* repository variables; locally, pass -var.
pg_zone = "" # this subscription has no zone 1 in eastus2; let Azure choose

# New-subscription quota in eastus2: 10 vCPUs per VM family and none for DSv5.
# Raise (after a quota increase): Standard_D8s_v6, min 3, max 10.
aks_sku_tier     = "Standard" # uptime SLA
aks_node_vm_size = "Standard_D4s_v6"
aks_min_nodes    = 2
aks_max_nodes    = 2

# Raise: GP_Standard_D4ds_v5 with pg_high_availability = true (zone-redundant needs usable zones).
pg_sku_name          = "GP_Standard_D2ds_v5"
pg_storage_mb        = 131072
pg_high_availability = false

# Zone-redundant Redis hung and then failed to create here. Raise to true where zones are usable.
redis_high_availability = false
redis_checkpoint_sku    = "Balanced_B1" # NoEviction: size with headroom (alert at 75 %); raise to B5 for real load
redis_cache_sku         = "Balanced_B0"
redis_toolcache_sku     = "Balanced_B0"

# Needs a real backend hostname before it can be turned on.
enable_edge      = false
edge_origin_host = "api.example.invalid" # replace: the backend ingress hostname

alert_emails = {} # replace: on-call recipients

# gpt-4.1 Standard quota on this subscription is 50K TPM in total, so the two share it.
# Raise (after a quota increase): 150 / 150 / 150 / 50.
openai_deployments = {
  "chat-director"       = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 25 }
  "chat-troubleshooter" = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 25 }
  "chat-small"          = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 50 }
  "chat-evaluator"      = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 30 }
  # Optional reasoning tier for disputes, refunds and cancellations. Add it only after a quota
  # request is approved for a reasoning model in the chosen region (D15), then set
  # AZURE_OPENAI_REASONING_DEPLOYMENT=chat-reasoning on the agents. Left out on purpose now.
  # "chat-reasoning"    = { model_name = "<reasoning model>", model_version = "<version>", capacity_k = 10 }
}

# From infra/bootstrap outputs.
acr_name                = "acrsupportppiia"
acr_resource_group_name = "rg-support-shared"
