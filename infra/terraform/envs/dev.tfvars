env      = "dev"
location = "eastus2" # placeholder until D15 is decided

pg_admin_object_id = "00000000-0000-0000-0000-000000000000" # replace with the Entra group/user object ID
pg_admin_name      = "support-db-admins"

# D19: stronger model for customer-facing agents, small model for cheap steps.
# Names and versions are pinned here; confirm availability and quota in the chosen region (Risk 7, D15).
openai_deployments = {
  "chat-director"       = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 50 }
  "chat-troubleshooter" = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 50 }
  "chat-small"          = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 50 }
  "chat-evaluator"      = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 30 }
}

# From infra/bootstrap outputs.
acr_name                = "acrsupportppiia"
acr_resource_group_name = "rg-support-shared"
