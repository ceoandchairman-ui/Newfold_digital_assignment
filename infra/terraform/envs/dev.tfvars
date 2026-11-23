env      = "dev"
location = "eastus2" # placeholder until D15 is decided

pg_admin_object_id = "00000000-0000-0000-0000-000000000000" # replace with the Entra group/user object ID
pg_admin_name      = "support-db-admins"
pg_zone            = "" # this subscription has no zone 1 in eastus2; let Azure pick

# New subscription quota in eastus2: 10 vCPUs in total and none for the DSv5 family, so dev uses a
# DSv6 4-vCPU node and at most 2 nodes (8 vCPUs). Raise these after a quota increase.
aks_node_vm_size = "Standard_D4s_v6"
aks_min_nodes    = 1
aks_max_nodes    = 2

# D19: stronger model for customer-facing agents, small model for cheap steps.
# Names and versions are pinned here; confirm availability and quota in the chosen region (Risk 7, D15).
openai_deployments = {
  # gpt-4.1 Standard quota on this subscription is 50K TPM in total, so the two share it.
  "chat-director"       = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 25 }
  "chat-troubleshooter" = { model_name = "gpt-4.1", model_version = "2025-04-14", capacity_k = 25 }
  "chat-small"          = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 50 }
  "chat-evaluator"      = { model_name = "gpt-4.1-mini", model_version = "2025-04-14", capacity_k = 30 }
}

# From infra/bootstrap outputs.
acr_name                = "acrsupportppiia"
acr_resource_group_name = "rg-support-shared"
