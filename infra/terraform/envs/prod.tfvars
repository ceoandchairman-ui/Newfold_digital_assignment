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

# Every deployment is DataZoneStandard: eastus2 does not list "Standard" for the gpt-5.4 models,
# and DataZoneStandard keeps processing inside the US data zone (GlobalStandard would not).
# Availability and quota were checked on 2026-11-25.
#
# One deployment per job (src/support_assistant/models.py). The names must match the defaults there
# (llm_*_deployment settings); the model behind each name is chosen here, so swapping a model is a
# change to this block only.
#
# Models and versions are the starting shortlist from Microsoft's catalog (GA, pages dated
# 2026-11). The GPT-4.1 family is Deprecated, so it is not used. capacity_k (thousand tokens per
# minute) is within the quota but not yet sized from traffic. Choose the final models by running
# each candidate through the journey, red-team and retrieval evaluations.
openai_deployments = {
  # support_agent: the conversation (route, generic answers). Fast and cheap, low reasoning effort.
  "chat-support" = { model_name = "gpt-5.4-mini", model_version = "2026-03-17", capacity_k = 50, sku_name = "DataZoneStandard" }
  # support_agent: reasoning and task questions (plans, tool use, payment proposals).
  "chat-planner" = { model_name = "gpt-5.4", model_version = "2026-03-05", capacity_k = 25, sku_name = "DataZoneStandard" }
  # fat backend: query rewrite. A 64-token job, so the smallest model.
  "chat-rewrite" = { model_name = "gpt-5.4-nano", model_version = "2026-03-17", capacity_k = 20, sku_name = "DataZoneStandard" }
  # troubleshoot_agent: multi-step diagnosis.
  "chat-troubleshooter" = { model_name = "gpt-5.4", model_version = "2026-03-05", capacity_k = 25, sku_name = "DataZoneStandard" }
  # eval_agent: LLM-as-judge groundedness check. Prefer a different model family than the
  # generators once one is available, because judges favor their own family.
  "chat-judge" = { model_name = "gpt-5.4-mini", model_version = "2026-03-17", capacity_k = 20, sku_name = "DataZoneStandard" }
  # Embeddings: documents, search queries and episodic memory. The name must equal the
  # embedding_model setting. Changing the model means every document is embedded again.
  "text-embedding-3-large" = { model_name = "text-embedding-3-large", model_version = "1", capacity_k = 100, sku_name = "DataZoneStandard" }
}

# From infra/bootstrap outputs.
acr_name                = "acrsupportppiia"
acr_resource_group_name = "rg-support-shared"
