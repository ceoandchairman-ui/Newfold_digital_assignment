variable "env" {
  type        = string
  description = "Environment name (dev or prod)."
}

variable "location" {
  type        = string
  description = "Azure region. Open decision D15: drives HA, OpenAI quota and cost."
}

variable "prefix" {
  type        = string
  default     = "support"
  description = "Resource name prefix."
}

variable "tags" {
  type    = map(string)
  default = {}
}

# --- AKS ---------------------------------------------------------------------
variable "aks_sku_tier" {
  type        = string
  default     = "Free"
  description = "Free for dev, Standard for prod (uptime SLA)."
}

variable "aks_node_vm_size" {
  type    = string
  default = "Standard_D4s_v5"
}

variable "aks_min_nodes" {
  type    = number
  default = 1
}

variable "aks_max_nodes" {
  type    = number
  default = 3
}

variable "aks_api_authorized_ip_ranges" {
  type        = list(string)
  default     = []
  description = "Reserved for a non-private cluster; the cluster is private by default."
}

# --- Postgres ----------------------------------------------------------------
variable "pg_sku_name" {
  type    = string
  default = "B_Standard_B2s"
}

variable "pg_storage_mb" {
  type    = number
  default = 32768
}

variable "pg_high_availability" {
  type        = bool
  default     = false
  description = "Zone-redundant HA. Enable in prod only."
}

variable "pg_admin_object_id" {
  type        = string
  description = "Entra object ID of the Postgres administrator (user or group)."
}

variable "pg_admin_name" {
  type        = string
  description = "Display name of the Postgres Entra administrator."
}

# --- Redis (Azure Managed Redis) ---------------------------------------------
variable "redis_checkpoint_sku" {
  type        = string
  default     = "Balanced_B1"
  description = "Checkpoint cluster (NoEviction: a full cluster refuses writes, so size with headroom and alert on memory)."
}

variable "redis_cache_sku" {
  type        = string
  default     = "Balanced_B0"
  description = "Cache cluster (conversation window, verified flag). LRU eviction, so small is fine."
}

variable "redis_toolcache_sku" {
  type        = string
  default     = "Balanced_B0"
  description = "Tool-result cache cluster. LRU eviction, so small is fine."
}

# --- Azure OpenAI ------------------------------------------------------------
variable "openai_deployments" {
  description = "Model deployments keyed by purpose. Model names and versions must be pinned (Risk 7)."
  type = map(object({
    model_name    = string
    model_version = string
    capacity_k    = number # thousand tokens per minute
  }))
}

# --- Edge protection (E10-16) ----------------------------------------------------
variable "enable_edge" {
  type        = bool
  default     = false
  description = "Create Front Door + WAF. Needs edge_origin_host (the backend ingress hostname)."
}

variable "edge_origin_host" {
  type        = string
  default     = ""
  description = "Public hostname of the fat-backend ingress that Front Door forwards to."
}

variable "edge_verify_rate_per_minute" {
  type        = number
  default     = 10
  description = "Max /verify requests per client per minute (brute-force guard)."
}

variable "edge_chat_rate_per_minute" {
  type        = number
  default     = 60
  description = "Max /chat requests per client per minute (model-cost guard)."
}

# --- Alerts (E10-10) ---------------------------------------------------------------
variable "alert_emails" {
  type        = map(string)
  default     = {}
  description = "On-call recipients, keyed by a short name. Empty creates the action group with no receivers."
}

variable "checkpoint_redis_memory_alert_percent" {
  type        = number
  default     = 75
  description = "Alert when the checkpoint Redis (NoEviction) passes this memory use."
}

# --- SLOs (E12-B8) -----------------------------------------------------------------
variable "slo_availability" {
  type        = number
  default     = 99.5
  description = "Availability SLO for customer turns, in percent."
}

variable "slo_turn_p95_ms" {
  type        = number
  default     = 8000
  description = "p95 customer-turn latency SLO in milliseconds (model calls dominate)."
}

# --- Shared container registry (created by infra/bootstrap) ----------------------
variable "acr_name" {
  type        = string
  description = "Name of the shared ACR. Output `acr_name` of infra/bootstrap."
}

variable "acr_resource_group_name" {
  type        = string
  description = "Resource group of the shared ACR. Output `acr_resource_group` of infra/bootstrap."
}
