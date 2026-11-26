# One registry shared by dev and prod, created by infra/bootstrap (the pipeline must be able to
# build and push an image BEFORE this configuration has ever been applied).
data "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = var.acr_resource_group_name
}

resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "aks-${local.name}"
  sku_tier            = var.aks_sku_tier

  private_cluster_enabled   = true
  oidc_issuer_enabled       = true
  workload_identity_enabled = true
  local_account_disabled    = true

  default_node_pool {
    name                 = "system"
    vm_size              = var.aks_node_vm_size
    vnet_subnet_id       = azurerm_subnet.aks.id
    auto_scaling_enabled = true
    min_count            = var.aks_min_nodes
    max_count            = var.aks_max_nodes
    os_disk_type         = "Managed"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    network_policy      = "azure" # NetworkPolicy support (E10-7)
    load_balancer_sku   = "standard"
    service_cidr        = "10.100.0.0/16"
    dns_service_ip      = "10.100.0.10"
  }

  azure_active_directory_role_based_access_control {
    azure_rbac_enabled = true
    tenant_id          = data.azurerm_client_config.current.tenant_id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  # Azure Monitor managed Prometheus scraping (E12-B5); data flows via the DCR in monitoring.tf.
  monitor_metrics {}

  oms_agent {
    log_analytics_workspace_id      = azurerm_log_analytics_workspace.main.id
    msi_auth_for_monitoring_enabled = true
  }

  tags = local.tags
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = data.azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
}

# --- One workload identity per service (D17, E10-6, E10-18) -------------------
locals {
  k8s_namespace = "support"
  workloads = toset([
    "fat-backend",
    "orchestrator",
    "agent-director",
    "agent-troubleshooter",
    "agent-evaluator",
    "retrieval", # search + reranker (reads knowledge_base_chunks)
    "mcp-servers",
    "notification-worker",
    "indexer",      # reads documents from blob storage and writes knowledge_base_chunks
    "index-worker", # the same, but always running: indexes uploads within about a minute
    "db-bootstrap", # runs migrations and creates Postgres roles for the others (pre-deploy hook)
  ])
  openai_users = toset([
    "fat-backend",  # query rewrite and embeddings
    "retrieval",    # query embeddings
    "indexer",      # document embeddings
    "index-worker", # document embeddings
    "orchestrator", # history_check: finds relevant past episodes
    "agent-director",
    "agent-troubleshooter",
    "agent-evaluator",
  ])
}

resource "azurerm_user_assigned_identity" "workload" {
  for_each            = local.workloads
  name                = "id-${local.name}-${each.key}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

# The Kubernetes ServiceAccount of each workload is named after the workload.
resource "azurerm_federated_identity_credential" "workload" {
  for_each            = local.workloads
  name                = "fic-${each.key}"
  resource_group_name = azurerm_resource_group.main.name
  parent_id           = azurerm_user_assigned_identity.workload[each.key].id
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.main.oidc_issuer_url
  subject             = "system:serviceaccount:${local.k8s_namespace}:${each.key}"
}

resource "azurerm_role_assignment" "openai_user" {
  for_each             = local.openai_users
  scope                = azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_user_assigned_identity.workload[each.key].principal_id
}
