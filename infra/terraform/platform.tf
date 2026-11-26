# Azure OpenAI, Key Vault, Event Hubs (outbox) and Communication Services.

# --- Azure OpenAI (Foundry) -----------------------------------------------------
resource "azurerm_cognitive_account" "openai" {
  name                          = "oai-${local.name}-${local.suffix}"
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  kind                          = "OpenAI"
  sku_name                      = "S0"
  custom_subdomain_name         = "oai-${local.name}-${local.suffix}"
  public_network_access_enabled = false
  local_auth_enabled            = false # Entra only
  tags                          = local.tags
}

resource "azurerm_cognitive_deployment" "model" {
  for_each             = var.openai_deployments
  name                 = each.key
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = each.value.model_name
    version = each.value.model_version
  }

  sku {
    name     = each.value.sku_name
    capacity = each.value.capacity_k
  }
}

resource "azurerm_private_endpoint" "openai" {
  name                = "pe-openai-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "openai"
    private_connection_resource_id = azurerm_cognitive_account.openai.id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "openai"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["openai"].id]
  }
}

# --- Key Vault ------------------------------------------------------------------
resource "azurerm_key_vault" "main" {
  name                          = "kv-${var.prefix}-${var.env}-${local.suffix}"
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = "standard"
  rbac_authorization_enabled    = true
  purge_protection_enabled      = var.env == "prod"
  soft_delete_retention_days    = 7
  public_network_access_enabled = false
  tags                          = local.tags
}

resource "azurerm_private_endpoint" "keyvault" {
  name                = "pe-kv-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "keyvault"
    private_connection_resource_id = azurerm_key_vault.main.id
    subresource_names              = ["vault"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "keyvault"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["keyvault"].id]
  }
}

# --- Event Hubs (Kafka protocol): outbox for notifications (E6/E9) ---------------
# Standard is the lowest tier with the Kafka endpoint that also supports private endpoints. It
# replaces Service Bus Premium, which was billed per messaging unit per hour even when idle.
resource "azurerm_eventhub_namespace" "main" {
  name                          = "evh-${local.name}-${local.suffix}"
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  sku                           = "Standard"
  capacity                      = 1 # throughput units
  minimum_tls_version           = "1.2"
  public_network_access_enabled = false
  local_authentication_enabled  = false # Entra only
  tags                          = local.tags
}

# Kafka topics. "notifications-dlq" is the dead-letter topic: the worker copies there what it cannot
# send. Standard keeps events for at most 7 days; nothing reads the dead-letter topic automatically.
resource "azurerm_eventhub" "notifications" {
  for_each          = { "notifications" = 7, "notifications-dlq" = 7 } # name -> retention in days
  name              = each.key
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = 2
  message_retention = each.value
}

resource "azurerm_private_endpoint" "eventhubs" {
  name                = "pe-evh-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "eventhubs"
    private_connection_resource_id = azurerm_eventhub_namespace.main.id
    subresource_names              = ["namespace"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "eventhubs"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["servicebus"].id] # same domain
  }
}

# The MCP servers (comms) may only publish to "notifications". The worker reads it and may only
# publish to the dead-letter topic.
resource "azurerm_role_assignment" "notifications_sender" {
  scope                = azurerm_eventhub.notifications["notifications"].id
  role_definition_name = "Azure Event Hubs Data Sender"
  principal_id         = azurerm_user_assigned_identity.workload["mcp-servers"].principal_id
}

resource "azurerm_role_assignment" "notifications_receiver" {
  scope                = azurerm_eventhub.notifications["notifications"].id
  role_definition_name = "Azure Event Hubs Data Receiver"
  principal_id         = azurerm_user_assigned_identity.workload["notification-worker"].principal_id
}

resource "azurerm_role_assignment" "notifications_dead_letter_sender" {
  scope                = azurerm_eventhub.notifications["notifications-dlq"].id
  role_definition_name = "Azure Event Hubs Data Sender"
  principal_id         = azurerm_user_assigned_identity.workload["notification-worker"].principal_id
}

# --- Communication Services (email + SMS, D13) ----------------------------------
resource "azurerm_communication_service" "main" {
  name                = "acs-${local.name}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  data_location       = "United States"
  tags                = local.tags
}

# --- Azure AI Content Safety: Prompt Shields (E10-9) ------------------------------
resource "azurerm_cognitive_account" "content_safety" {
  name                          = "cs-${local.name}-${local.suffix}"
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  kind                          = "ContentSafety"
  sku_name                      = "S0"
  custom_subdomain_name         = "cs-${local.name}-${local.suffix}"
  public_network_access_enabled = false
  local_auth_enabled            = false # Entra only
  tags                          = local.tags
}

resource "azurerm_private_endpoint" "content_safety" {
  name                = "pe-cs-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "content-safety"
    private_connection_resource_id = azurerm_cognitive_account.content_safety.id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "cognitive"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["cognitive"].id]
  }
}

# Only the fat backend screens customer messages.
resource "azurerm_role_assignment" "content_safety_user" {
  scope                = azurerm_cognitive_account.content_safety.id
  role_definition_name = "Cognitive Services User"
  principal_id         = azurerm_user_assigned_identity.workload["fat-backend"].principal_id
}
