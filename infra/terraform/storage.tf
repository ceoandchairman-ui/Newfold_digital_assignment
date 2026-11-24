# Document landing zone (E2-2): where source documents (the support manual, policy) are stored
# before the indexer turns them into knowledge_base_chunks.
#
#   operators / pipeline  ->  blob container `knowledge-base`  ->  indexer job  ->  Postgres (pgvector)
#
# Private only (no public endpoint, no account keys). Versioning and soft delete mean an overwritten
# or deleted manual can be recovered, which matters because the knowledge base is rebuilt from it.

resource "azurerm_storage_account" "docs" {
  name                            = "st${var.prefix}docs${var.env}${local.suffix}"
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = var.env == "prod" ? "GRS" : "LRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false # Entra only: there is no key to leak
  default_to_oauth_authentication = true
  # Fully private by default. To let the team upload from known networks (their office or VPN
  # egress IPs) without joining the VNet, list them in docs_allowed_ip_ranges: the account then
  # accepts traffic ONLY from those addresses and the private endpoint, and still needs Entra sign-in.
  public_network_access_enabled = length(var.docs_allowed_ip_ranges) > 0
  tags                          = local.tags

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
    ip_rules       = var.docs_allowed_ip_ranges
  }

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 30
    }
  }
}

resource "azurerm_storage_container" "knowledge_base" {
  name                  = "knowledge-base"
  storage_account_id    = azurerm_storage_account.docs.id
  container_access_type = "private"
}

resource "azurerm_private_endpoint" "docs_blob" {
  name                = "pe-docs-blob-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "docs-blob"
    private_connection_resource_id = azurerm_storage_account.docs.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "blob"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["blob"].id]
  }
}

# The indexer only ever READS documents.
resource "azurerm_role_assignment" "indexer_blob_reader" {
  scope                = azurerm_storage_container.knowledge_base.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.workload["indexer"].principal_id
}

resource "azurerm_role_assignment" "index_worker_blob_reader" {
  scope                = azurerm_storage_container.knowledge_base.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.workload["index-worker"].principal_id
}

# Document owners upload with this group. Set docs_writers_object_id to an Entra group; leave empty
# to manage access by hand.
resource "azurerm_role_assignment" "docs_writers" {
  count                = var.docs_writers_object_id == "" ? 0 : 1
  scope                = azurerm_storage_container.knowledge_base.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.docs_writers_object_id
}
