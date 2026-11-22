# Customer upload storage (E13): files a CUSTOMER sends during a conversation. A separate account
# from the team's knowledge-base documents (storage.tf) on purpose: this content is untrusted.
#
#   customer -> fat-backend (validates, stores) -> blob `customer-uploads` (quarantine)
#                                                        |  Defender for Storage scans on upload
#                                                        v  and tags the blob with the result
#   the backend releases a file only when the scan tag says "No threats found"
#
# Private only (no public endpoint, no account keys). Files expire automatically.

resource "azurerm_storage_account" "uploads" {
  name                            = "st${var.prefix}upl${var.env}${local.suffix}"
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS" # short-lived data: no geo copy needed
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  default_to_oauth_authentication = true
  public_network_access_enabled   = false
  tags                            = local.tags
}

resource "azurerm_storage_container" "customer_uploads" {
  name                  = "customer-uploads"
  storage_account_id    = azurerm_storage_account.uploads.id
  container_access_type = "private"
}

# Retention: customer files are deleted automatically. The database rows are removed by the
# retention job and by erasure requests (governance.py).
resource "azurerm_storage_management_policy" "uploads" {
  storage_account_id = azurerm_storage_account.uploads.id

  rule {
    name    = "expire-customer-uploads"
    enabled = true

    filters {
      blob_types = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = var.upload_retention_days
      }
    }
  }
}

# Malware scanning on upload. Results are written to the blob as an index tag
# ("Malware Scanning scan result"), which uploads.py checks before releasing any file.
resource "azurerm_security_center_storage_defender" "uploads" {
  storage_account_id                          = azurerm_storage_account.uploads.id
  malware_scanning_on_upload_enabled          = true
  malware_scanning_on_upload_cap_gb_per_month = var.upload_scan_cap_gb
}

resource "azurerm_private_endpoint" "uploads_blob" {
  name                = "pe-uploads-blob-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "uploads-blob"
    private_connection_resource_id = azurerm_storage_account.uploads.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "blob"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["blob"].id]
  }
}

# Only the fat backend touches customer uploads. It needs write (accept) and delete (reject,
# erase). NOTE: reading blob index tags may need a richer data role; confirm on the first run.
resource "azurerm_role_assignment" "backend_uploads" {
  scope                = azurerm_storage_container.customer_uploads.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.workload["fat-backend"].principal_id
}
