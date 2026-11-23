# Postgres Flexible Server (relational + pgvector + LangGraph checkpoints) and Azure Managed Redis.

resource "azurerm_postgresql_flexible_server" "main" {
  name                          = "pg-${local.name}-${local.suffix}"
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  version                       = "16"
  sku_name                      = var.pg_sku_name
  storage_mb                    = var.pg_storage_mb
  delegated_subnet_id           = azurerm_subnet.postgres.id
  private_dns_zone_id           = azurerm_private_dns_zone.zones["postgres"].id
  public_network_access_enabled = false
  backup_retention_days         = var.env == "prod" ? 35 : 7
  geo_redundant_backup_enabled  = var.env == "prod"
  zone                          = var.pg_zone == "" ? null : var.pg_zone # "" lets Azure choose a zone this subscription can use

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = false
    tenant_id                     = data.azurerm_client_config.current.tenant_id
  }

  dynamic "high_availability" {
    for_each = var.pg_high_availability ? [1] : []
    content {
      mode                      = "ZoneRedundant"
      standby_availability_zone = "2"
    }
  }

  tags = local.tags

  depends_on = [azurerm_private_dns_zone_virtual_network_link.zones]

  lifecycle {
    ignore_changes = [zone, high_availability[0].standby_availability_zone]
  }
}

resource "azurerm_postgresql_flexible_server_active_directory_administrator" "admin" {
  server_name         = azurerm_postgresql_flexible_server.main.name
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = var.pg_admin_object_id
  principal_name      = var.pg_admin_name
  principal_type      = var.pg_admin_type

  lifecycle {
    # Azure stores the principal name cut to 64 characters, which would otherwise force a
    # replacement of the administrator on every plan.
    ignore_changes = [principal_name]
  }
}

# pgvector must be allow-listed before CREATE EXTENSION works (E1-1).
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "VECTOR"
}

resource "azurerm_postgresql_flexible_server_database" "support" {
  name      = "support"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azurerm_postgresql_flexible_server_database" "support_test" {
  count     = var.env == "prod" ? 0 : 1
  name      = "support_test"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

