# The central metadata registry (registry.py): Azure Cosmos DB for NoSQL.
#
# Metadata only. Chunk text and vectors stay in Postgres.
#
#   documents  partition /collection  one record per knowledge document (hash, settings, chunks)
#   catalog    partition /kind        write-once releases of shared settings (releases.py)
#
# Private endpoint only, Entra only: account keys are disabled, so every call is made by a workload
# identity holding one of the data-plane roles below. These are Cosmos DB's own role assignments
# (azurerm_cosmosdb_sql_role_*), not Azure RBAC, and Azure RBAC roles grant no data access here.

resource "azurerm_cosmosdb_account" "registry" {
  name                = "cosmos-${local.name}-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"
  tags                = local.tags

  public_network_access_enabled         = false
  local_authentication_enabled          = false # no account keys or connection strings
  access_key_metadata_writes_enabled    = false # data-plane identities cannot change the account
  minimal_tls_version                   = "Tls12"
  network_acl_bypass_for_azure_services = false

  # Serverless: this holds little data and is read at start-up and indexing time, not per turn.
  capabilities {
    name = "EnableServerless"
  }

  # One region, so Strong costs no cross-region latency: a record or release that was written is
  # what every other workload reads next. (Reads cost twice the request units of Session.)
  consistency_policy {
    consistency_level = "Strong"
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  backup {
    type = "Continuous"
    tier = "Continuous7Days" # point-in-time restore
  }
}

resource "azurerm_cosmosdb_sql_database" "registry" {
  name                = "registry"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.registry.name
}

locals {
  registry_containers = {
    documents = "/collection"
    catalog   = "/kind"
  }

  # Every data-plane role needs readMetadata for the SDK to work.
  registry_read = [
    "Microsoft.DocumentDB/databaseAccounts/readMetadata",
    "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read",
    "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery",
    "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/readChangeFeed",
  ]

  registry_roles = {
    reader = local.registry_read
    # Document records are rewritten when a document changes and deleted when it is removed.
    writer = concat(local.registry_read, [
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/replace",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/upsert",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/delete",
    ])
    # Create only, with NO replace, upsert or delete: a published release cannot be changed.
    publisher = concat(local.registry_read, [
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
    ])
  }

  # workload -> container -> role. Only workloads whose code uses the registry are listed.
  # The fat backend, orchestrator and notification worker do not, so they get nothing.
  registry_access = {
    "indexer"              = { documents = "writer", catalog = "reader" }
    "index-worker"         = { documents = "writer", catalog = "reader" }
    "retrieval"            = { catalog = "reader" }
    "mcp-servers"          = { catalog = "reader" }
    "agent-director"       = { catalog = "reader" }
    "agent-troubleshooter" = { catalog = "reader" }
    "agent-evaluator"      = { catalog = "reader" }
    "db-bootstrap"         = { catalog = "publisher" } # the deploy hook publishes releases
  }

  registry_assignments = merge([
    for workload, grants in local.registry_access : {
      for container, role in grants :
      "${workload}/${container}" => { workload = workload, container = container, role = role }
    }
  ]...)
}

resource "azurerm_cosmosdb_sql_container" "registry" {
  for_each              = local.registry_containers
  name                  = each.key
  resource_group_name   = azurerm_resource_group.main.name
  account_name          = azurerm_cosmosdb_account.registry.name
  database_name         = azurerm_cosmosdb_sql_database.registry.name
  partition_key_paths   = [each.value]
  partition_key_version = 2
}

resource "azurerm_cosmosdb_sql_role_definition" "registry" {
  for_each            = local.registry_roles
  name                = "support-registry-${each.key}-${local.name}"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.registry.name
  type                = "CustomRole"
  assignable_scopes   = [azurerm_cosmosdb_account.registry.id]

  permissions {
    data_actions = each.value
  }
}

# Scoped to ONE container, so a role on `catalog` gives no access to `documents` and vice versa.
resource "azurerm_cosmosdb_sql_role_assignment" "registry" {
  for_each            = local.registry_assignments
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.registry.name
  role_definition_id  = azurerm_cosmosdb_sql_role_definition.registry[each.value.role].id
  principal_id        = azurerm_user_assigned_identity.workload[each.value.workload].principal_id
  scope               = "${azurerm_cosmosdb_account.registry.id}/dbs/${azurerm_cosmosdb_sql_database.registry.name}/colls/${azurerm_cosmosdb_sql_container.registry[each.value.container].name}"
}

resource "azurerm_private_endpoint" "cosmos" {
  name                = "pe-cosmos-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "cosmos"
    private_connection_resource_id = azurerm_cosmosdb_account.registry.id
    subresource_names              = ["Sql"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "cosmos"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["cosmos"].id]
  }
}

output "cosmos_endpoint" {
  value = azurerm_cosmosdb_account.registry.endpoint
}
