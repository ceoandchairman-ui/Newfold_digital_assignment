# Azure Managed Redis: one cluster per purpose (D9), each with the policy its data needs.
#
#   checkpoint  LangGraph checkpoints, graph state, run locks. A payment paused for approval lives
#               here, so: NoEviction, AOF persistence, and the RediSearch + RedisJSON modules the
#               Redis checkpointer indexes with. Entries expire only through the saver's TTL.
#   cache       Conversation window and verified-session flag. Rebuildable from Postgres, so LRU
#               eviction and no persistence.
#   toolcache   Tool-call results. Disposable by design, so LRU eviction and no persistence.
#
# Separate clusters mean a full checkpoint store can never evict or block the caches, and a cache
# flood can never evict a checkpoint.

locals {
  redis_clusters = {
    checkpoint = {
      sku         = var.redis_checkpoint_sku
      eviction    = "NoEviction"
      persistence = true
      modules     = ["RediSearch", "RedisJSON"]
    }
    cache = {
      sku         = var.redis_cache_sku
      eviction    = "AllKeysLRU"
      persistence = false
      modules     = []
    }
    toolcache = {
      sku         = var.redis_toolcache_sku
      eviction    = "AllKeysLRU"
      persistence = false
      modules     = []
    }
  }
}

resource "azapi_resource" "redis" {
  for_each = local.redis_clusters

  # properties.publicNetworkAccess needs api-version 2025-07-01 or higher (2024-10-01 rejects it).
  type      = "Microsoft.Cache/redisEnterprise@2025-07-01"
  name      = "redis-${each.key}-${local.name}-${local.suffix}"
  parent_id = azurerm_resource_group.main.id
  location  = azurerm_resource_group.main.location
  tags      = merge(local.tags, { purpose = each.key })

  # The embedded schema does not list publicNetworkAccess, but we require it off. Validation is
  # disabled for this resource only; if Azure rejects the property, `apply` fails loudly (E10-1).
  schema_validation_enabled = false

  body = {
    sku = { name = each.value.sku }
    properties = {
      minimumTlsVersion   = "1.2"
      publicNetworkAccess = "Disabled"
      # Azure rejects persistence without HA, so a persisted cluster always has it. For the
      # others it is a switch (redis_high_availability): a subscription without usable zones in
      # the region can make the create hang and then fail.
      highAvailability = each.value.persistence || var.redis_high_availability ? "Enabled" : "Disabled"
    }
  }

  timeouts {
    create = "90m"
    update = "90m"
  }

  response_export_values = ["properties.hostName"]
}

resource "azapi_resource" "redis_db" {
  for_each = local.redis_clusters

  type      = "Microsoft.Cache/redisEnterprise/databases@2024-10-01"
  name      = "default"
  parent_id = azapi_resource.redis[each.key].id

  body = {
    properties = merge(
      {
        clientProtocol   = "Encrypted"
        port             = 10000
        evictionPolicy   = each.value.eviction
        clusteringPolicy = "EnterpriseCluster" # RediSearch requires the Enterprise cluster policy
      },
      length(each.value.modules) > 0 ? { modules = [for m in each.value.modules : { name = m }] } : {},
      each.value.persistence ? { persistence = { aofEnabled = true, aofFrequency = "1s" } } : {},
    )
  }
}

resource "azurerm_private_endpoint" "redis" {
  for_each = local.redis_clusters

  name                = "pe-redis-${each.key}-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "redis-${each.key}"
    private_connection_resource_id = azapi_resource.redis[each.key].id
    subresource_names              = ["redisEnterprise"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "redis"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["redis"].id]
  }
}

# --- Entra access to Redis (E10-6): each workload reaches only the clusters it uses ---------------
# The pod authenticates as its workload identity (username = object id, password = Entra token;
# see redis_factory.py). NOTE: the access-policy-assignment API version is unverified, so embedded
# schema validation is off for it; `apply` fails loudly if Azure rejects it.
locals {
  redis_access = {
    checkpoint = ["orchestrator"]
    # window + verified flags; device simulation state; ids of notifications already sent
    cache     = ["fat-backend", "mcp-servers", "notification-worker"]
    toolcache = ["agent-director", "agent-troubleshooter", "orchestrator"]
  }
  redis_access_pairs = merge([
    for cluster, workloads in local.redis_access : {
      for w in workloads : "${cluster}-${w}" => { cluster = cluster, workload = w }
    }
  ]...)
}

resource "azapi_resource" "redis_access" {
  for_each = local.redis_access_pairs

  type                      = "Microsoft.Cache/redisEnterprise/databases/accessPolicyAssignments@2025-04-01"
  name                      = replace(each.key, "-", "")
  parent_id                 = azapi_resource.redis_db[each.value.cluster].id
  schema_validation_enabled = false

  body = {
    properties = {
      accessPolicyName = "default"
      user             = { objectId = azurerm_user_assigned_identity.workload[each.value.workload].principal_id }
    }
  }
}
