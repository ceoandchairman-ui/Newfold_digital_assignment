# Alerts (E10-10, E7-2). The checkpoint Redis runs with NoEviction: when it fills up it REFUSES
# writes, so paused payments would fail. Alert well before that.
#
# NOTE: metric names below are from Azure Monitor's documented metrics but have not been checked
# against a deployed resource; `terraform validate` cannot catch a wrong metric name, `apply` will.

resource "azurerm_monitor_action_group" "oncall" {
  name                = "ag-${local.name}-oncall"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "oncall"
  tags                = local.tags

  dynamic "email_receiver" {
    for_each = var.alert_emails
    content {
      name                    = "email-${email_receiver.key}"
      email_address           = email_receiver.value
      use_common_alert_schema = true
    }
  }
}

resource "azurerm_monitor_metric_alert" "checkpoint_redis_memory" {
  name                = "alert-${local.name}-checkpoint-redis-memory"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azapi_resource.redis["checkpoint"].id]
  description         = "Checkpoint Redis is filling up. With NoEviction it will refuse writes when full."
  severity            = 1
  frequency           = "PT5M"
  window_size         = "PT15M"
  tags                = local.tags

  criteria {
    metric_namespace = "Microsoft.Cache/redisEnterprise"
    metric_name      = "usedmemorypercentage"
    aggregation      = "Maximum"
    operator         = "GreaterThan"
    threshold        = var.checkpoint_redis_memory_alert_percent
  }

  action {
    action_group_id = azurerm_monitor_action_group.oncall.id
  }
}

resource "azurerm_monitor_metric_alert" "postgres_cpu" {
  name                = "alert-${local.name}-postgres-cpu"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Postgres CPU is high: ledger, retrieval and audit writes will slow down."
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  tags                = local.tags

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "cpu_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.oncall.id
  }
}

resource "azurerm_monitor_metric_alert" "postgres_storage" {
  name                = "alert-${local.name}-postgres-storage"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Postgres storage is running out."
  severity            = 1
  frequency           = "PT15M"
  window_size         = "PT1H"
  tags                = local.tags

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "storage_percent"
    aggregation      = "Maximum"
    operator         = "GreaterThan"
    threshold        = 85
  }

  action {
    action_group_id = azurerm_monitor_action_group.oncall.id
  }
}
