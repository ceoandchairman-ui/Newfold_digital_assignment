# Observability backends (E12-B5, B8, B9). Telemetry arrives via the in-cluster OTel Collector:
#   traces + logs -> Application Insights (workspace-based, in the existing Log Analytics workspace)
#   metrics       -> Azure Monitor managed Prometheus (scraped from the collector) -> Managed Grafana
# SLO burn-rate alerts run as scheduled KQL queries over Application Insights.

resource "azurerm_application_insights" "main" {
  name                         = "appi-${local.name}"
  location                     = azurerm_resource_group.main.location
  resource_group_name          = azurerm_resource_group.main.name
  workspace_id                 = azurerm_log_analytics_workspace.main.id
  application_type             = "web"
  local_authentication_enabled = true # the collector's azuremonitor exporter uses the connection string
  retention_in_days            = var.env == "prod" ? 90 : 30
  tags                         = local.tags
}

# --- Managed Prometheus ---------------------------------------------------------------
resource "azurerm_monitor_workspace" "main" {
  name                = "amw-${local.name}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_monitor_data_collection_endpoint" "prometheus" {
  name                = "dce-${local.name}-prom"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  kind                = "Linux"
  tags                = local.tags
}

resource "azurerm_monitor_data_collection_rule" "prometheus" {
  name                        = "dcr-${local.name}-prom"
  resource_group_name         = azurerm_resource_group.main.name
  location                    = azurerm_resource_group.main.location
  data_collection_endpoint_id = azurerm_monitor_data_collection_endpoint.prometheus.id
  kind                        = "Linux"
  tags                        = local.tags

  destinations {
    monitor_account {
      monitor_account_id = azurerm_monitor_workspace.main.id
      name               = "MonitoringAccount1"
    }
  }

  data_flow {
    streams      = ["Microsoft-PrometheusMetrics"]
    destinations = ["MonitoringAccount1"]
  }

  data_sources {
    prometheus_forwarder {
      streams = ["Microsoft-PrometheusMetrics"]
      name    = "PrometheusDataSource"
    }
  }
}

resource "azurerm_monitor_data_collection_rule_association" "prometheus" {
  name                    = "dcra-${local.name}-prom"
  target_resource_id      = azurerm_kubernetes_cluster.main.id
  data_collection_rule_id = azurerm_monitor_data_collection_rule.prometheus.id
}

# --- Managed Grafana --------------------------------------------------------------------
resource "azurerm_dashboard_grafana" "main" {
  name                          = "amg-${var.prefix}-${var.env}-${local.suffix}" # max 23 chars
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  grafana_major_version         = 12
  public_network_access_enabled = true # Entra sign-in protects it; set false with a private endpoint in prod
  tags                          = local.tags

  identity {
    type = "SystemAssigned"
  }

  azure_monitor_workspace_integrations {
    resource_id = azurerm_monitor_workspace.main.id
  }
}

resource "azurerm_role_assignment" "grafana_monitoring_reader" {
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Monitoring Reader"
  principal_id         = azurerm_dashboard_grafana.main.identity[0].principal_id
}

resource "azurerm_role_assignment" "grafana_prometheus_reader" {
  scope                = azurerm_monitor_workspace.main.id
  role_definition_name = "Monitoring Data Reader"
  principal_id         = azurerm_dashboard_grafana.main.identity[0].principal_id
}

# --- SLOs as multi-window burn-rate alerts (E12-B8) ----------------------------------------
# Availability SLO for customer turns (/chat): var.slo_availability (default 99.5 %).
# Fast burn: 14.4x the budget over 1 h (pages). Slow burn: 6x over 6 h (tickets).
locals {
  error_budget = 1 - var.slo_availability / 100
  burn_alerts = {
    fast = { window = "PT1H", factor = 14.4, severity = 1 }
    slow = { window = "PT6H", factor = 6, severity = 2 }
  }
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "availability_burn" {
  for_each             = local.burn_alerts
  name                 = "slo-${local.name}-availability-${each.key}-burn"
  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  scopes               = [azurerm_application_insights.main.id]
  description          = "Customer-turn error budget burning ${each.value.factor}x too fast (${each.key})."
  severity             = each.value.severity
  evaluation_frequency = each.key == "fast" ? "PT5M" : "PT30M"
  window_duration      = each.value.window
  tags                 = local.tags

  criteria {
    query                   = <<-KQL
      requests
      | where name has "/chat"
      | summarize total = count(), failed = countif(success == false)
      | where total > 20
      | extend error_rate = todouble(failed) / total
      | where error_rate > ${local.error_budget * each.value.factor}
    KQL
    time_aggregation_method = "Count"
    operator                = "GreaterThan"
    threshold               = 0
  }

  action {
    action_groups = [azurerm_monitor_action_group.oncall.id]
  }
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "turn_latency" {
  name                 = "slo-${local.name}-turn-latency"
  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  scopes               = [azurerm_application_insights.main.id]
  description          = "p95 customer-turn latency above ${var.slo_turn_p95_ms} ms for 15 minutes."
  severity             = 2
  evaluation_frequency = "PT5M"
  window_duration      = "PT15M"
  tags                 = local.tags

  criteria {
    query                   = <<-KQL
      requests
      | where name has "/chat"
      | summarize p95 = percentile(duration, 95), n = count()
      | where n > 20 and p95 > ${var.slo_turn_p95_ms}
    KQL
    time_aggregation_method = "Count"
    operator                = "GreaterThan"
    threshold               = 0
  }

  action {
    action_groups = [azurerm_monitor_action_group.oncall.id]
  }
}

# A payment whose outcome is unknown must be reconciled; any such outcome alerts (E3-5).
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "payment_outcome_unknown" {
  name                 = "slo-${local.name}-payment-outcome-unknown"
  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  scopes               = [azurerm_application_insights.main.id]
  description          = "A payment ended with an unknown gateway outcome: reconcile it."
  severity             = 2
  evaluation_frequency = "PT5M"
  window_duration      = "PT15M"
  tags                 = local.tags

  criteria {
    query                   = <<-KQL
      customMetrics
      | where name == "support.payments"
      | where tostring(customDimensions["outcome"]) == "payment_outcome_unknown"
      | summarize n = sum(valueCount)
      | where n > 0
    KQL
    time_aggregation_method = "Count"
    operator                = "GreaterThan"
    threshold               = 0
  }

  action {
    action_groups = [azurerm_monitor_action_group.oncall.id]
  }
}
