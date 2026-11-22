# Edge protection (E10-16): Azure Front Door Premium with a WAF in front of the fat backend.
# Off by default: it needs the public hostname of the backend's ingress (var.edge_origin_host).
# Rate limiting and bot protection stop abuse and runaway cost before it reaches AKS or the models.

resource "azurerm_cdn_frontdoor_profile" "edge" {
  count               = var.enable_edge ? 1 : 0
  name                = "afd-${local.name}"
  resource_group_name = azurerm_resource_group.main.name
  sku_name            = "Premium_AzureFrontDoor"
  tags                = local.tags
}

resource "azurerm_cdn_frontdoor_endpoint" "edge" {
  count                    = var.enable_edge ? 1 : 0
  name                     = "ep-${local.name}-${local.suffix}"
  cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.edge[0].id
  tags                     = local.tags
}

resource "azurerm_cdn_frontdoor_origin_group" "backend" {
  count                    = var.enable_edge ? 1 : 0
  name                     = "og-backend"
  cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.edge[0].id

  load_balancing {}

  health_probe {
    path                = "/healthz"
    protocol            = "Https"
    request_type        = "GET"
    interval_in_seconds = 30
  }
}

resource "azurerm_cdn_frontdoor_origin" "backend" {
  count                          = var.enable_edge ? 1 : 0
  name                           = "origin-backend"
  cdn_frontdoor_origin_group_id  = azurerm_cdn_frontdoor_origin_group.backend[0].id
  enabled                        = true
  host_name                      = var.edge_origin_host
  origin_host_header             = var.edge_origin_host
  certificate_name_check_enabled = true
}

resource "azurerm_cdn_frontdoor_route" "backend" {
  count                         = var.enable_edge ? 1 : 0
  name                          = "route-backend"
  cdn_frontdoor_endpoint_id     = azurerm_cdn_frontdoor_endpoint.edge[0].id
  cdn_frontdoor_origin_group_id = azurerm_cdn_frontdoor_origin_group.backend[0].id
  cdn_frontdoor_origin_ids      = [azurerm_cdn_frontdoor_origin.backend[0].id]
  supported_protocols           = ["Https"]
  patterns_to_match             = ["/*"]
  forwarding_protocol           = "HttpsOnly"
  https_redirect_enabled        = true
}

resource "azurerm_cdn_frontdoor_firewall_policy" "edge" {
  count               = var.enable_edge ? 1 : 0
  name                = "waf${var.prefix}${var.env}"
  resource_group_name = azurerm_resource_group.main.name
  sku_name            = azurerm_cdn_frontdoor_profile.edge[0].sku_name
  enabled             = true
  mode                = "Prevention"
  tags                = local.tags

  # Per-client rate limit on the verification endpoint: it is the brute-force target.
  custom_rule {
    name                           = "RateLimitVerify"
    enabled                        = true
    priority                       = 10
    type                           = "RateLimitRule"
    rate_limit_duration_in_minutes = 1
    rate_limit_threshold           = var.edge_verify_rate_per_minute
    action                         = "Block"

    match_condition {
      match_variable = "RequestUri"
      operator       = "Contains"
      match_values   = ["/verify"]
    }
  }

  # Per-client rate limit on chat turns: protects model spend (E6-5).
  custom_rule {
    name                           = "RateLimitChat"
    enabled                        = true
    priority                       = 20
    type                           = "RateLimitRule"
    rate_limit_duration_in_minutes = 1
    rate_limit_threshold           = var.edge_chat_rate_per_minute
    action                         = "Block"

    match_condition {
      match_variable = "RequestUri"
      operator       = "Contains"
      match_values   = ["/chat"]
    }
  }

  managed_rule {
    type    = "Microsoft_DefaultRuleSet"
    version = "2.1"
    action  = "Block"
  }

  managed_rule {
    type    = "Microsoft_BotManagerRuleSet"
    version = "1.1"
    action  = "Block"
  }
}

resource "azurerm_cdn_frontdoor_security_policy" "edge" {
  count                    = var.enable_edge ? 1 : 0
  name                     = "secpol-${local.name}"
  cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.edge[0].id

  security_policies {
    firewall {
      cdn_frontdoor_firewall_policy_id = azurerm_cdn_frontdoor_firewall_policy.edge[0].id

      association {
        patterns_to_match = ["/*"]

        domain {
          cdn_frontdoor_domain_id = azurerm_cdn_frontdoor_endpoint.edge[0].id
        }
      }
    }
  }
}
