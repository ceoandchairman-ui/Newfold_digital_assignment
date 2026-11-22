variable "github_repository" {
  type        = string
  description = "GitHub repository that deploys, as owner/name. Federated credentials trust only this repo."
}

variable "location" {
  type        = string
  default     = "eastus2"
  description = "Region for the shared resources. Placeholder until D15 is decided."
}

variable "prefix" {
  type    = string
  default = "support"
}

variable "tags" {
  type    = map(string)
  default = {}
}
