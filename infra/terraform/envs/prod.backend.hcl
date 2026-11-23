# terraform init -backend-config=envs/prod.backend.hcl  (the only environment)
resource_group_name  = "rg-support-tfstate"
storage_account_name = "stsupporttfppiia"
container_name       = "tfstate"
key                  = "support-prod.tfstate"
