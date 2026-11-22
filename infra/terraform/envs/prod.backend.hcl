# terraform init -backend-config=envs/prod.backend.hcl  (separate state from dev)
resource_group_name  = "rg-support-tfstate"
storage_account_name = "REPLACE_TFSTATE_ACCOUNT"
container_name       = "tfstate"
key                  = "support-prod.tfstate"
