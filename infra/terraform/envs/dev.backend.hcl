# terraform init -backend-config=envs/dev.backend.hcl
# The storage account is created once, outside this configuration (bootstrap), with versioning on.
resource_group_name  = "rg-support-tfstate"
storage_account_name = "REPLACE_TFSTATE_ACCOUNT"
container_name       = "tfstate"
key                  = "support-dev.tfstate"
