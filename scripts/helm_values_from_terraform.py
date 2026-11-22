"""Turn `terraform output -json` into a Helm values file (JSON is valid YAML for Helm).

    terraform output -json > tf.json
    python scripts/helm_values_from_terraform.py tf.json prod > values.generated.json
    helm upgrade --install support ./chart -f values.generated.json

This fills in every per-service identity, the Key Vault details, the identity prefix and every
address a service needs as a setting, so nothing is copied by hand between Terraform and Helm. It
fails loudly on a missing output: a pod with a blank identity cannot sign in to anything, and a
service with a missing address silently falls back to a stub, so a default would be worse than a stop.
"""

from __future__ import annotations

import json
import sys
from typing import Any

REQUIRED_OUTPUTS = (
    "workload_identity_client_ids",
    "workload_principal_ids",
    "key_vault_name",
    "tenant_id",
    "csi_client_id",
    "identity_prefix",
    "docs_storage_account",
    "docs_container",
    "uploads_storage_account",
    "uploads_container",
    "openai_endpoint",
    "content_safety_endpoint",
    "servicebus_namespace",
)

# Identities that are NOT Deployments: the pre-deploy hook and the CronJobs.
JOB_IDENTITIES = ("indexer",)
HOOK_IDENTITY = "db-bootstrap"


class MissingOutput(Exception):
    pass


def _value(outputs: dict[str, Any], key: str) -> Any:
    entry = outputs.get(key)
    if entry is None or entry.get("value") in (None, "", {}):
        raise MissingOutput(f"terraform output {key!r} is missing or empty")
    return entry["value"]


def build_values(outputs: dict[str, Any], environment: str) -> dict[str, Any]:
    missing = [k for k in REQUIRED_OUTPUTS if k not in outputs]
    if missing:
        raise MissingOutput(f"terraform outputs missing: {', '.join(missing)}")
    client_ids = _value(outputs, "workload_identity_client_ids")
    object_ids = _value(outputs, "workload_principal_ids")
    for name in (HOOK_IDENTITY, *JOB_IDENTITIES):
        if name not in client_ids or name not in object_ids:
            raise MissingOutput(f"no identity for {name!r} in the terraform outputs")

    def identity(name: str) -> dict[str, str]:
        found = {"clientId": client_ids[name], "objectId": object_ids[name]}
        if not found["clientId"] or not found["objectId"]:
            raise MissingOutput(f"identity for {name!r} has an empty client or object id")
        return found

    not_deployments = {HOOK_IDENTITY, *JOB_IDENTITIES}
    services = {name: identity(name) for name in sorted(client_ids) if name not in not_deployments}
    servicebus = _value(outputs, "servicebus_namespace")

    return {
        "environment": environment,
        "identityPrefix": _value(outputs, "identity_prefix"),
        "keyVault": {
            "name": _value(outputs, "key_vault_name"),
            "tenantId": _value(outputs, "tenant_id"),
            "csiClientId": _value(outputs, "csi_client_id"),
        },
        "dbBootstrap": identity(HOOK_IDENTITY),
        "jobIdentities": {name: identity(name) for name in JOB_IDENTITIES},
        "services": services,
        # Addresses delivered to every pod as settings. These names must match config.py.
        "endpoints": {
            "AZURE_OPENAI_ENDPOINT": _value(outputs, "openai_endpoint"),
            "CONTENT_SAFETY_ENDPOINT": _value(outputs, "content_safety_endpoint"),
            "SERVICEBUS_NAMESPACE_FQDN": f"{servicebus}.servicebus.windows.net",
            "DOCS_STORAGE_ACCOUNT": _value(outputs, "docs_storage_account"),
            "DOCS_CONTAINER": _value(outputs, "docs_container"),
            "UPLOADS_STORAGE_ACCOUNT": _value(outputs, "uploads_storage_account"),
            "UPLOADS_CONTAINER": _value(outputs, "uploads_container"),
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: helm_values_from_terraform.py <terraform-output.json> <environment>",
            file=sys.stderr,
        )
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        outputs = json.load(fh)
    try:
        values = build_values(outputs, argv[1])
    except MissingOutput as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json.dump(values, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
