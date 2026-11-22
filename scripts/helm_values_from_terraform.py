"""Turn `terraform output -json` into a Helm values file (JSON is valid YAML for Helm).

    terraform output -json > tf.json
    python scripts/helm_values_from_terraform.py tf.json prod > values.generated.json
    helm upgrade --install support ./chart -f values.generated.json

This is what fills in every per-service identity, the Key Vault details and the identity prefix, so
nothing is copied by hand between Terraform and Helm. It fails loudly on a missing output: a pod
with a blank identity cannot sign in to anything, so a silent default would be worse than a stop.
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Services that run as Deployments (they get clientId/objectId). db-bootstrap has its own block.
REQUIRED_OUTPUTS = (
    "workload_identity_client_ids",
    "workload_principal_ids",
    "key_vault_name",
    "tenant_id",
    "csi_client_id",
    "identity_prefix",
)


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
    for name in ("db-bootstrap",):
        if name not in client_ids or name not in object_ids:
            raise MissingOutput(f"no identity for {name!r} in the terraform outputs")

    services = {
        name: {"clientId": cid, "objectId": object_ids[name]}
        for name, cid in sorted(client_ids.items())
        if name != "db-bootstrap"
    }
    for name, svc in services.items():
        if not svc["clientId"] or not svc["objectId"]:
            raise MissingOutput(f"identity for {name!r} has an empty client or object id")

    return {
        "environment": environment,
        "identityPrefix": _value(outputs, "identity_prefix"),
        "keyVault": {
            "name": _value(outputs, "key_vault_name"),
            "tenantId": _value(outputs, "tenant_id"),
            "csiClientId": _value(outputs, "csi_client_id"),
        },
        "dbBootstrap": {
            "clientId": client_ids["db-bootstrap"],
            "objectId": object_ids["db-bootstrap"],
        },
        "services": services,
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
