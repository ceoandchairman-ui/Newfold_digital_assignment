"""The Terraform-to-Helm bridge fills in every identity, and refuses to deploy with a blank one."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "hv", ROOT / "scripts" / "helm_values_from_terraform.py"
)
assert _spec and _spec.loader
hv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hv)

WORKLOADS = [
    "fat-backend",
    "orchestrator",
    "agent-director",
    "agent-troubleshooter",
    "agent-evaluator",
    "mcp-servers",
    "notification-worker",
    "db-bootstrap",
]


def outputs(**overrides):
    out = {
        "workload_identity_client_ids": {"value": {w: f"cid-{w}" for w in WORKLOADS}},
        "workload_principal_ids": {"value": {w: f"oid-{w}" for w in WORKLOADS}},
        "key_vault_name": {"value": "kv-support-dev-abc12"},
        "tenant_id": {"value": "tenant-1"},
        "csi_client_id": {"value": "csi-1"},
        "identity_prefix": {"value": "support-dev"},
    }
    out.update(overrides)
    return out


def test_every_service_gets_its_own_client_and_object_id():
    v = hv.build_values(outputs(), "dev")
    assert set(v["services"]) == set(WORKLOADS) - {"db-bootstrap"}
    assert v["services"]["orchestrator"] == {
        "clientId": "cid-orchestrator",
        "objectId": "oid-orchestrator",
    }


def test_the_bootstrap_identity_is_separate_from_the_services():
    v = hv.build_values(outputs(), "dev")
    assert v["dbBootstrap"] == {"clientId": "cid-db-bootstrap", "objectId": "oid-db-bootstrap"}
    assert "db-bootstrap" not in v["services"]  # it must never become a Deployment


def test_key_vault_prefix_and_environment_are_passed_through():
    v = hv.build_values(outputs(), "prod")
    assert v["environment"] == "prod" and v["identityPrefix"] == "support-dev"
    assert v["keyVault"] == {
        "name": "kv-support-dev-abc12",
        "tenantId": "tenant-1",
        "csiClientId": "csi-1",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "workload_identity_client_ids",
        "key_vault_name",
        "csi_client_id",
        "identity_prefix",
        "tenant_id",
    ],
)
def test_a_missing_terraform_output_stops_the_deploy(missing):
    out = outputs()
    del out[missing]
    with pytest.raises(hv.MissingOutput):
        hv.build_values(out, "dev")


def test_an_empty_value_is_as_bad_as_a_missing_one():
    with pytest.raises(hv.MissingOutput):
        hv.build_values(outputs(key_vault_name={"value": ""}), "dev")


def test_a_blank_identity_is_refused_rather_than_deployed():
    ids = {w: f"cid-{w}" for w in WORKLOADS}
    ids["mcp-servers"] = ""
    with pytest.raises(hv.MissingOutput, match="mcp-servers"):
        hv.build_values(outputs(workload_identity_client_ids={"value": ids}), "dev")


def test_the_bootstrap_identity_must_exist():
    ids = {w: f"cid-{w}" for w in WORKLOADS if w != "db-bootstrap"}
    with pytest.raises(hv.MissingOutput, match="db-bootstrap"):
        hv.build_values(outputs(workload_identity_client_ids={"value": ids}), "dev")


def test_the_cli_writes_json_helm_can_read_and_fails_on_bad_input(tmp_path, capsys):
    good = tmp_path / "tf.json"
    good.write_text(json.dumps(outputs()))
    assert hv.main([str(good), "dev"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["services"]["fat-backend"]["clientId"]
        == "cid-fat-backend"
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert hv.main([str(bad), "dev"]) == 1 and hv.main([]) == 2
