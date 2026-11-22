"""The Terraform-to-Helm bridge fills in every identity and endpoint, and refuses to deploy with a
blank one."""

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

DEPLOYMENTS = [
    "fat-backend",
    "orchestrator",
    "agent-director",
    "agent-troubleshooter",
    "agent-evaluator",
    "mcp-servers",
    "notification-worker",
]
WORKLOADS = [*DEPLOYMENTS, "indexer", "db-bootstrap"]


def outputs(**overrides):
    out = {
        "workload_identity_client_ids": {"value": {w: f"cid-{w}" for w in WORKLOADS}},
        "workload_principal_ids": {"value": {w: f"oid-{w}" for w in WORKLOADS}},
        "key_vault_name": {"value": "kv-support-dev-abc12"},
        "tenant_id": {"value": "tenant-1"},
        "csi_client_id": {"value": "csi-1"},
        "identity_prefix": {"value": "support-dev"},
        "docs_storage_account": {"value": "stsupportdocsdev"},
        "docs_container": {"value": "knowledge-base"},
        "uploads_storage_account": {"value": "stsupportupldev"},
        "uploads_container": {"value": "customer-uploads"},
        "openai_endpoint": {"value": "https://oai-x.openai.azure.com/"},
        "content_safety_endpoint": {"value": "https://cs-x.cognitiveservices.azure.com/"},
        "servicebus_namespace": {"value": "sb-support-dev-abc12"},
    }
    out.update(overrides)
    return out


def test_every_deployment_gets_its_own_client_and_object_id():
    v = hv.build_values(outputs(), "dev")
    assert set(v["services"]) == set(DEPLOYMENTS)
    assert v["services"]["orchestrator"] == {
        "clientId": "cid-orchestrator",
        "objectId": "oid-orchestrator",
    }


def test_the_hook_and_job_identities_are_not_deployments():
    v = hv.build_values(outputs(), "dev")
    assert v["dbBootstrap"] == {"clientId": "cid-db-bootstrap", "objectId": "oid-db-bootstrap"}
    assert v["jobIdentities"] == {"indexer": {"clientId": "cid-indexer", "objectId": "oid-indexer"}}
    assert not {"db-bootstrap", "indexer"} & set(v["services"])  # never become Deployments


def test_key_vault_prefix_and_environment_are_passed_through():
    v = hv.build_values(outputs(), "prod")
    assert v["environment"] == "prod" and v["identityPrefix"] == "support-dev"
    assert v["keyVault"] == {
        "name": "kv-support-dev-abc12",
        "tenantId": "tenant-1",
        "csiClientId": "csi-1",
    }


def test_every_address_a_service_needs_is_generated_not_typed():
    e = hv.build_values(outputs(), "dev")["endpoints"]
    assert e == {
        "AZURE_OPENAI_ENDPOINT": "https://oai-x.openai.azure.com/",
        "CONTENT_SAFETY_ENDPOINT": "https://cs-x.cognitiveservices.azure.com/",
        "SERVICEBUS_NAMESPACE_FQDN": "sb-support-dev-abc12.servicebus.windows.net",
        "DOCS_STORAGE_ACCOUNT": "stsupportdocsdev",
        "DOCS_CONTAINER": "knowledge-base",
        "UPLOADS_STORAGE_ACCOUNT": "stsupportupldev",
        "UPLOADS_CONTAINER": "customer-uploads",
    }


def test_the_endpoint_names_match_the_settings_the_services_actually_read():
    from support_assistant.config import Settings

    fields = {n.upper() for n in Settings.model_fields}
    assert set(hv.build_values(outputs(), "dev")["endpoints"]) <= fields


@pytest.mark.parametrize(
    "missing",
    [
        "workload_identity_client_ids",
        "key_vault_name",
        "csi_client_id",
        "identity_prefix",
        "tenant_id",
        "docs_storage_account",
        "uploads_storage_account",
        "openai_endpoint",
        "servicebus_namespace",
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
    with pytest.raises(hv.MissingOutput):
        hv.build_values(outputs(openai_endpoint={"value": ""}), "dev")


def test_a_blank_identity_is_refused_rather_than_deployed():
    ids = {w: f"cid-{w}" for w in WORKLOADS}
    ids["mcp-servers"] = ""
    with pytest.raises(hv.MissingOutput, match="mcp-servers"):
        hv.build_values(outputs(workload_identity_client_ids={"value": ids}), "dev")


@pytest.mark.parametrize("absent", ["db-bootstrap", "indexer"])
def test_the_hook_and_indexer_identities_must_exist(absent):
    ids = {w: f"cid-{w}" for w in WORKLOADS if w != absent}
    with pytest.raises(hv.MissingOutput, match=absent):
        hv.build_values(outputs(workload_identity_client_ids={"value": ids}), "dev")


def test_the_cli_writes_json_helm_can_read_and_fails_on_bad_input(tmp_path, capsys):
    good = tmp_path / "tf.json"
    good.write_text(json.dumps(outputs()))
    assert hv.main([str(good), "dev"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["services"]["fat-backend"]["clientId"] == "cid-fat-backend"
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert hv.main([str(bad), "dev"]) == 1 and hv.main([]) == 2
