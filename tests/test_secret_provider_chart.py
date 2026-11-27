"""The SecretProviderClass every pod mounts, read the way the Key Vault driver reads it."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy" / "helm" / "support" / "templates" / "secrets.yaml"


def _spec() -> dict:
    text = TEMPLATE.read_text(encoding="utf-8")
    return yaml.safe_load(re.sub(r"\{\{.*?\}\}", '"x"', text))["spec"]


def _objects() -> list[dict]:
    array = yaml.safe_load(_spec()["parameters"]["objects"])["array"]
    # The driver parses `array` as a list of STRINGS, each holding one object's YAML. A plain map
    # fails the mount ("cannot unmarshal !!map into string") and no pod can start.
    assert array and all(isinstance(item, str) for item in array)
    return [yaml.safe_load(item) for item in array]


def test_every_object_is_a_string_naming_one_secret():
    for item in _objects():
        assert item["objectType"] == "secret" and item["objectName"]


def test_every_synced_key_comes_from_a_fetched_secret_that_terraform_creates():
    fetched = {item["objectName"] for item in _objects()}
    synced = {d["objectName"] for s in _spec()["secretObjects"] for d in s["data"]}
    assert synced == fetched
    terraform = "".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "infra" / "terraform").glob("*.tf")
    )
    for name in fetched:
        assert re.search(rf'(?<![\w-]){name}(?![\w-])|"?{name}"?\s*=', terraform), name
