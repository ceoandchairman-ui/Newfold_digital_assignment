import json
import math
from pathlib import Path

import httpx
import pytest

from support_assistant.config import Settings
from support_assistant.embeddings import FakeEmbedder, HttpEmbedder, azure_embedder
from support_assistant.errors import ExternalServiceUnavailable, ValidationError


async def test_fake_embedder_is_deterministic_unit_vector():
    emb = FakeEmbedder(dim=1024)
    a, b = await emb.embed(["hello", "hello"])
    assert a == b
    assert len(a) == 1024
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)


async def test_fake_embedder_differs_per_text():
    emb = FakeEmbedder(dim=64)
    a, b = await emb.embed(["alpha", "beta"])
    assert a != b


def _handler(dim: int, fail_first: int = 0):
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return httpx.Response(503)
        count = len(json.loads(request.read())["input"])
        data = [{"index": i, "embedding": [0.1] * dim} for i in range(count)]
        return httpx.Response(200, json={"data": data})

    return handle, calls


async def test_http_embedder_batches_and_checks_dimension():
    handle, calls = _handler(dim=8)
    emb = HttpEmbedder(
        "http://embedder", "m", dim=8, batch_size=2, transport=httpx.MockTransport(handle)
    )
    out = await emb.embed(["a", "b", "c"])
    assert len(out) == 3
    assert calls["n"] == 2  # 3 texts with batch size 2 -> 2 calls
    await emb.aclose()


async def test_http_embedder_rejects_wrong_dimension():
    handle, _ = _handler(dim=4)
    emb = HttpEmbedder("http://embedder", "m", dim=8, transport=httpx.MockTransport(handle))
    with pytest.raises(ValidationError):
        await emb.embed(["a"])
    await emb.aclose()


async def test_http_embedder_retries_then_succeeds():
    handle, calls = _handler(dim=8, fail_first=2)
    emb = HttpEmbedder("http://embedder", "m", dim=8, transport=httpx.MockTransport(handle))
    out = await emb.embed(["a"])
    assert len(out) == 1
    assert calls["n"] == 3
    await emb.aclose()


async def test_http_embedder_maps_persistent_failure():
    handle, _ = _handler(dim=8, fail_first=99)
    emb = HttpEmbedder("http://embedder", "m", dim=8, transport=httpx.MockTransport(handle))
    with pytest.raises(ExternalServiceUnavailable):
        await emb.embed(["a"])
    await emb.aclose()


REQUIRED = {"_env_file": None, "database_url": "postgresql://x", "redis_url": "redis://x"}


async def test_azure_embedder_calls_the_deployment_with_an_entra_token(monkeypatch):
    seen: dict[str, object] = {}

    class Credential:
        async def get_token(self, scope):
            seen["scope"] = scope
            return type("Token", (), {"token": "entra-token"})()

        async def close(self):
            seen["closed"] = True

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 1024}]})

    client = httpx.AsyncClient
    monkeypatch.setattr("azure.identity.aio.DefaultAzureCredential", Credential)
    monkeypatch.setattr(
        "support_assistant.embeddings.httpx.AsyncClient",
        lambda **kw: client(**{**kw, "transport": httpx.MockTransport(handle)}),
    )
    emb = azure_embedder(
        Settings(**REQUIRED, azure_openai_endpoint="https://acct.openai.azure.com/")
    )
    await emb.embed(["hello"])
    await emb.aclose()
    assert seen == {
        "scope": "https://cognitiveservices.azure.com/.default",
        "url": "https://acct.openai.azure.com/openai/v1/embeddings",
        "auth": "Bearer entra-token",
        "body": {"model": "text-embedding-3-large", "input": ["hello"], "dimensions": 1024},
        "closed": True,
    }


def test_azure_embedder_refuses_to_start_without_an_endpoint():
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        azure_embedder(Settings(**REQUIRED, azure_openai_endpoint=None))


def test_terraform_deploys_the_embedding_model_and_lets_every_embedding_service_call_it():
    root = Path(__file__).resolve().parents[1] / "infra" / "terraform"
    model = Settings.model_fields["embedding_model"].default
    assert f'"{model}" = {{' in (root / "envs" / "prod.tfvars").read_text(encoding="utf-8")
    block = (root / "aks.tf").read_text(encoding="utf-8").split("openai_users = toset([")[1]
    users = block.split("])")[0]
    for workload in ("fat-backend", "retrieval", "indexer", "index-worker", "orchestrator"):
        assert f'"{workload}"' in users
