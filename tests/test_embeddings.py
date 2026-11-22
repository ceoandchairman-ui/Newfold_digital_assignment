import json
import math

import httpx
import pytest

from support_assistant.embeddings import FakeEmbedder, HttpEmbedder
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
