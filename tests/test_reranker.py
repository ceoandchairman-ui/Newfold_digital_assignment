import httpx
import pytest

from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.reranker import HttpReranker, NoopReranker


async def test_noop_keeps_order_and_limits():
    assert await NoopReranker().rerank("q", ["a", "b", "c"], 2) == [0, 1]


async def test_http_reranker_orders_by_score():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "score": 0.1},
                    {"index": 2, "score": 0.9},
                    {"index": 1, "score": 0.5},
                ]
            },
        )

    r = HttpReranker("http://rr", "m", transport=httpx.MockTransport(handle))
    assert await r.rerank("q", ["a", "b", "c"], 2) == [2, 1]
    await r.aclose()


async def test_http_reranker_maps_failure():
    r = HttpReranker("http://rr", "m", transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    with pytest.raises(ExternalServiceUnavailable):
        await r.rerank("q", ["a"], 1)
    await r.aclose()
