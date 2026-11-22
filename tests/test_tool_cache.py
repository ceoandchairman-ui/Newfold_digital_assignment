import time
from typing import Any

from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.tool_cache import CACHEABLE_TOOLS, CachingRegistry, ToolResultCache


def with_cache(tool: str, fn, cache: ToolResultCache):
    """The real CachingRegistry over one function, called like the function itself."""
    registry = CachingRegistry(StaticRegistry({tool: fn}), cache)

    async def call(**args: Any) -> dict[str, Any]:
        return await registry.call(tool, args)

    return call


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiry: dict[str, int] = {}
        self.fail = fail

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    async def hget(self, key: str, field: str):
        self._check()
        return self.hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str):
        self._check()
        self.hashes.setdefault(key, {})[field] = value

    async def expire(self, key: str, seconds: int):
        self._check()
        self.expiry[key] = seconds

    async def delete(self, key: str):
        self._check()
        self.hashes.pop(key, None)


TTL = {"get_account_summary": 60, "get_bills": 30}


def _tool(results: list[dict[str, Any]]):
    calls: list[dict[str, Any]] = []

    async def fn(**args: Any) -> dict[str, Any]:
        calls.append(args)
        return results[min(len(calls) - 1, len(results) - 1)]

    return fn, calls


async def test_second_identical_call_is_served_from_cache():
    fn, calls = _tool([{"ok": True, "bills": [1]}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    assert await tool(customer_id="C1") == await tool(customer_id="C1")
    assert len(calls) == 1


async def test_customers_never_share_cached_results():
    fn, calls = _tool([{"ok": True, "who": "any"}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    await tool(customer_id="C1")
    await tool(customer_id="C2")
    assert [c["customer_id"] for c in calls] == ["C1", "C2"]


async def test_different_arguments_are_cached_separately():
    fn, calls = _tool([{"ok": True}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    await tool(customer_id="C1", status="x")
    await tool(customer_id="C1", status="y")
    assert len(calls) == 2


async def test_expired_entries_are_refetched(monkeypatch):
    fn, calls = _tool([{"ok": True}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    await tool(customer_id="C1")
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 31)  # past the 30 s bills TTL
    await tool(customer_id="C1")
    assert len(calls) == 2


async def test_invalidate_drops_a_customers_entries_only():
    redis = FakeRedis()
    cache = ToolResultCache(redis, TTL)
    fn, calls = _tool([{"ok": True}])
    tool = with_cache("get_bills", fn, cache)
    await tool(customer_id="C1")
    await tool(customer_id="C2")
    await cache.invalidate("C1")
    await tool(customer_id="C1")
    await tool(customer_id="C2")
    assert [c["customer_id"] for c in calls] == ["C1", "C2", "C1"]  # C2 stayed cached


async def test_failed_results_are_not_cached():
    fn, calls = _tool([{"ok": False, "error": "x"}, {"ok": True}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    assert (await tool(customer_id="C1"))["ok"] is False
    assert (await tool(customer_id="C1"))["ok"] is True
    assert len(calls) == 2


async def test_redis_outage_falls_through_to_the_real_tool():
    fn, calls = _tool([{"ok": True}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(fail=True), TTL))
    assert (await tool(customer_id="C1"))["ok"] is True
    assert len(calls) == 1


async def test_no_customer_id_means_no_caching():
    fn, calls = _tool([{"ok": True}])
    tool = with_cache("get_bills", fn, ToolResultCache(FakeRedis(), TTL))
    await tool()
    await tool()
    assert len(calls) == 2


def test_only_safe_read_tools_are_cacheable():
    assert set(CACHEABLE_TOOLS) == {"get_account_summary", "get_bills"}
    for never in ("check_recent_payments", "process_payment", "send_notification"):
        assert never not in CACHEABLE_TOOLS


async def test_uncacheable_tools_always_reach_the_real_registry():
    """check_recent_payments must never be served stale, even through the caching registry."""
    fn, calls = _tool([{"ok": True, "recent_payments": 0}])
    reg = CachingRegistry(
        StaticRegistry({"check_recent_payments": fn}), ToolResultCache(FakeRedis(), TTL)
    )
    await reg.call("check_recent_payments", {"customer_id": "C1", "bill_id": "B1"})
    await reg.call("check_recent_payments", {"customer_id": "C1", "bill_id": "B1"})
    assert len(calls) == 2


def test_the_caching_registry_reports_the_same_tools_as_the_registry_it_wraps():
    inner = StaticRegistry({"get_bills": _tool([{"ok": True}])[0]})
    reg = CachingRegistry(inner, ToolResultCache(FakeRedis(), TTL))
    assert reg.has("get_bills") and not reg.has("process_payment")


async def test_an_uncached_registry_never_serves_stale_data():
    """The evaluator's registry is the plain one: every call reaches the real tool."""
    fn, calls = _tool([{"ok": True, "bills": [1]}])
    plain = StaticRegistry({"get_bills": fn})
    await plain.call("get_bills", {"customer_id": "C1"})
    await plain.call("get_bills", {"customer_id": "C1"})
    assert len(calls) == 2
