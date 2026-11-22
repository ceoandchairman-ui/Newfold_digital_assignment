"""Tool-call result cache on its own Redis cluster (E6-5, D9).

Only read-only, customer-scoped tools are cacheable, with a short TTL, and a customer's entries are
dropped when a payment is attempted. Never cached:
  - check_recent_payments: it is a velocity check; a stale answer could approve a duplicate payment.
  - process_payment / send_notification: they change state.
  - check_suspension_lift / run_diagnostics: they depend on state that changes between turns.
The evaluator agent's registry is never wrapped in CachingRegistry, so a stale bill can never
decide a payment.

Layout: one hash per customer, `tool:{customer_id}`, field = tool + hash of its arguments. A customer
can only ever read their own hash, and invalidation is a single DEL.
Redis is an optimisation: any Redis error is swallowed and the real tool is called.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from support_assistant.agents.mcp_client import ToolRegistry
from support_assistant.observability import TOOL_CACHE

log = structlog.get_logger()

CACHEABLE_TOOLS = ("get_account_summary", "get_bills")


class ToolResultCache:
    def __init__(self, redis: Any, ttl_by_tool: dict[str, int], prefix: str = "tool:") -> None:
        self._redis, self._ttl, self._prefix = redis, ttl_by_tool, prefix

    def _key(self, customer_id: str) -> str:
        return f"{self._prefix}{customer_id}"

    @staticmethod
    def _field(tool: str, args: dict[str, Any]) -> str:
        digest = hashlib.sha1(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()  # noqa: S324
        return f"{tool}:{digest}"

    async def get(self, customer_id: str, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        try:
            raw = await self._redis.hget(self._key(customer_id), self._field(tool, args))
            if raw is None:
                return None
            entry = json.loads(raw)
            if time.time() - entry["at"] > self._ttl[tool]:
                return None  # the hash outlives individual entries, so age is checked here
            result: dict[str, Any] = entry["result"]
            return result
        except Exception:  # noqa: BLE001
            log.warning("tool_cache_read_failed", tool=tool)
            return None

    async def set(
        self, customer_id: str, tool: str, args: dict[str, Any], result: dict[str, Any]
    ) -> None:
        try:
            key = self._key(customer_id)
            await self._redis.hset(
                key, self._field(tool, args), json.dumps({"at": time.time(), "result": result})
            )
            await self._redis.expire(key, max(self._ttl.values()))
        except Exception:  # noqa: BLE001
            log.warning("tool_cache_write_failed", tool=tool)

    async def invalidate(self, customer_id: str) -> None:
        try:
            await self._redis.delete(self._key(customer_id))
        except Exception:  # noqa: BLE001
            log.warning("tool_cache_invalidate_failed")


class CachingRegistry:
    """A registry that serves CACHEABLE_TOOLS from the cache and passes every other tool through.

    Wrap the real registry in this only for agents allowed to read slightly stale data. The
    evaluator's registry is never wrapped. Failures from the real tool are never cached.
    """

    def __init__(self, inner: ToolRegistry, cache: ToolResultCache) -> None:
        self._inner, self._cache = inner, cache

    def has(self, tool: str) -> bool:
        return self._inner.has(tool)

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        customer_id = args.get("customer_id")
        if tool not in CACHEABLE_TOOLS or not customer_id:
            return await self._inner.call(tool, args)  # not cacheable, or no tenant: never cached
        hit = await self._cache.get(customer_id, tool, args)
        TOOL_CACHE.add(1, {"tool": tool, "result": "hit" if hit is not None else "miss"})
        if hit is not None:
            return hit
        result = await self._inner.call(tool, args)
        if result.get("ok"):
            await self._cache.set(customer_id, tool, args, result)
        return result
