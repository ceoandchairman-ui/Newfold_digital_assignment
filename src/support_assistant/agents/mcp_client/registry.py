"""MCP client registry (E8-2): which tool lives on which MCP server, and how to call it.

The agent harness owns the rules (verified customer, allow-list, identity injection, audit) and
calls tools ONLY through a registry. The registry owns the routing and the connections:

    harness --call("get_bills")--> registry --route--> connection --> MCP server `database`

Three implementations share one small interface (`ToolRegistry`):
  McpRegistry      real: routes to MCP servers over long-lived connections.
  StaticRegistry   in-process functions: tests and the in-process agent client.
  CachingRegistry  wraps another registry with the tool-result cache (see tool_cache.py).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, Protocol

import structlog
from opentelemetry.trace import SpanKind, Status, StatusCode

from support_assistant.agents.mcp_client.connection import McpConnection
from support_assistant.errors import ExternalServiceUnavailable, NotFoundError
from support_assistant.logs import error_fields
from support_assistant.mcp_catalog import ALL_SERVERS, WRITE_TOOLS, ServerSpec
from support_assistant.observability import CIRCUIT_OPENED, tracer

log = structlog.get_logger()

ToolFn = Callable[..., Awaitable[dict[str, Any]]]


class ToolRegistry(Protocol):
    def has(self, tool: str) -> bool: ...

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]: ...


class CircuitBreaker:
    """Stops calling a server that keeps failing, so a dead server costs microseconds, not timeouts.

    closed: calls go through. After `threshold` consecutive failures -> open: calls fail at once
    for `cooldown_s`. Then one trial call is let through; success closes it, failure reopens it.
    """

    def __init__(
        self,
        threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold, self._cooldown, self._clock = threshold, cooldown_s, clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return self._clock() - self._opened_at < self._cooldown  # past cooldown: allow a trial

    def success(self) -> None:
        self._failures, self._opened_at = 0, None

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()


class McpRegistry:
    """Routes each tool to its MCP server using the shared catalog, over one connection per server.

    Per server: a circuit breaker. Per call: the connection's timeout, and an automatic retry only
    for tools that do not change state (WRITE_TOOLS are never retried by the client).
    """

    def __init__(
        self,
        urls: Mapping[str, str],
        specs: Iterable[ServerSpec] = ALL_SERVERS,
        connection_factory: Callable[[str], McpConnection] = McpConnection,
        breaker_factory: Callable[[], CircuitBreaker] = CircuitBreaker,
    ) -> None:
        self._specs = {s.name: s for s in specs}
        self._route = {tool: s.name for s in self._specs.values() for tool in s.tools}
        self._conns = {name: connection_factory(urls[name]) for name in self._specs}
        self._breakers = {name: breaker_factory() for name in self._specs}

    def has(self, tool: str) -> bool:
        return tool in self._route

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        server = self._route.get(tool)
        if server is None:
            raise NotFoundError(f"no MCP server serves {tool}")
        breaker = self._breakers[server]
        with tracer.start_as_current_span(
            f"mcp {tool}",
            kind=SpanKind.CLIENT,
            attributes={"rpc.system": "mcp", "rpc.service": server, "rpc.method": tool},
        ) as span:
            if breaker.is_open:
                log.warning("circuit_rejected_call", dependency=server, tool=tool)
                span.set_status(Status(StatusCode.ERROR, "circuit open"))
                raise ExternalServiceUnavailable(
                    f"mcp server {server} is unavailable (circuit open)"
                )
            try:
                result = await self._conns[server].call(tool, args, retry=tool not in WRITE_TOOLS)
            except Exception as exc:
                was_open = breaker.is_open
                breaker.failure()
                if breaker.is_open and not was_open:
                    CIRCUIT_OPENED.add(1, {"rpc.service": server})
                    log.error("circuit_opened", dependency=server, **error_fields(exc))
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            breaker.success()
        content = getattr(result, "structuredContent", None)
        if content is not None:
            return dict(content)
        return {"ok": False, "error": "no_structured_content"}

    async def verify(self, tools: Iterable[str]) -> list[str]:
        """Readiness check: which of `tools` are NOT actually served by their server right now.

        Returns the missing tool names (empty means ready). A server that cannot be reached counts
        as all of its needed tools missing, so a pod does not report ready against a dead server.
        """
        wanted: dict[str, set[str]] = {}
        for tool in tools:
            wanted.setdefault(self._route.get(tool, ""), set()).add(tool)
        missing = sorted(wanted.pop("", set()))  # tools the catalog does not even know
        for server, names in wanted.items():
            try:
                listed = await self._conns[server].list_tools()
                served = {t.name for t in listed.tools}
            except Exception as exc:  # noqa: BLE001
                log.warning("mcp_readiness_failed", dependency=server, **error_fields(exc))
                served = set()
            missing.extend(sorted(names - served))
        return missing

    async def aclose(self) -> None:
        for conn in self._conns.values():
            await conn.aclose()


class StaticRegistry:
    """A registry over plain async functions. No network."""

    def __init__(self, tools: Mapping[str, ToolFn]) -> None:
        self._tools = dict(tools)

    def has(self, tool: str) -> bool:
        return tool in self._tools

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool not in self._tools:
            raise NotFoundError(f"no tool named {tool}")
        return await self._tools[tool](**args)
