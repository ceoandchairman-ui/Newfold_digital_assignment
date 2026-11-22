"""Trace correlation and business metrics (E12: B1, B7).

Correlation follows W3C Trace Context: the `traceparent` header is created, extracted and injected
by the OpenTelemetry instrumentations (telemetry.py). This module only exposes the trace id to
customers and support staff: every response carries `x-request-id: <trace id>`, so a ticket can be
looked up as one trace. An incoming `x-request-id` is no longer trusted for correlation.

Metrics are OTel instruments, no-op until a MeterProvider is configured. Attributes are low
cardinality on purpose: never customer ids, session ids or free text as metric labels.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from opentelemetry import metrics, trace

REQUEST_ID_HEADER = "x-request-id"
_request_id: ContextVar[str] = ContextVar("request_id", default="")

_meter = metrics.get_meter("support_assistant")
tracer = trace.get_tracer("support_assistant")

# --- RED and business metrics ----------------------------------------------------------------
TURNS = _meter.create_counter("support.turns", unit="{turn}", description="Conversation turns")
TURN_LATENCY_MS = _meter.create_histogram("support.turn.duration", unit="ms")
TOOL_CALLS = _meter.create_counter(
    "support.tool.calls", unit="{call}", description="MCP tool calls"
)
ESCALATIONS = _meter.create_counter("support.escalations", unit="{escalation}")
PAYMENTS = _meter.create_counter(
    "support.payments", unit="{payment}", description="Payment attempts by outcome"
)
TOOL_CACHE = _meter.create_counter("support.tool_cache.lookups", unit="{lookup}")
CIRCUIT_OPENED = _meter.create_counter("support.mcp.circuit_opened", unit="{event}")
BUDGET_REFUSALS = _meter.create_counter("support.budget.refusals", unit="{turn}")
PROMPT_ATTACKS = _meter.create_counter("support.prompt_attacks.blocked", unit="{turn}")


def current_request_id() -> str:
    return _request_id.get()


def outgoing_headers() -> dict[str, str]:
    """Kept for call sites; propagation is now done by OTel instrumentation (traceparent)."""
    return {}


class RequestContextMiddleware:
    """Exposes the trace id as `x-request-id` and on log lines. Runs inside the OTel server span."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ctx = trace.get_current_span().get_span_context()
        rid = format(ctx.trace_id, "032x") if ctx.is_valid else uuid.uuid4().hex
        token = _request_id.set(rid)
        structlog.contextvars.bind_contextvars(request_id=rid)

        async def send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((REQUEST_ID_HEADER.encode(), rid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
            _request_id.reset(token)
