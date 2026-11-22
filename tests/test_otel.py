"""OpenTelemetry behaviour (E12: B1-B4, B6), checked with an in-memory span exporter."""

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import structlog
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from support_assistant import telemetry
from support_assistant.agents.mcp_client import McpRegistry
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.mcp_catalog import COMMS, DATABASE, DEVICE
from support_assistant.observability import REQUEST_ID_HEADER, RequestContextMiddleware
from support_assistant.orchestrator.graph import build_graph

EXPORTER = InMemorySpanExporter()
PROVIDER = TracerProvider()
PROVIDER.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(PROVIDER)  # the module-level `tracer` objects resolve to this

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = f"00-{TRACE_ID}-00f067aa0ba902b7-01"


@pytest.fixture(autouse=True)
def _clear():
    EXPORTER.clear()
    yield


def _spans(prefix: str = "") -> list[Any]:
    return [s for s in EXPORTER.get_finished_spans() if s.name.startswith(prefix)]


# --- B1: W3C trace context in, trace id out ---------------------------------------------------------
def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/x")
    async def x() -> dict[str, str]:
        ctx = trace.get_current_span().get_span_context()
        return {"trace_id": format(ctx.trace_id, "032x")}

    FastAPIInstrumentor.instrument_app(app, tracer_provider=PROVIDER)
    return app


async def test_an_incoming_traceparent_continues_the_callers_trace():
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://t")
    r = await client.get("/x", headers={"traceparent": TRACEPARENT})
    assert r.json()["trace_id"] == TRACE_ID  # same trace as the caller
    assert r.headers[REQUEST_ID_HEADER] == TRACE_ID  # the id support staff look up


async def test_outgoing_calls_carry_traceparent_for_the_next_service():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["traceparent"] = request.headers.get("traceparent", "")
        return httpx.Response(200)

    with trace.get_tracer("t").start_as_current_span("turn") as span:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            HTTPXClientInstrumentor.instrument_client(c, tracer_provider=PROVIDER)
            await c.get("http://orchestrator/invoke")
        trace_id = format(span.get_span_context().trace_id, "032x")
    assert seen["traceparent"].split("-")[1] == trace_id


# --- B6: logs carry trace and span ids --------------------------------------------------------------------
def test_log_lines_inside_a_span_carry_its_ids():
    with trace.get_tracer("t").start_as_current_span("work") as span:
        event = telemetry._add_trace_ids(None, "info", {"event": "x"})
        ctx = span.get_span_context()
    assert event["trace_id"] == format(ctx.trace_id, "032x")
    assert event["span_id"] == format(ctx.span_id, "016x")
    assert "trace_id" not in telemetry._add_trace_ids(None, "info", {"event": "outside"})


def test_without_an_otlp_endpoint_setup_is_a_safe_no_op(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    telemetry.setup_telemetry("svc")
    structlog.get_logger().info("still_logs")


def test_resource_attributes_follow_semantic_conventions(monkeypatch):
    monkeypatch.setenv("SERVICE_VERSION", "abc123")
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "prod")
    attrs = telemetry.resource_attributes("orchestrator")
    assert attrs == {
        "service.name": "orchestrator",
        "service.namespace": "support-assistant",
        "service.version": "abc123",
        "deployment.environment.name": "prod",
    }


# --- B2/B3: one client span per MCP tool call ---------------------------------------------------------------
class Conn:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def call(self, tool, args, *, retry=True):
        if self.fail:
            raise ExternalServiceUnavailable("down")
        return SimpleNamespace(structuredContent={"ok": True})

    async def aclose(self):
        pass


URLS = {DATABASE.name: "db", COMMS.name: "comms", DEVICE.name: "dev"}


async def test_mcp_calls_are_client_spans_with_rpc_attributes():
    reg = McpRegistry(URLS, connection_factory=lambda url: Conn())  # type: ignore[arg-type]
    await reg.call("get_bills", {"customer_id": "C1"})
    (span,) = _spans("mcp ")
    assert span.name == "mcp get_bills" and span.kind is trace.SpanKind.CLIENT
    assert dict(span.attributes) == {
        "rpc.system": "mcp",
        "rpc.service": "database",
        "rpc.method": "get_bills",
    }
    assert "C1" not in json.dumps(dict(span.attributes))  # no customer data on spans


async def test_a_failed_mcp_call_is_an_error_span():
    reg = McpRegistry(URLS, connection_factory=lambda url: Conn(fail=True))  # type: ignore[arg-type]
    with pytest.raises(ExternalServiceUnavailable):
        await reg.call("get_bills", {})
    (span,) = _spans("mcp ")
    assert span.status.status_code is StatusCode.ERROR


# --- B4: graph node spans, payments tagged for tail sampling ------------------------------------------------
class Client:
    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        if agent == "director":
            return AgentResponse(
                proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "5"}
            )
        return AgentResponse(verdict="approve")


async def _pay(*a):
    return {"ok": True}


def _state() -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S",
        "identity_verified": True,
        "route": "billing",
        "messages": [{"role": "user", "content": "pay"}],
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
        "audit": None,
    }


async def test_every_node_is_a_span_and_an_approval_pause_is_not_an_error():
    graph = build_graph(Client(), _pay, MemorySaver())
    cfg = {"configurable": {"thread_id": "otel"}}
    await graph.ainvoke(_state(), cfg)
    approve = [s for s in _spans("graph.node approve")]
    assert approve and approve[-1].attributes["support.awaiting_approval"] is True
    assert approve[-1].status.status_code is not StatusCode.ERROR
    assert {"graph.node auth_gate", "graph.node director", "graph.node evaluate"} <= {
        s.name for s in _spans("graph.node")
    }


async def test_the_payment_node_is_tagged_so_the_collector_keeps_every_payment_trace():
    graph = build_graph(Client(), _pay, MemorySaver())
    cfg = {"configurable": {"thread_id": "otel-pay"}}
    await graph.ainvoke(_state(), cfg)
    EXPORTER.clear()
    await graph.ainvoke(Command(resume={"approved": True}, update={"audit": None}), cfg)
    (pay,) = _spans("graph.node pay")
    assert pay.attributes["support.payment"] is True
