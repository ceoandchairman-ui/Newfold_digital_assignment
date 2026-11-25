"""Agent service: one FastAPI app, selected per container by the AGENT env var (D17, E10-18).

AGENT=director|troubleshooter|evaluator. Each Deployment runs exactly one role with its own
identity, tool allow-list and model deployment. Stateless: all state arrives in the request.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException

from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.intent import (
    KeywordIntentClassifier,
    KeywordRouteClassifier,
    LlmIntentClassifier,
    LlmRouteClassifier,
)
from support_assistant.agents.llm import build_role_llm, build_support_llms
from support_assistant.agents.mcp_client import McpRegistry, ToolRegistry
from support_assistant.config import get_settings
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.errors import SupportError
from support_assistant.logs import conversation
from support_assistant.mcp_catalog import server_urls
from support_assistant.models import Role
from support_assistant.observability import RequestContextMiddleware
from support_assistant.redis_factory import make_redis
from support_assistant.registry import build_registry
from support_assistant.releases import ReleaseGate, tools_release
from support_assistant.retrieval_client import HttpRetrievalClient
from support_assistant.telemetry import instrument_fastapi, setup_telemetry
from support_assistant.tool_cache import CachingRegistry, ToolResultCache

log = structlog.get_logger()
AGENT = os.environ.get("AGENT", "director")
ALLOWED = roles.ALLOWED_TOOLS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_telemetry(f"agent-{AGENT}")
    app.state.settings = settings
    # This agent's MCP client registry. The harness calls tools only through it.
    mcp = McpRegistry(server_urls(settings))
    registry: ToolRegistry = mcp
    # The evaluator's registry is never cached: a stale bill must not decide a payment.
    tool_redis = None
    if AGENT != "evaluator" and settings.tool_cache_redis_url:
        tool_redis = make_redis(settings.tool_cache_redis_url.get_secret_value(), settings)
        registry = CachingRegistry(
            mcp,
            ToolResultCache(
                tool_redis,
                {
                    "get_account_summary": settings.tool_cache_account_ttl_s,
                    "get_bills": settings.tool_cache_bills_ttl_s,
                },
            ),
        )
    app.state.tool_redis = tool_redis
    app.state.mcp = mcp
    app.state.registry = registry
    # One model per job (models.py). The support_agent (the "director" service) hosts four modes on
    # two models; the troubleshooter has its own reasoning model; the evaluator has an optional
    # judge. Each is a deployment name from settings, so a model swap is a Terraform change.
    app.state.llms = build_support_llms(settings) if AGENT == "director" else {}
    app.state.llm = (
        build_role_llm(settings, Role.TROUBLESHOOTER)
        if AGENT == "troubleshooter"
        else app.state.llms.get("billing")
    )
    app.state.judge = (
        build_role_llm(settings, Role.JUDGE)
        if AGENT == "evaluator" and settings.azure_openai_endpoint and settings.online_judge_enabled
        else None
    )
    # Validated model classification when a model is configured; keywords otherwise (E4-3).
    modelled = AGENT == "director" and bool(settings.azure_openai_endpoint)
    app.state.classifier = (
        LlmIntentClassifier(app.state.llms["route"]) if modelled else KeywordIntentClassifier()
    )
    app.state.route_classifier = (
        LlmRouteClassifier(app.state.llms["route"]) if modelled else KeywordRouteClassifier()
    )
    # Search and reranking live in the retrieval service; no agent reads the search tables itself.
    retrieval = HttpRetrievalClient(settings.retrieval_url, settings.retrieval_timeout_seconds)
    app.state.retrieval = retrieval
    metadata = build_registry(settings)
    app.state.release_gate = ReleaseGate(metadata, tools_release())
    yield
    if metadata is not None:
        await metadata.aclose()
    await mcp.aclose()
    await retrieval.aclose()
    if app.state.tool_redis is not None:
        await app.state.tool_redis.aclose()


app = FastAPI(title=f"agent-{AGENT}", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
instrument_fastapi(app)  # W3C traceparent in, server spans out


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "agent": AGENT}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    """Ready only when every MCP tool this agent may call is really served by its server."""
    missing = await app.state.mcp.verify(ALLOWED[AGENT])
    if missing:
        raise HTTPException(503, detail=f"MCP tools unavailable: {', '.join(missing)}")
    gate: ReleaseGate | None = getattr(app.state, "release_gate", None)
    if gate is not None and not await gate.check():
        raise HTTPException(503, detail="tool release not verified")
    return {"status": "ready", "agent": AGENT}


@app.post("/run", response_model=AgentResponse)
async def run(req: AgentRequest) -> AgentResponse:
    turn = str(req.context.get("turn_id") or "") or None
    mode = str(req.context.get("mode", "-"))
    with conversation(req.session_id, req.customer_id, turn):
        started = time.perf_counter()
        try:
            res = await _dispatch(req)
        except HTTPException as exc:
            log.warning("agent_run_refused", agent=AGENT, mode=mode, status=exc.status_code)
            raise
        log.info(
            "agent_run_completed",
            agent=AGENT,
            mode=mode,
            duration_ms=round((time.perf_counter() - started) * 1000),
            escalated=res.escalate,
            verdict=res.verdict,
            tool_calls=len(res.audit),
        )
        return res


async def _dispatch(req: AgentRequest) -> AgentResponse:
    harness = Harness(AGENT, ALLOWED[AGENT], app.state.registry)  # fresh audit trail per request
    try:
        if AGENT == "director":
            mode = req.context.get("mode", "billing")  # no mode = the pre-split behaviour
            if mode == "route":
                return await roles.support_route(req, harness, app.state.route_classifier)
            if mode == "normal":
                return await roles.normal_resp_gen(req, harness, app.state.llms["normal"])
            if mode == "instruct":
                return await roles.instruct_gen(
                    req, harness, app.state.llms["instruct"], app.state.retrieval
                )
            return await roles.billing_gen(
                req,
                harness,
                app.state.llms["billing"],
                app.state.classifier,
            )
        if AGENT == "evaluator":
            return await roles.evaluator(
                req,
                harness,
                judge=app.state.judge,
                judge_timeout_s=app.state.settings.judge_timeout_seconds,
                retrieval=app.state.retrieval,
            )

        return await roles.troubleshooter(
            req,
            harness,
            app.state.llm,
            app.state.retrieval,
            tool_steps=app.state.settings.troubleshooter_tool_steps,
        )
    except SupportError as exc:
        raise HTTPException(exc.http_status, detail=exc.customer_message) from exc
