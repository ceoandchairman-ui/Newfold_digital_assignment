"""Agent service: one FastAPI app, selected per container by the AGENT env var (D17, E10-18).

AGENT=director|troubleshooter|evaluator. Each Deployment runs exactly one role with its own
identity, tool allow-list and model deployment. Stateless: all state arrives in the request.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.intent import (
    KeywordIntentClassifier,
    KeywordRouteClassifier,
    LlmIntentClassifier,
    LlmRouteClassifier,
)
from support_assistant.agents.llm import build_llm, build_support_llms
from support_assistant.agents.mcp_client import McpRegistry, ToolRegistry
from support_assistant.config import get_settings
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.db import make_pool
from support_assistant.embeddings import HttpEmbedder
from support_assistant.errors import SupportError
from support_assistant.mcp_catalog import server_urls
from support_assistant.observability import RequestContextMiddleware
from support_assistant.redis_factory import make_redis
from support_assistant.reranker import HttpReranker
from support_assistant.services.retrieval import Chunk, search_kb
from support_assistant.telemetry import instrument_fastapi, setup_telemetry
from support_assistant.tool_cache import CachingRegistry, ToolResultCache

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
    app.state.llm = build_llm(settings, f"chat-{AGENT}", use_reasoning=AGENT != "director")
    # Validated model classification when a model is configured; keywords otherwise (E4-3).
    app.state.classifier = (
        LlmIntentClassifier(app.state.llm)
        if settings.azure_openai_endpoint
        else KeywordIntentClassifier()
    )
    # The support_agent (the director service) hosts four modes, each with its own model.
    app.state.llms = build_support_llms(settings, f"chat-{AGENT}")
    app.state.route_classifier = (
        LlmRouteClassifier(app.state.llms["route"])
        if settings.azure_openai_endpoint
        else KeywordRouteClassifier()
    )
    app.state.pool = None
    if AGENT == "troubleshooter":
        pool = make_pool(settings)
        await pool.open()
        app.state.pool = pool
        base = settings.embedder_url
        app.state.reranker = (
            HttpReranker(
                settings.reranker_url,
                settings.reranker_model,
                os.environ.get("EMBEDDER_API_KEY", ""),
            )
            if settings.rerank_enabled and settings.reranker_url
            else None
        )
        key = os.environ.get("EMBEDDER_API_KEY", "")

        async def token() -> str:
            return key

        app.state.embedder = HttpEmbedder(
            base, settings.embedding_model, dim=settings.embedding_dim, token_provider=token
        )
    yield
    await mcp.aclose()
    if app.state.pool is not None:
        await app.state.pool.close()
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
    return {"status": "ready", "agent": AGENT}


@app.post("/run", response_model=AgentResponse)
async def run(req: AgentRequest) -> AgentResponse:
    harness = Harness(AGENT, ALLOWED[AGENT], app.state.registry)  # fresh audit trail per request
    try:
        if AGENT == "director":
            mode = req.context.get("mode", "billing")  # no mode = the pre-split behaviour
            if mode == "route":
                return await roles.support_route(req, harness, app.state.route_classifier)
            if mode == "normal":
                return await roles.normal_resp_gen(req, harness, app.state.llms["normal"])
            if mode == "instruct":
                return await roles.instruct_gen(req, harness, app.state.llms["instruct"])
            return await roles.billing_gen(
                req,
                harness,
                app.state.llms["billing"],
                app.state.classifier,
                tool_calling=app.state.settings.director_tool_calling,
            )
        if AGENT == "evaluator":
            return await roles.evaluator(req, harness)

        async def search(q: str) -> list[Chunk]:
            return await search_kb(
                app.state.pool,
                app.state.embedder,
                q,
                candidates=app.state.settings.rerank_candidates,
                reranker=app.state.reranker,
            )

        return await roles.troubleshooter(req, harness, app.state.llm, search)
    except SupportError as exc:
        raise HTTPException(exc.http_status, detail=exc.customer_message) from exc
