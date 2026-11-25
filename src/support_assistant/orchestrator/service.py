"""Orchestrator service: hosts the LangGraph, its Redis checkpointer and the per-session run lock.

Graph state and checkpoints live in Redis (D9): key-value reads and writes keep per-step latency
low. Postgres is not on this path: the orchestrator only writes the audit trail to it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from support_assistant.agents.mcp_client import McpRegistry
from support_assistant.audit import BackgroundAudit, PgAuditSink
from support_assistant.config import get_settings
from support_assistant.db import make_pool
from support_assistant.errors import ExternalServiceUnavailable, SupportError
from support_assistant.locks import RunLock
from support_assistant.logs import conversation, error_fields, new_turn_id
from support_assistant.mcp_catalog import server_urls
from support_assistant.observability import (
    ESCALATIONS,
    TURN_LATENCY_MS,
    TURN_TIMEOUTS,
    TURNS,
    RequestContextMiddleware,
)
from support_assistant.orchestrator.client import HttpAgentClient
from support_assistant.orchestrator.graph import STATE_SCHEMA_VERSION, build_graph
from support_assistant.redis_factory import make_redis
from support_assistant.telemetry import instrument_fastapi, setup_telemetry
from support_assistant.tool_cache import ToolResultCache

log = structlog.get_logger()


class InvokeBody(BaseModel):
    customer_id: str
    session_id: str
    identity_verified: bool
    route_hint: str | None = None
    search_query: str | None = None
    messages: list[dict[str, Any]]
    # Correlates this customer message across services. Restricted: it is written to logs.
    turn_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ResumeBody(BaseModel):
    session_id: str
    approved: bool


class TurnResult(BaseModel):
    reply: str = ""
    pending_approval: dict[str, Any] | None = None
    escalated: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_telemetry("orchestrator")

    # Checkpoints live on their own cluster: no eviction, AOF, RediSearch + RedisJSON (Terraform).
    # Refusing to start beats silently checkpointing onto the LRU cache cluster.
    if settings.checkpoint_redis_url is None:
        raise RuntimeError("CHECKPOINT_REDIS_URL is required by the orchestrator")
    redis_url = settings.checkpoint_redis_url.get_secret_value()
    redis = make_redis(
        redis_url, settings
    )  # one client, owned here: shared by the saver and the lock
    saver = AsyncRedisSaver(
        redis_client=redis,
        ttl={"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True},
    )
    await saver.asetup()  # creates the RediSearch indexes; needs RediSearch + RedisJSON modules

    app.state.lock = RunLock(redis, ttl_ms=settings.run_lock_ttl_ms)
    app.state.turn_timeout = settings.turn_timeout_seconds

    mcp = McpRegistry(server_urls(settings))  # the orchestrator only pays; it reads no cached data
    tool_redis = (
        make_redis(settings.tool_cache_redis_url.get_secret_value(), settings)
        if settings.tool_cache_redis_url
        else None
    )
    tool_cache = (
        ToolResultCache(tool_redis, {"get_account_summary": 1, "get_bills": 1})
        if tool_redis is not None
        else None
    )

    async def pay(customer_id: str, bill_id: str, amount: str, key: str) -> dict[str, Any]:
        try:
            return await mcp.call(
                "process_payment",
                {
                    "customer_id": customer_id,
                    "bill_id": bill_id,
                    "amount": amount,
                    "idempotency_key": key,
                },
            )
        finally:
            # Any attempt, successful or not, may change bills: drop this customer's cached reads.
            if tool_cache is not None:
                await tool_cache.invalidate(customer_id)

    audit_pool = make_pool(settings)
    await audit_pool.open()
    app.state.audit = BackgroundAudit(PgAuditSink(audit_pool))

    client = HttpAgentClient(settings)
    app.state.graph = build_graph(
        client,
        pay,
        saver,
        settings.max_eval_retries,
        settings.max_troubleshoot_turns,
    )
    yield
    await app.state.audit.drain()  # flush pending audit writes before closing the pool
    await mcp.aclose()
    await client.aclose()
    await audit_pool.close()
    await redis.aclose()
    if tool_redis is not None:
        await tool_redis.aclose()


app = FastAPI(title="orchestrator", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
instrument_fastapi(app)  # W3C traceparent in, server spans out


NOTHING_PENDING_REPLY = "There is nothing waiting for your confirmation."

MAX_STATE_BYTES = 256_000  # checkpoint hygiene (E7-3): alert well before Redis memory suffers


def check_state_size(state: dict[str, Any], session_id: str) -> int:
    """Log loudly when a session's graph state grows past the limit. Returns the size in bytes."""
    size = len(json.dumps(state, default=str))
    if size > MAX_STATE_BYTES:
        log.warning("graph_state_oversized", session_id=session_id, bytes=size)
    return size


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _run(session_id: str, graph_input: Any, deadline_s: float | None = None) -> TurnResult:
    """One active run per session (E4-12): a Redis lock around the whole turn.

    `deadline_s` bounds the run. A turn that overruns is cancelled and the customer is told to try
    again, instead of waiting on a stuck agent until the caller's own timeout fires. It is only given
    to a new message: a resume may execute the payment, and cancelling that mid-call would only
    leave a PENDING attempt for reconcile_payment to clean up, which is better avoided.
    """
    cfg = {"configurable": {"thread_id": session_id}}
    started = time.perf_counter()
    async with app.state.lock.hold(session_id):
        # A repeated /approve (double click, client retry) must not resume a finished run. Checked
        # inside the lock, so two approvals cannot both see the pause.
        if isinstance(graph_input, Command):
            read_state = getattr(app.state.graph, "aget_state", None)
            if read_state is not None and not (await read_state(cfg)).next:
                log.info("resume_ignored", reason="nothing_pending")
                return TurnResult(reply=NOTHING_PENDING_REPLY)
        # Latency: "async" writes each checkpoint in the background while the next node runs,
        # instead of blocking before it. Safe here because the payment node uses a stable
        # idempotency key, so a replay after a crash cannot charge twice.
        try:
            async with asyncio.timeout(deadline_s):  # None means no deadline
                out = await app.state.graph.ainvoke(graph_input, cfg, durability="async")
        except TimeoutError:
            TURN_TIMEOUTS.add(1)
            log.error("turn_timeout", deadline_s=deadline_s)
            raise ExternalServiceUnavailable("turn deadline exceeded") from None
    # Persist this turn's tool calls in the background (E5-5): the customer never waits for it.
    app.state.audit.submit(
        session_id, str(out.get("customer_id", "")), list(out.get("audit") or [])
    )
    check_state_size(out, session_id)
    TURNS.add(1)
    TURN_LATENCY_MS.record((time.perf_counter() - started) * 1000)
    log.info(
        "graph_run_completed",
        duration_ms=round((time.perf_counter() - started) * 1000),
        intent=out.get("intent"),
        escalated=bool(out.get("escalated")),
        escalation_reason=out.get("escalation_reason") if out.get("escalated") else None,
        awaiting_approval=bool(out.get("__interrupt__")),
        tool_calls=len(out.get("audit") or []),
    )
    if out.get("escalated"):
        ESCALATIONS.add(1, {"reason": str(out.get("escalation_reason"))})
    interrupts = out.get("__interrupt__")
    if interrupts:
        return TurnResult(pending_approval=interrupts[0].value)
    return TurnResult(reply=out.get("reply", ""), escalated=bool(out.get("escalated")))


@app.post("/invoke", response_model=TurnResult)
async def invoke(body: InvokeBody) -> TurnResult:
    state = body.model_dump()
    state.update(
        proposed_action=None,
        eval_attempts=0,
        reply="",
        escalated=False,
        payment_verified=False,
        evidence_refs=[],
        audit=None,  # resets the per-turn audit list
        schema_version=STATE_SCHEMA_VERSION,
    )
    try:
        deadline = getattr(app.state, "turn_timeout", None)
        with conversation(body.session_id, body.customer_id, body.turn_id or new_turn_id()):
            return await _run(body.session_id, state, deadline_s=deadline)
    except SupportError as exc:
        log.warning("graph_run_refused", session_id=body.session_id, **error_fields(exc))
        raise HTTPException(exc.http_status, detail=exc.customer_message) from exc


@app.post("/resume", response_model=TurnResult)
async def resume(body: ResumeBody) -> TurnResult:
    try:
        # A resume is a new turn for auditing: drop the entries already written when it paused.
        command: Command[Any] = Command(resume={"approved": body.approved}, update={"audit": None})
        with conversation(body.session_id, turn_id=new_turn_id()):
            return await _run(body.session_id, command)
    except SupportError as exc:
        log.warning("graph_resume_refused", session_id=body.session_id, **error_fields(exc))
        raise HTTPException(exc.http_status, detail=exc.customer_message) from exc
