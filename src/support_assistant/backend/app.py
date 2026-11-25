"""Fat backend: identity, history hydration, routing hints, then hands off to the orchestrator."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, Literal

import httpx
import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from support_assistant import repositories as repo
from support_assistant.agents.intent import KeywordRouteClassifier
from support_assistant.backend.channels import router as twilio_router
from support_assistant.budget import BUDGET_EXCEEDED_MESSAGE, TokenBudget, estimate_tokens
from support_assistant.chains import build_rewrite_chain, rewrite_query
from support_assistant.config import get_settings
from support_assistant.content_safety import PromptShields, local_shield
from support_assistant.context import (
    compact_history,
    episode_context,
    history_context,
    summarize_session,
)
from support_assistant.db import make_pool
from support_assistant.embeddings import HttpEmbedder
from support_assistant.errors import (
    AuthenticationError,
    ConflictError,
    DuplicateOperation,
    ExternalServiceUnavailable,
    Overloaded,
    RateLimited,
    SupportError,
    UploadRejected,
)
from support_assistant.logs import (
    TURN_ID,
    conversation,
    error_fields,
    fingerprint,
    new_turn_id,
)
from support_assistant.memory import SessionMemory
from support_assistant.observability import (
    BUDGET_REFUSALS,
    LOAD_SHED,
    PROMPT_ATTACKS,
    RATE_LIMITED,
    RequestContextMiddleware,
    outgoing_headers,
)
from support_assistant.redis_factory import make_redis
from support_assistant.security import (
    BodyLimitMiddleware,
    Bulkhead,
    RateLimiter,
    SecurityHeadersMiddleware,
    client_ip,
    subject_key,
    valid_session_id,
)
from support_assistant.services.episodes import save_episode, search_episodes
from support_assistant.telemetry import instrument_fastapi, setup_telemetry
from support_assistant.uploads import UploadService

log = structlog.get_logger()
VERIFIED_TTL_S = 1800
ATTACK_REPLY = "I can only help with your account, bills and device. What can I help you with?"


class VerifyBody(BaseModel):
    """Phone + date of birth, or full name + date of birth (E3-1). Exactly one identifier."""

    session_id: str = Field(min_length=1)
    phone: str | None = None
    full_name: str | None = Field(None, min_length=2, max_length=200)
    date_of_birth: date

    @model_validator(mode="after")
    def _one_identifier(self) -> VerifyBody:
        if bool(self.phone) == bool(self.full_name):
            raise ValueError("give either phone or full_name")
        return self


class EndBody(BaseModel):
    session_id: str = Field(min_length=1)
    outcome: Literal["resolved", "escalated", "abandoned"]


class ChatBody(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)
    # Optional. A client that retries a message sends the same id, and the message then runs once.
    message_id: str | None = Field(None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ApproveBody(BaseModel):
    session_id: str
    approved: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    pool = make_pool(s)
    await pool.open()
    redis = make_redis(s.redis_url.get_secret_value(), s, decode_responses=True)
    app.state.s, app.state.pool, app.state.redis = s, pool, redis
    app.state.limiter = RateLimiter(redis)
    app.state.bulkhead = Bulkhead(s.max_inflight_turns)
    app.state.memory = SessionMemory(pool, redis, s.redis_window_size, s.redis_session_ttl_s)
    app.state.budget = TokenBudget(redis, s.daily_token_budget) if s.daily_token_budget else None
    app.state.uploads = None
    uploads_store = None
    if s.uploads_storage_account:
        from support_assistant.blobstore import AzureBlobStore

        uploads_store = AzureBlobStore(s.uploads_storage_account, s.uploads_container)
        app.state.uploads = UploadService(
            pool, uploads_store, max_bytes=s.upload_max_bytes, daily_limit=s.upload_daily_limit
        )
    app.state.shields = None
    if s.content_safety_endpoint:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()

        async def safety_token() -> str:
            tok = await credential.get_token("https://cognitiveservices.azure.com/.default")
            return tok.token

        app.state.shields = PromptShields(s.content_safety_endpoint, safety_token)
    app.state.http = httpx.AsyncClient(base_url=s.orchestrator_url, timeout=60.0)
    # Episodic memory needs the embedder API; without it, episodes are simply skipped.
    api_key = os.environ.get("EMBEDDER_API_KEY", "")

    async def token() -> str:
        return api_key

    app.state.embedder = (
        HttpEmbedder(s.embedder_url, s.embedding_model, dim=s.embedding_dim, token_provider=token)
        if s.embedder_url
        else None
    )
    setup_telemetry("fat-backend")
    # Query rewrite is optional: without an OpenAI endpoint the raw message is used.
    app.state.rewrite = build_rewrite_chain(s) if s.azure_openai_endpoint else None
    yield
    await app.state.http.aclose()
    if uploads_store is not None:
        await uploads_store.aclose()
    if app.state.shields is not None:
        await app.state.shields.aclose()
    if app.state.embedder is not None:
        await app.state.embedder.aclose()
    await redis.aclose()
    await pool.close()


app = FastAPI(title="fat-backend", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(BodyLimitMiddleware)  # outside the app: an oversized body is refused unread
app.add_middleware(SecurityHeadersMiddleware)  # outermost: even a 413 carries the headers
instrument_fastapi(app)  # W3C traceparent in, server spans out
app.include_router(twilio_router)  # the inbound Twilio chat channel (off unless enabled)


@app.exception_handler(SupportError)
async def support_error_handler(_: Request, exc: SupportError) -> JSONResponse:
    # Every refusal that ends a request passes here, so this one line covers all those `raise`s.
    emit = log.error if exc.http_status >= 500 else log.warning
    emit("request_refused", status=exc.http_status, **{"error.code": exc.code}, **error_fields(exc))
    headers = {}
    if (wait := getattr(exc, "retry_after", None)) is not None:
        headers["Retry-After"] = str(wait)  # 429 and 503: when to come back
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.code, "message": exc.customer_message},
        headers=headers,
    )


def _cfg(name: str, default: Any) -> Any:
    """A setting, or a default where lifespan is bypassed (tests that set a partial settings stub)."""
    return getattr(getattr(app.state, "s", None), name, default)


def _caller(request: Request) -> str:
    peer = request.client.host if request.client else None
    return client_ip(request.headers, peer, int(_cfg("trusted_proxy_hops", 0)))


@asynccontextmanager
async def _slot() -> AsyncIterator[None]:
    """A place in this pod's concurrency limit, or Overloaded (a fast 503 with Retry-After)."""
    bulkhead = getattr(app.state, "bulkhead", None)
    if bulkhead is None:
        yield
        return
    try:
        async with bulkhead.slot():
            yield
    except Overloaded:
        LOAD_SHED.add(1)
        log.warning("load_shed")
        raise


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _check_session_id(session_id: str) -> None:
    """A session id is a bearer token. A short or guessable one is refused with the same answer as
    an unverified session, so nothing is learned from the refusal."""
    if not valid_session_id(session_id, int(_cfg("min_session_id_length", 0))):
        raise AuthenticationError("invalid session id")


async def _verified_customer(session_id: str) -> str:
    _check_session_id(session_id)
    cid = await app.state.redis.get(f"verified:{session_id}")
    if not cid:
        raise AuthenticationError("session not verified")
    return str(cid)


@app.post("/verify")
async def verify(body: VerifyBody, request: Request) -> dict[str, bool]:
    return await verify_identity(body, _caller(request))


async def verify_identity(body: VerifyBody, ip: str) -> dict[str, bool]:
    """Identity check with a lockout, because a date of birth is a small secret and can be guessed.

    Failed attempts are counted three ways: against this session, against the person being guessed
    (the phone or name, whether or not they exist, so the lockout does not reveal who is a customer),
    and against the caller's address. Reaching any limit locks that key out for the window. If the
    counters themselves are unavailable the check fails CLOSED: better a customer retries than an
    attacker gets unlimited guesses.
    """
    _check_session_id(body.session_id)
    limiter: RateLimiter | None = getattr(app.state, "limiter", None)
    window = int(_cfg("verify_window_seconds", 900))
    subject = subject_key(body.phone or "", body.full_name or "")
    keys = [
        (f"vf:s:{body.session_id}", int(_cfg("verify_max_failures_session", 5))),
        (f"vf:p:{subject}", int(_cfg("verify_max_failures_subject", 10))),
        (f"vf:i:{ip}", int(_cfg("verify_max_failures_ip", 30))),
    ]
    if limiter is not None:
        try:
            for key, limit in keys:
                if (wait := await limiter.blocked(key, limit)) is not None:
                    RATE_LIMITED.add(1, {"kind": "verify"})
                    log.warning("identity_locked_out", session_id=body.session_id, wait_s=wait)
                    raise RateLimited(wait)
        except RateLimited:
            raise
        except Exception as exc:  # noqa: BLE001 - cannot count, so cannot safely let guesses through
            log.error("verify_limiter_unavailable", **error_fields(exc))
            raise ExternalServiceUnavailable("rate limiter unavailable") from exc
    async with app.state.pool.connection() as conn:
        if body.phone:
            row = await repo.get_customer_by_phone(conn, body.phone)
        else:
            matches = await repo.find_customers_by_name_dob(
                conn, body.full_name or "", body.date_of_birth
            )
            row = matches[0] if len(matches) == 1 else None  # ambiguous name: refuse, never guess
    # Same answer for unknown customer, wrong DOB and ambiguous name: no account enumeration.
    if row is None or row["date_of_birth"] != body.date_of_birth:
        if limiter is not None:
            for key, _ in keys:
                try:
                    await limiter.record(key, window)
                except Exception:  # noqa: BLE001 - the failure is still refused
                    log.warning("verify_failure_not_counted")
        log.warning("identity_verification_failed", session_id=body.session_id)
        raise AuthenticationError("verification failed")
    await app.state.redis.set(f"verified:{body.session_id}", row["id"], ex=VERIFIED_TTL_S)
    if limiter is not None:
        for key, _ in keys[:2]:  # a success clears the session and the person, never the address
            try:
                await limiter.reset(key)
            except Exception:  # noqa: BLE001
                log.warning("verify_counter_not_reset")
    log.info("identity_verified", session_id=body.session_id, customer_id=str(row["id"]))
    return {"verified": True}


async def _orchestrate(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = await app.state.http.post(path, json=payload, headers=outgoing_headers())
    except httpx.TransportError as exc:
        log.error("orchestrator_unreachable", **error_fields(exc))
        raise ExternalServiceUnavailable("orchestrator unreachable") from exc
    if resp.status_code >= 400:
        log.warning("orchestrator_refused", status=resp.status_code, path=path)
        raise HTTPException(resp.status_code, detail=resp.json().get("detail", "Request failed."))
    data: dict[str, Any] = resp.json()
    return data


@app.post("/chat")
async def chat(body: ChatBody, request: Request) -> dict[str, Any]:
    return await run_chat(body, _caller(request))


async def _admit(customer_id: str, ip: str) -> None:
    """Request-rate limits. A failing limiter lets the request through: unlike identity, a missed
    rate limit costs money (tokens), not safety, and the daily token budget still applies."""
    limiter: RateLimiter | None = getattr(app.state, "limiter", None)
    if limiter is None:
        return
    try:
        await limiter.hit(f"chat:c:{customer_id}", int(_cfg("chat_rate_per_minute", 30)), 60)
        await limiter.hit(f"chat:i:{ip}", int(_cfg("ip_rate_per_minute", 120)), 60)
    except RateLimited:
        RATE_LIMITED.add(1, {"kind": "chat"})
        raise
    except Exception:  # noqa: BLE001
        log.warning("rate_limiter_unavailable")


def _replay(stored: Any, request_hash: str) -> dict[str, Any] | None:
    """The saved answer for this message, None if this id is new. Raises if the first request is
    still running, or if the id was used for a different message."""
    if not stored:
        return None
    entry: dict[str, Any] = json.loads(stored)
    if entry.get("hash") != request_hash:
        raise ConflictError("message_id was already used for a different message")
    if "result" not in entry:
        raise DuplicateOperation("message in progress")
    result: dict[str, Any] = entry["result"]
    return result


async def run_chat(body: ChatBody, ip: str) -> dict[str, Any]:
    """One customer message: checks, then the turn, run at most once per `message_id`."""
    customer_id = await _verified_customer(body.session_id)
    turn_id = body.message_id or new_turn_id()
    with conversation(body.session_id, customer_id, turn_id):
        await _admit(customer_id, ip)
        return await _run_once(body, customer_id)


async def _run_once(body: ChatBody, customer_id: str) -> dict[str, Any]:
    redis = app.state.redis
    ttl = int(_cfg("idempotency_ttl_seconds", 300))
    idem = f"idem:{customer_id}:{body.message_id}" if body.message_id else None
    # What this message_id stands for. The same id with a different message is a client bug, and
    # answering it with the first message's reply would be wrong.
    request_hash = hashlib.sha256(f"{body.session_id}\n{body.message}".encode()).hexdigest()
    if idem is not None:
        try:
            replayed = _replay(await redis.get(idem), request_hash)
            if replayed is not None:  # the same message already ran: return its answer, run nothing
                log.info("chat_replayed")
                return replayed
            marker = json.dumps({"hash": request_hash})
            if not await redis.set(idem, marker, ex=ttl, nx=True):
                raise DuplicateOperation("message in progress")
        except (DuplicateOperation, ConflictError) as exc:
            log.warning("chat_duplicate_refused", **error_fields(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - no idempotency without Redis, the turn still works
            idem = None
            log.warning("idempotency_unavailable", **error_fields(exc))
    started = time.perf_counter()
    try:
        async with _slot():
            result = await _chat_turn(body, customer_id)
    except BaseException:
        if idem is not None:  # a failed turn may be retried with the same id
            try:
                await redis.delete(idem)
            except Exception:  # noqa: BLE001
                log.warning("idempotency_marker_not_cleared")
        raise
    if idem is not None:
        try:
            await redis.set(idem, json.dumps({"hash": request_hash, "result": result}), ex=ttl)
        except Exception:  # noqa: BLE001
            log.warning("idempotency_result_not_stored")
    log.info(
        "chat_turn_completed",
        duration_ms=round((time.perf_counter() - started) * 1000),
        escalated=bool(result.get("escalated")),
        awaiting_approval=bool(result.get("pending_approval")),
        **{"query.hash": fingerprint(body.message)},
    )
    return result


async def _chat_turn(body: ChatBody, customer_id: str) -> dict[str, Any]:
    # E4-9: a payment confirmation left open in an earlier session is offered again here.
    pending = await _pending_for(customer_id)
    if pending and pending["session_id"] != body.session_id:
        log.info("approval_reoffered")
        return {"reply": "", "pending_approval": pending["question"], "escalated": False}
    # Jailbreak / injection check on the customer's own words (E5-6, E10-9). A detected attack
    # is answered without ever reaching a model or a tool.
    shields = getattr(app.state, "shields", None)
    verdict = await shields.check(body.message) if shields else local_shield(body.message, [])
    if verdict.user_attack:
        log.warning("prompt_attack_blocked", source=verdict.source)
        PROMPT_ATTACKS.add(1, {"source": verdict.source})
        return {"reply": ATTACK_REPLY, "pending_approval": None, "escalated": False}
    budget = getattr(app.state, "budget", None)
    if budget is not None and not await budget.consume(customer_id, estimate_tokens(body.message)):
        BUDGET_REFUSALS.add(1)
        log.warning("token_budget_exceeded")
        return {"reply": BUDGET_EXCEEDED_MESSAGE, "pending_approval": None, "escalated": False}
    history = compact_history(await app.state.memory.load(customer_id, body.session_id))
    messages = [*history, {"role": "user", "content": body.message}]
    # E6-1: on a session's FIRST turn only (one embedding call per session, not per turn),
    # hydrate relevant past episodes for this customer as context.
    embedder = getattr(app.state, "embedder", None)
    if not history and embedder is not None:
        try:
            past = await search_episodes(
                app.state.pool, embedder, customer_id=customer_id, query=body.message
            )
        except Exception as exc:  # noqa: BLE001 - memory is a bonus, never a blocker
            log.warning("episode_recall_failed", **error_fields(exc))
            past = []
        if context := episode_context(past):
            messages = [{"role": "system", "content": context}, *messages]
    # E6-1: the first turn of a session also gets a short note about the customer's other
    # conversations from the last few days, so they do not have to repeat themselves.
    settings = getattr(app.state, "s", None)  # unset only where lifespan is bypassed (tests)
    days = getattr(settings, "history_days", 0)
    max_sessions = getattr(settings, "history_max_sessions", 5)
    if not history and days:
        try:
            async with app.state.pool.connection() as conn:
                rows = await repo.recent_sessions(conn, customer_id, days, max_sessions + 1)
            prior = [r for r in rows if r["session_id"] != body.session_id]
            note = history_context(prior[:max_sessions])
        except Exception as exc:  # noqa: BLE001 - history is a bonus, never a blocker
            log.warning("recent_history_failed", **error_fields(exc))
            note = ""
        if note:
            messages = [{"role": "system", "content": note}, *messages]
    # A cheap keyword guess; the graph's support_agent makes the real routing decision. It only
    # tells us whether a retrieval is likely, so we pay for a rewrite just for those turns.
    hint = (await KeywordRouteClassifier().classify(body.message)).route
    query = (
        await rewrite_query(app.state.rewrite, history, body.message)
        if hint in ("TROUBLESHOOT", "INSTRUCT")
        else None
    )
    log.info(
        "chat_turn_prepared",
        route_hint=hint,
        rewritten=query is not None,
        history_messages=len(history),
        memory_notes=sum(1 for m in messages if m.get("role") == "system"),
    )
    result = await _orchestrate(
        "/invoke",
        {
            "customer_id": customer_id,
            "session_id": body.session_id,
            "identity_verified": True,
            "route_hint": hint,
            "search_query": query,
            "turn_id": structlog.contextvars.get_contextvars().get(TURN_ID),
            "messages": messages,
        },
    )
    if result.get("pending_approval"):
        await _remember_pending(customer_id, body.session_id, result["pending_approval"])
    new = [{"role": "user", "content": body.message}]
    if result.get("reply"):
        new.append({"role": "assistant", "content": result["reply"]})
    await app.state.memory.append(customer_id, body.session_id, new)
    return result


PENDING_TTL_S = 24 * 3600  # matches the checkpoint idle TTL: the paused run lives that long


async def _remember_pending(customer_id: str, session_id: str, question: Any) -> None:
    value = json.dumps({"session_id": session_id, "question": question})
    await app.state.redis.set(f"pending:{customer_id}", value, ex=PENDING_TTL_S)


async def _pending_for(customer_id: str) -> dict[str, Any] | None:
    try:
        raw = await app.state.redis.get(f"pending:{customer_id}")
    except Exception as exc:  # noqa: BLE001 - the cache is optional; the graph still guards the payment
        log.warning("pending_lookup_failed", **error_fields(exc))
        return None
    if not raw:
        return None
    data: dict[str, Any] = json.loads(raw)
    return data


@app.post("/approve")
async def approve(body: ApproveBody) -> dict[str, Any]:
    customer_id = await _verified_customer(body.session_id)
    # Resume the run that is actually paused, which may belong to an earlier session (E4-9).
    pending = await _pending_for(customer_id)
    thread = pending["session_id"] if pending else body.session_id
    with conversation(thread, customer_id, new_turn_id()):
        log.info("approval_submitted", approved=body.approved)
        result = await _orchestrate("/resume", {"session_id": thread, "approved": body.approved})
    try:
        await app.state.redis.delete(f"pending:{customer_id}")
    except Exception:  # noqa: BLE001 - a stale marker only re-asks; the paused run is gone
        log.warning("pending_marker_not_cleared")
    if result.get("reply"):
        await app.state.memory.append(
            customer_id, body.session_id, [{"role": "assistant", "content": result["reply"]}]
        )
    return result


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks and stop as soon as it is too big (never load a huge file)."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise UploadRejected(f"the file is larger than {max_bytes // 1_000_000} MB", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def _uploads() -> UploadService:
    svc: UploadService | None = getattr(app.state, "uploads", None)
    if svc is None:
        raise ExternalServiceUnavailable("uploads are not configured")
    return svc


@app.post("/upload")
async def upload(
    session_id: Annotated[str, Form(min_length=1)], file: Annotated[UploadFile, File()]
) -> dict[str, Any]:
    """A verified customer uploads a file (screenshot, PDF, text). It is quarantined until scanned."""
    customer_id = await _verified_customer(session_id)
    svc = _uploads()
    data = await _read_limited(file, app.state.s.upload_max_bytes)
    record = await svc.accept(customer_id, session_id, file.filename or "file", data)
    log.info("upload_accepted", session_id=session_id, customer_id=customer_id, upload_id=record.id)
    return record.public()


@app.get("/uploads")
async def list_uploads(session_id: str) -> dict[str, Any]:
    customer_id = await _verified_customer(session_id)
    return {"uploads": [r.public() for r in await _uploads().list_for(customer_id)]}


@app.get("/uploads/{upload_id}")
async def upload_status(upload_id: str, session_id: str) -> dict[str, Any]:
    customer_id = await _verified_customer(session_id)
    record = await _uploads().status(customer_id, upload_id)
    return record.public()


@app.post("/end")
async def end(body: EndBody) -> dict[str, bool]:
    """Close a session and store it as an episode for this customer (E2-5).

    The episode summary is deterministic (no model call): what the customer asked and how it ended.
    Idempotent: ending the same session twice stores one episode.
    """
    customer_id = await _verified_customer(body.session_id)
    embedder = getattr(app.state, "embedder", None)
    if embedder is None:
        return {"stored": False}
    messages = await app.state.memory.load(customer_id, body.session_id)
    summary = summarize_session(messages)
    if not summary:
        return {"stored": False}
    stored = await save_episode(
        app.state.pool,
        embedder,
        customer_id=customer_id,
        summary=summary,
        outcome=body.outcome,
        model=app.state.s.embedding_model,
    )
    await app.state.redis.delete(f"verified:{body.session_id}")  # the session is over
    log.info(
        "session_ended",
        session_id=body.session_id,
        customer_id=customer_id,
        outcome=body.outcome,
        episode_stored=stored,
    )
    return {"stored": stored}
