"""Retrieval service: hybrid search and reranking, the only answer-time reader of the search tables.

    POST /v1/search  {store, query, top_k}  -> ranked evidence, each with a reference
    POST /v1/fetch   {refs}                 -> the same evidence again, plus what is gone or changed

Phase 1 serves one store, `company_knowledge`, which is shared and holds no customer data. Callers
are limited by the network policy (values.yaml `allowedCallers`); the service does not authenticate
them itself. Before a customer store is added it must verify both the caller and the customer.

Query text can contain what a customer typed, so it is never logged or put on a span.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request

from support_assistant.config import get_settings
from support_assistant.contracts import (
    Evidence,
    EvidenceRef,
    FetchRequest,
    FetchResponse,
    SearchRequest,
    SearchResponse,
)
from support_assistant.db import make_pool
from support_assistant.embeddings import azure_embedder
from support_assistant.errors import SupportError
from support_assistant.logs import (
    SESSION_HEADER,
    TURN_HEADER,
    conversation,
    error_fields,
    header_value,
)
from support_assistant.observability import RequestContextMiddleware
from support_assistant.registry import build_registry
from support_assistant.releases import ReleaseGate, search_release
from support_assistant.reranker import HttpReranker
from support_assistant.security import BodyLimitMiddleware
from support_assistant.services.retrieval import Chunk, fetch_kb, search_kb
from support_assistant.telemetry import instrument_fastapi, setup_telemetry

log = structlog.get_logger()

# What a caller (and so a model) is given from one chunk. Search and fetch trim the same way, so the
# evaluator judges exactly the text the answering agent saw.
MAX_EVIDENCE_CHARS = 1500


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_telemetry("retrieval")
    pool = make_pool(settings)
    await pool.open()
    embedder = azure_embedder(settings)
    reranker = (
        HttpReranker(
            settings.reranker_url, settings.reranker_model, os.environ.get("RERANKER_API_KEY", "")
        )
        if settings.rerank_enabled and settings.reranker_url
        else None
    )
    app.state.pool, app.state.embedder, app.state.reranker = pool, embedder, reranker
    app.state.settings = settings
    registry = build_registry(settings)
    app.state.release_gate = ReleaseGate(registry, search_release(settings))
    yield
    if registry is not None:
        await registry.aclose()
    await embedder.aclose()
    if reranker is not None:
        await reranker.aclose()
    await pool.close()


app = FastAPI(title="retrieval", lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware, exempt=())
app.add_middleware(RequestContextMiddleware)
instrument_fastapi(app)  # W3C traceparent in, server spans out


def _evidence(chunk: Chunk) -> Evidence:
    return Evidence(
        ref=EvidenceRef(store="company_knowledge", id=chunk.id, content_hash=chunk.content_hash),
        section_id=chunk.section_id,
        heading_path=chunk.heading_path,
        content=chunk.content[:MAX_EVIDENCE_CHARS],
        score=chunk.score,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    """Ready only when the knowledge table can really be read. The embedder is not called here: a
    provider blip must not take every pod out of the Service."""
    try:
        async with app.state.pool.connection() as conn:
            await conn.execute("SELECT 1 FROM knowledge_base_chunks LIMIT 1")
    except Exception as exc:  # noqa: BLE001 - any database problem means not ready
        log.error("not_ready", dependency="postgres", **error_fields(exc))
        raise HTTPException(503, detail="knowledge store unavailable") from None
    # A pod whose embedding settings were never published would search with the wrong vectors.
    gate: ReleaseGate | None = getattr(app.state, "release_gate", None)
    if gate is not None and not await gate.check():
        raise HTTPException(503, detail="search settings not verified")
    return {"status": "ready"}


def _caller_context(request: Request) -> AbstractContextManager[None]:
    """The caller's conversation and turn, as sent in headers: this body names no session."""
    return conversation(
        session_ref=header_value(request.headers, SESSION_HEADER),
        turn_id=header_value(request.headers, TURN_HEADER),
    )


@app.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    s = app.state.settings
    with _caller_context(request):
        try:
            chunks = await search_kb(
                app.state.pool,
                app.state.embedder,
                body.query,
                top_k=body.top_k,
                candidates=s.rerank_candidates,
                reranker=app.state.reranker,
                min_similarity=s.retrieval_min_similarity,
                embedding=(s.embedding_model, s.embedding_version),
            )
        except SupportError as exc:
            log.error("search_failed", store=body.store, **error_fields(exc))
            raise HTTPException(exc.http_status, detail=exc.customer_message) from exc
    return SearchResponse(results=[_evidence(c) for c in chunks])


@app.post("/v1/fetch", response_model=FetchResponse)
async def fetch(body: FetchRequest, request: Request) -> FetchResponse:
    try:
        found = {c.id: c for c in await fetch_kb(app.state.pool, [r.id for r in body.refs])}
    except SupportError as exc:
        log.error("fetch_failed", **error_fields(exc))
        raise HTTPException(exc.http_status, detail=exc.customer_message) from exc
    out = FetchResponse()
    for ref in body.refs:
        chunk = found.get(ref.id)
        if chunk is None or chunk.content_hash != ref.content_hash:
            out.missing.append(ref)
        else:
            out.results.append(_evidence(chunk))
    with _caller_context(request):
        log.info("evidence_fetched", requested=len(body.refs), missing=len(out.missing))
    return out
