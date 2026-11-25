"""Hybrid retrieval: pgvector cosine + tsvector keyword, fused with reciprocal rank (E2-4, E2-7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from pgvector.psycopg import register_vector_async

from support_assistant.db import Pool
from support_assistant.embeddings import Embedder
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields, fingerprint
from support_assistant.reranker import Reranker

log = structlog.get_logger()

RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    id: int
    section_id: str
    heading_path: str
    content: str
    score: float
    content_hash: str = ""


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score = sum(1 / (k + rank))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


async def search_kb(
    pool: Pool,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
    candidates: int = 20,
    reranker: Reranker | None = None,
    min_similarity: float | None = None,
    embedding: tuple[str, str] | None = None,
) -> list[Chunk]:
    """Hybrid search. `min_similarity` (cosine, -1 to 1) drops weak VECTOR matches, so an unrelated
    question can return nothing. Keyword matches are kept: they already require the query's words.
    None means no floor.

    `embedding` is the (model, version) that embedded the query. Vector matches are then limited to
    chunks embedded the same way: distances between two models' vectors mean nothing. Keyword
    matches do not depend on the model, so they keep working while documents are re-embedded."""
    (vec,) = await embedder.embed([query])
    model, version = embedding or (None, None)
    async with pool.connection() as conn:
        await register_vector_async(conn)
        vec_rows = await (
            await conn.execute(
                """
                SELECT id, section_id, heading_path, content, content_hash,
                       embedding <=> %s::vector AS distance
                  FROM knowledge_base_chunks
                 WHERE (%s::text IS NULL OR embedding_model = %s)
                   AND (%s::text IS NULL OR embedding_version = %s)
                 ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (vec, model, model, version, version, vec, candidates),
            )
        ).fetchall()
        kw_rows = await (
            await conn.execute(
                """
                SELECT id, section_id, heading_path, content, content_hash
                  FROM knowledge_base_chunks
                 WHERE content_tsv @@ plainto_tsquery('simple', %s)
                 ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', %s)) DESC LIMIT %s
                """,
                (query, query, candidates),
            )
        ).fetchall()

    below_floor = 0
    if min_similarity is not None:
        kept = [r for r in vec_rows if 1.0 - float(r["distance"]) >= min_similarity]
        below_floor, vec_rows = len(vec_rows) - len(kept), kept

    by_id = {r["id"]: r for r in (*vec_rows, *kw_rows)}
    fused = rrf([[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]])
    ranked = _distinct_text(sorted(fused, key=fused.__getitem__, reverse=True), by_id)
    if reranker is not None and len(ranked) > 1:
        try:
            order = await reranker.rerank(query, [by_id[i]["content"] for i in ranked], top_k)
            ranked = [ranked[j] for j in order if 0 <= j < len(ranked)]
        except ExternalServiceUnavailable as exc:
            log.warning("rerank_failed", action="keep_fused_order", **error_fields(exc))
    ranked = ranked[:top_k]
    log.info(
        "knowledge_searched",
        vector_matches=len(vec_rows),
        keyword_matches=len(kw_rows),
        dropped_below_floor=below_floor,
        reranker=reranker is not None,
        chunk_ids=ranked,
        **{"query.hash": fingerprint(query)},
    )
    return [_chunk(by_id[i], fused[i]) for i in ranked]


def _distinct_text(ranked: list[int], by_id: dict[int, Any]) -> list[int]:
    """Two documents may now hold the same chunk: return that text once, at its best rank."""
    seen: set[str] = set()
    kept: list[int] = []
    for i in ranked:
        key = str(by_id[i].get("content_hash") or f"id-{i}")
        if key not in seen:
            seen.add(key)
            kept.append(i)
    return kept


def _chunk(row: dict[str, Any], score: float) -> Chunk:
    return Chunk(
        row["id"],
        row["section_id"],
        row["heading_path"],
        row["content"],
        score,
        str(row.get("content_hash") or ""),
    )


async def fetch_kb(pool: Pool, ids: list[int]) -> list[Chunk]:
    """The chunks with these ids that still exist, in no particular order."""
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT id, section_id, heading_path, content, content_hash
                  FROM knowledge_base_chunks WHERE id = ANY(%s)
                """,
                (ids,),
            )
        ).fetchall()
    return [_chunk(r, 0.0) for r in rows]
