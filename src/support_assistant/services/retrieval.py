"""Hybrid retrieval: pgvector cosine + tsvector keyword, fused with reciprocal rank (E2-4, E2-7)."""

from __future__ import annotations

from dataclasses import dataclass

from pgvector.psycopg import register_vector_async

from support_assistant.db import Pool
from support_assistant.embeddings import Embedder
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.reranker import Reranker

RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    id: int
    section_id: str
    heading_path: str
    content: str
    score: float


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
) -> list[Chunk]:
    (vec,) = await embedder.embed([query])
    async with pool.connection() as conn:
        await register_vector_async(conn)
        vec_rows = await (
            await conn.execute(
                """
                SELECT id, section_id, heading_path, content FROM knowledge_base_chunks
                 ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (vec, candidates),
            )
        ).fetchall()
        kw_rows = await (
            await conn.execute(
                """
                SELECT id, section_id, heading_path, content FROM knowledge_base_chunks
                 WHERE content_tsv @@ plainto_tsquery('simple', %s)
                 ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', %s)) DESC LIMIT %s
                """,
                (query, query, candidates),
            )
        ).fetchall()

    by_id = {r["id"]: r for r in (*vec_rows, *kw_rows)}
    fused = rrf([[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]])
    ranked = sorted(fused, key=fused.__getitem__, reverse=True)
    if reranker is not None and len(ranked) > 1:
        try:
            order = await reranker.rerank(query, [by_id[i]["content"] for i in ranked], top_k)
            ranked = [ranked[j] for j in order if 0 <= j < len(ranked)]
        except ExternalServiceUnavailable:
            pass  # keep the fused order
    ranked = ranked[:top_k]
    return [
        Chunk(i, by_id[i]["section_id"], by_id[i]["heading_path"], by_id[i]["content"], fused[i])
        for i in ranked
    ]
