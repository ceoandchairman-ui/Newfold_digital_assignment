"""Episodic memory: per-customer summaries of past sessions, searched by similarity (E2-5).

Strict tenant isolation: every query is filtered by customer_id in SQL, so one customer's episodes
can never be returned for another, whatever the similarity.
"""

from __future__ import annotations

import hashlib

from pgvector.psycopg import register_vector_async

from support_assistant.db import Pool
from support_assistant.embeddings import Embedder
from support_assistant.errors import ValidationError

MAX_SUMMARY_CHARS = 1000


def episode_hash(customer_id: str, summary: str, outcome: str) -> str:
    return hashlib.sha256(f"{customer_id}\n{summary}\n{outcome}".encode()).hexdigest()


async def save_episode(
    pool: Pool,
    embedder: Embedder,
    *,
    customer_id: str,
    summary: str,
    outcome: str,
    model: str,
    version: str = "1",
) -> bool:
    """Store one episode. Returns False if the identical episode already exists (idempotent)."""
    summary = summary.strip()
    if not customer_id or not summary:
        raise ValidationError("an episode needs a customer and a summary")
    summary = summary[:MAX_SUMMARY_CHARS]
    (vec,) = await embedder.embed([summary])
    async with pool.connection() as conn:
        await register_vector_async(conn)
        cur = await conn.execute(
            """
            INSERT INTO customer_episodes
              (customer_id, summary, outcome, content_hash, embedding, embedding_model, embedding_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (
                customer_id,
                summary,
                outcome,
                episode_hash(customer_id, summary, outcome),
                vec,
                model,
                version,
            ),
        )
        return await cur.fetchone() is not None


async def search_episodes(
    pool: Pool, embedder: Embedder, *, customer_id: str, query: str, top_k: int = 3
) -> list[dict[str, str]]:
    (vec,) = await embedder.embed([query])
    async with pool.connection() as conn:
        await register_vector_async(conn)
        cur = await conn.execute(
            """
            SELECT summary, outcome, created_at FROM customer_episodes
             WHERE customer_id = %s
             ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (customer_id, vec, top_k),
        )
        rows = await cur.fetchall()
    return [
        {"summary": r["summary"], "outcome": r["outcome"], "at": str(r["created_at"])} for r in rows
    ]
