"""Short-term conversation memory: Redis hot window over durable Postgres (D6).

Reads are cache-aside, writes are write-through (Postgres first, then Redis). If Redis is down the
Postgres copy still serves the request, so Redis is an optimisation, never the source of truth.
Key: sess:{customer_id}:{session_id}  (customer-scoped so one customer can never read another's).
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from psycopg.types.json import Jsonb
from redis.asyncio import Redis

from support_assistant.db import Pool

log = structlog.get_logger()

Message = dict[str, Any]


class SessionMemory:
    def __init__(self, pool: Pool, redis: Redis, window: int, ttl_s: int) -> None:
        self._pool, self._redis, self._window, self._ttl = pool, redis, window, ttl_s

    @staticmethod
    def _key(customer_id: str, session_id: str) -> str:
        return f"sess:{customer_id}:{session_id}"

    async def load(self, customer_id: str, session_id: str) -> list[Message]:
        key = self._key(customer_id, session_id)
        try:
            cached = await self._redis.get(key)
            if cached is not None:
                cached_messages: list[Message] = json.loads(cached)
                return cached_messages
        except Exception:  # noqa: BLE001 - Redis is optional
            log.warning("redis_read_failed")
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT messages FROM session_conversations WHERE session_id = %s AND customer_id = %s",
                (session_id, customer_id),
            )
            row = await cur.fetchone()
        messages: list[Message] = (row["messages"] if row else [])[-self._window :]
        await self._cache(key, messages)
        return messages

    async def append(self, customer_id: str, session_id: str, new: list[Message]) -> list[Message]:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT messages FROM session_conversations WHERE session_id = %s AND customer_id = %s FOR UPDATE",
                    (session_id, customer_id),
                )
                row = await cur.fetchone()
                messages = [*(row["messages"] if row else []), *new]
                await conn.execute(
                    """
                    INSERT INTO session_conversations (session_id, customer_id, messages, updated_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (session_id) DO UPDATE
                       SET messages = EXCLUDED.messages, updated_at = now()
                     WHERE session_conversations.customer_id = EXCLUDED.customer_id
                    """,
                    (session_id, customer_id, Jsonb(messages)),
                )
        window = messages[-self._window :]
        await self._cache(self._key(customer_id, session_id), window)
        return window

    async def _cache(self, key: str, messages: list[Message]) -> None:
        try:
            await self._redis.set(key, json.dumps(messages), ex=self._ttl)
        except Exception:  # noqa: BLE001
            log.warning("redis_write_failed")
