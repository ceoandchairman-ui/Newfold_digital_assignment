"""Data governance: erasure on request and retention (E5-8).

Scope: conversational and memory data only (session history, episodes, cached copies). The billing
ledger (bills, transaction_history) is a financial record with its own legal retention, so it is
NEVER deleted here. Retention periods and the legal basis come from the compliance owner; the
numbers below are configuration, not decisions.
"""

from __future__ import annotations

from typing import Any

from support_assistant.db import Pool


async def erase_customer_memory(
    pool: Pool, redis_clients: list[Any], customer_id: str
) -> dict[str, int]:
    """Delete a customer's conversations, episodes and cached copies. Returns what was removed."""
    async with pool.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "DELETE FROM session_conversations WHERE customer_id = %s", (customer_id,)
            )
            sessions = cur.rowcount
            cur = await conn.execute(
                "DELETE FROM customer_episodes WHERE customer_id = %s", (customer_id,)
            )
            episodes = cur.rowcount

    cache_keys = 0
    # Cache cluster (window) and tool-cache cluster. Verified flags are keyed by session id, not
    # customer, so they are not matched here; they expire by TTL.
    for client in redis_clients:
        for pattern in (f"sess:{customer_id}:*", f"tool:{customer_id}"):
            async for key in client.scan_iter(match=pattern):
                cache_keys += await client.delete(key)
    return {"sessions": sessions, "episodes": episodes, "cache_keys": cache_keys}


async def prune_sessions(pool: Pool, older_than_days: int) -> int:
    """Retention: delete conversation rows not updated for `older_than_days`. Batched by the caller."""
    if older_than_days < 1:
        raise ValueError("older_than_days must be at least 1")
    async with pool.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "DELETE FROM session_conversations WHERE updated_at < now() - make_interval(days => %s)",
                (older_than_days,),
            )
            return int(cur.rowcount)
