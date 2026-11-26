"""What the graph's history_check node knows about a customer's earlier conversations (E6-1).

A returning customer has an earlier session still stored, or a saved episode. Sessions are pruned
after the retention period, so "new" means "nothing stored", not "never contacted us".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from support_assistant import repositories as repo
from support_assistant.config import Settings
from support_assistant.context import episode_context, history_context
from support_assistant.db import Pool
from support_assistant.embeddings import Embedder
from support_assistant.logs import error_fields
from support_assistant.services.episodes import search_episodes

log = structlog.get_logger()


@dataclass(frozen=True)
class CustomerMemory:
    returning: bool
    note: str = ""
    complete: bool = True  # False: part of the lookup failed, so it is worth trying again


# (customer_id, session_id, the customer's latest message)
HistoryLookup = Callable[[str, str, str], Awaitable[CustomerMemory]]


async def load_customer_memory(
    pool: Pool,
    embedder: Embedder | None,
    settings: Settings,
    customer_id: str,
    session_id: str,
    query: str,
) -> CustomerMemory:
    """Raises only if it cannot tell whether the customer is returning. A failure while fetching
    the memory itself returns what could be read, marked incomplete."""
    async with pool.connection() as conn:
        returning = await repo.has_earlier_history(conn, customer_id, session_id)
    if not returning:
        return CustomerMemory(returning=False)  # nothing to fetch, so no embedding call

    notes: list[str] = []
    complete = True
    if settings.history_days:  # 0 turns the recent-conversations note off
        try:
            async with pool.connection() as conn:
                rows = await repo.recent_sessions(
                    conn, customer_id, settings.history_days, settings.history_max_sessions + 1
                )
            prior = [r for r in rows if r["session_id"] != session_id]
            notes.append(history_context(prior[: settings.history_max_sessions]))
        except Exception as exc:  # noqa: BLE001 - history is a bonus, never a blocker
            complete = False
            log.warning("recent_history_failed", **error_fields(exc))
    if embedder is not None and query:
        try:
            past = await search_episodes(pool, embedder, customer_id=customer_id, query=query)
            notes.append(episode_context(past))
        except Exception as exc:  # noqa: BLE001 - memory is a bonus, never a blocker
            complete = False
            log.warning("episode_recall_failed", **error_fields(exc))
    return CustomerMemory(True, "\n".join(n for n in notes if n), complete)
