"""Per-customer daily token budget (E6-5): caps model spend per customer per day.

Counted in the cache Redis cluster as `budget:{customer_id}:{yyyymmdd}`, expiring after two days.
Tokens are estimated from text length (about 4 characters per token); the cap protects against
runaway or abusive use, it is not billing-accurate.

Fail-open: if Redis is down, the turn is allowed. A Redis outage must not take the assistant down;
the edge rate limits (E10-16) still bound abuse.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

log = structlog.get_logger()

BUDGET_EXCEEDED_MESSAGE = (
    "You've reached today's limit for this assistant. Please try again tomorrow, "
    "or contact us if this is urgent."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TokenBudget:
    def __init__(self, redis: Any, daily_limit: int, prefix: str = "budget:") -> None:
        self._redis, self._limit, self._prefix = redis, daily_limit, prefix

    def _key(self, customer_id: str, now: datetime | None = None) -> str:
        day = (now or datetime.now(UTC)).strftime("%Y%m%d")
        return f"{self._prefix}{customer_id}:{day}"

    async def consume(self, customer_id: str, tokens: int, now: datetime | None = None) -> bool:
        """Add `tokens` to today's total. False if that takes the customer over the limit."""
        key = self._key(customer_id, now)
        try:
            used = int(await self._redis.incrby(key, tokens))
            if used == tokens:  # first use today
                await self._redis.expire(key, 2 * 24 * 3600)
        except Exception:  # noqa: BLE001
            log.warning("token_budget_unavailable")
            return True
        return used <= self._limit
