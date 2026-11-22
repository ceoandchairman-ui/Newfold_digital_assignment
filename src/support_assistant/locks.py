"""One active graph run per session (E4-12), as a Redis lock.

Standard single-node pattern: `SET key token NX PX ttl` to acquire, and a Lua compare-and-delete to
release, so a run can only ever release its own lock (never one taken by a later run after expiry).

The TTL is a safety net for a crashed pod, not a timeout for the run: set it well above the longest
expected turn. If a turn outlives the TTL, a second run could start; the payment path stays safe
because it is idempotent (stable key) and row-locked in Postgres.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from support_assistant.errors import ConflictError

_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class RunLock:
    def __init__(self, redis: Any, ttl_ms: int = 120_000, prefix: str = "lock:run:") -> None:
        self._redis, self._ttl_ms, self._prefix = redis, ttl_ms, prefix

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        key, token = f"{self._prefix}{session_id}", uuid.uuid4().hex
        if not await self._redis.set(key, token, nx=True, px=self._ttl_ms):
            raise ConflictError("another run is active for this session")
        try:
            yield
        finally:
            await self._redis.eval(_RELEASE, 1, key, token)
