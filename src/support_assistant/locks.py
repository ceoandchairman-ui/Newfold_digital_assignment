"""One active graph run per session (E4-12), as a Redis lock.

Standard single-node pattern: `SET key token NX PX ttl` to acquire, and a Lua compare-and-delete to
release, so a run can only ever release its own lock (never one taken by a later run after expiry).

The TTL is a safety net for a crashed pod, not a timeout for the run. While the run is alive a
heartbeat renews the lock every third of the TTL, so a slow run (a payment resume has no deadline)
keeps it, and a dead pod's lock still expires.

This lock is NOT a strict guarantee. Redis replicates asynchronously, so a failover can lose a lock,
and a paused process can miss renewals. It prevents ordinary double runs. What must never happen
twice (a charge) is protected in Postgres, by the unique idempotency key and the bill row lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog

from support_assistant.errors import ConflictError
from support_assistant.logs import error_fields

_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

# Extend the expiry only if the lock is still ours.
_RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""

log = structlog.get_logger()


class RunLock:
    def __init__(self, redis: Any, ttl_ms: int = 120_000, prefix: str = "lock:run:") -> None:
        self._redis, self._ttl_ms, self._prefix = redis, ttl_ms, prefix

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        key, token = f"{self._prefix}{session_id}", uuid.uuid4().hex
        if not await self._redis.set(key, token, nx=True, px=self._ttl_ms):
            log.warning("run_lock_busy", session_id=session_id)
            raise ConflictError("another run is active for this session")
        heartbeat = asyncio.create_task(self._keep_alive(key, token))
        try:
            yield
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            try:
                await self._redis.eval(_RELEASE, 1, key, token)
            except Exception as exc:  # noqa: BLE001 - the TTL frees it; keep the run's own result
                log.warning("run_lock_release_failed", **error_fields(exc))

    async def _keep_alive(self, key: str, token: str) -> None:
        while True:
            await asyncio.sleep(self._ttl_ms / 3000)
            try:
                renewed = await self._redis.eval(_RENEW, 1, key, token, self._ttl_ms)
            except Exception as exc:  # noqa: BLE001 - try again at the next beat
                log.warning("run_lock_renew_failed", **error_fields(exc))
                continue
            if not renewed:
                log.error("run_lock_lost")
                return
