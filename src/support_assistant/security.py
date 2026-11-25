"""Security and resilience building blocks for the fat backend.

- RateLimiter: counters in Redis, used to lock out identity-guessing and to cap request rates.
- Bulkhead: a concurrency cap that sheds load (fast 503) instead of queueing until everything times out.
- valid_session_id: a session id is a bearer token, so it must be unguessable.
- SecurityHeadersMiddleware, BodyLimitMiddleware: standard hardening for an API that serves no pages.
- client_ip: the caller's address, trusting only the proxies we run behind.

Everything here is plain code with no model, and each piece is tested on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from support_assistant.errors import Overloaded, RateLimited

# A session id stands in for a login for as long as it lives, so it has to be long and random.
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def valid_session_id(session_id: str, min_length: int) -> bool:
    """True if the id is made of safe characters and is at least `min_length` long.

    `min_length` 0 turns the length rule off (tests); the character set is always enforced once a
    length is required.
    """
    if min_length <= 0:
        return bool(session_id)
    return bool(_SESSION_ID.match(session_id)) and len(session_id) >= min_length


def channel_session_id(secret: str, kind: str, identifier: str) -> str:
    """An unguessable session id for a channel whose caller we identify by something guessable.

    A phone number is not a secret. Deriving the id with a keyed hash means that knowing a number does
    not let anyone compute the session id, and so cannot reach the web API as that customer.
    """
    digest = hmac.new(secret.encode(), f"{kind}:{identifier}".encode(), hashlib.sha256).hexdigest()
    return f"{kind}-{digest[:48]}"


def subject_key(*parts: str) -> str:
    """A stable, non-reversible key for 'the thing being guessed' (a phone, a name), for counters."""
    return hashlib.sha256("|".join(p.strip().lower() for p in parts).encode()).hexdigest()[:32]


def client_ip(headers: Mapping[str, str], peer: str | None, trusted_hops: int) -> str:
    """The caller's address. `X-Forwarded-For` is only believed as far as we run proxies ourselves.

    With 0 trusted hops the header is ignored entirely (anyone can send it), and the direct peer is
    used. With N hops, the address N entries from the right is the one our outermost proxy saw.
    """
    if trusted_hops > 0:
        forwarded = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= trusted_hops:
            return forwarded[-trusted_hops]
    return peer or "unknown"


class RateLimiter:
    """Fixed-window counters in Redis. Two uses: cap a request rate, and lock out repeated failures."""

    def __init__(self, redis: Any, prefix: str = "rl:") -> None:
        self._redis, self._prefix = redis, prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def hit(self, key: str, limit: int, window_s: int) -> None:
        """Count one request; raise RateLimited once more than `limit` arrive in the window."""
        k = self._key(key)
        await self._redis.set(k, 0, ex=window_s, nx=True)  # the TTL is set once, with the key
        count = int(await self._redis.incr(k))
        if count > limit:
            raise RateLimited(retry_after=int(await self._redis.ttl(k)))

    async def blocked(self, key: str, limit: int) -> int | None:
        """Seconds to wait if `key` already has `limit` recorded failures, else None."""
        k = self._key(key)
        raw = await self._redis.get(k)
        if raw is not None and int(raw) >= limit:
            return max(1, int(await self._redis.ttl(k)))
        return None

    async def record(self, key: str, window_s: int) -> None:
        """Record one failure. The window starts at the first failure and does not slide."""
        k = self._key(key)
        await self._redis.set(k, 0, ex=window_s, nx=True)
        await self._redis.incr(k)

    async def reset(self, key: str) -> None:
        await self._redis.delete(self._key(key))


class Bulkhead:
    """At most `limit` things in flight. A request over the limit is refused at once, not queued."""

    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(limit)
        self.limit = limit

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._sem.locked():
            raise Overloaded()
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


# --- ASGI middleware -----------------------------------------------------------------------------
_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
]


class SecurityHeadersMiddleware:
    """Adds the standard hardening headers to every response. This API serves JSON and TwiML, never
    pages, so the policy can be as strict as it gets. No CORS headers are ever added: browsers from
    other origins are refused by default."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_hardened(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers") or []}
                extra = [(k, v) for k, v in _HEADERS if k not in present]
                message = {**message, "headers": [*(message.get("headers") or []), *extra]}
            await send(message)

        await self.app(scope, receive, send_hardened)


DEFAULT_MAX_BODY = 64_000


class BodyLimitMiddleware:
    """Refuses a request body over `max_bytes` before it is read (413), and a POST that does not say
    how big it is (411), except on the prefixes that take real files, which have their own limit."""

    def __init__(
        self, app: Any, max_bytes: int = DEFAULT_MAX_BODY, exempt: tuple[str, ...] = ("/upload",)
    ) -> None:
        self.app, self._max, self._exempt = app, max_bytes, exempt

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        if scope["path"].startswith(self._exempt):
            await self.app(scope, receive, send)
            return
        length = dict(scope.get("headers") or []).get(b"content-length")
        status = 0
        if length is None:
            status = 411
        else:
            try:
                status = 413 if int(length) > self._max else 0
            except ValueError:
                status = 400
        if status:
            body = b'{"error":"request_too_large","message":"That request is too large."}'
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
