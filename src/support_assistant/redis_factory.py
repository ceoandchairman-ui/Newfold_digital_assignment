"""Redis clients for the three clusters, with Entra (token) auth for Azure Managed Redis (E10-6).

With REDIS_ENTRA_AUTH=true, a client authenticates as its pod's workload identity: username = the
identity's object id, password = an Entra access token for Redis. Tokens are cached and refreshed
5 minutes before they expire, so no request waits for a token fetch in steady state. Without it,
the URL's own credentials (e.g. an access key from Key Vault) are used, as in dev.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio import Redis
from redis.credentials import CredentialProvider

from support_assistant.config import Settings

REDIS_ENTRA_SCOPE = "https://redis.azure.com/.default"
TokenWithExpiry = tuple[str, float]  # (token, expires_on as a unix timestamp)
TokenFetcher = Callable[[], Awaitable[TokenWithExpiry]]


class EntraRedisCredentials(CredentialProvider):
    def __init__(
        self,
        username: str,
        fetch: TokenFetcher,
        refresh_margin_s: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._username, self._fetch = username, fetch
        self._margin, self._clock = refresh_margin_s, clock
        self._token: str | None = None
        self._expires_on = 0.0

    async def get_credentials_async(self) -> tuple[str, str]:
        if self._token is None or self._clock() >= self._expires_on - self._margin:
            self._token, self._expires_on = await self._fetch()
        return self._username, self._token

    def get_credentials(self) -> tuple[str, str]:
        raise NotImplementedError("async clients only: use get_credentials_async")


def entra_token_fetcher() -> TokenFetcher:  # pragma: no cover - needs Azure identity
    from azure.identity.aio import DefaultAzureCredential

    credential = DefaultAzureCredential()

    async def fetch() -> TokenWithExpiry:
        tok = await credential.get_token(REDIS_ENTRA_SCOPE)
        return tok.token, float(tok.expires_on)

    return fetch


def make_redis(
    url: str,
    settings: Settings,
    *,
    decode_responses: bool = False,
    fetcher: TokenFetcher | None = None,
) -> Redis:
    kwargs: dict[str, Any] = {"decode_responses": decode_responses}
    if settings.redis_entra_auth:
        if not settings.redis_entra_object_id:
            raise RuntimeError("REDIS_ENTRA_OBJECT_ID is required when REDIS_ENTRA_AUTH=true")
        kwargs["credential_provider"] = EntraRedisCredentials(
            settings.redis_entra_object_id, fetcher or entra_token_fetcher()
        )
    return Redis.from_url(url, **kwargs)
