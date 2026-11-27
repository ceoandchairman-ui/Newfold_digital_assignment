"""Async Postgres pool (E1-4).

Production Postgres is Entra-only (password auth is disabled in Terraform). Entra access tokens
expire after about an hour, so a static password in DATABASE_URL cannot work. Instead the pool asks
for connection arguments per NEW connection, and each one gets a fresh token as its password
(`DATABASE_ENTRA_AUTH=true`). DATABASE_URL then carries host, database and user only.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from support_assistant.config import Settings

# Every connection from our pool returns rows as dicts.
Conn = AsyncConnection[DictRow]
Pool = AsyncConnectionPool[Conn]

POSTGRES_ENTRA_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"
TokenProvider = Callable[[], Awaitable[str]]


def entra_token_provider() -> TokenProvider:  # pragma: no cover - needs Azure identity
    from azure.identity.aio import DefaultAzureCredential

    credential = DefaultAzureCredential()

    async def token() -> str:
        return (await credential.get_token(POSTGRES_ENTRA_SCOPE)).token

    return token


def connection_kwargs(token_provider: TokenProvider | None, user: str | None = None) -> Any:
    """Static kwargs, or an async callable that adds a fresh token for each new connection.

    Under Entra auth the Postgres role is the identity's name, which differs per service, so it is
    supplied here (DATABASE_ENTRA_USER) rather than baked into the shared DATABASE_URL.
    """
    base: dict[str, Any] = {"row_factory": dict_row, "autocommit": False}
    if token_provider is None:
        return base

    async def with_token() -> dict[str, Any]:
        extra = {"user": user} if user else {}
        return {**base, **extra, "password": await token_provider(), "sslmode": "require"}

    return with_token


def make_pool(
    settings: Settings, token_provider: TokenProvider | None = None, dbname: str | None = None
) -> Pool:
    """Create (but do not open) the pool. Call `await pool.open()` at startup.

    `dbname` connects to another database on the same server."""
    if token_provider is None and settings.database_entra_auth:
        token_provider = entra_token_provider()
    conninfo = settings.database_url.get_secret_value()
    if dbname is not None:
        conninfo = make_conninfo(conninfo, dbname=dbname)
    return cast(
        Pool,
        AsyncConnectionPool(
            conninfo=conninfo,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            timeout=settings.db_pool_timeout_s,
            # Recycle connections before their token could matter (tokens last ~60 minutes).
            max_lifetime=45 * 60,
            kwargs=connection_kwargs(token_provider, settings.database_entra_user),
            open=False,
        ),
    )
