"""Entra-token Postgres connections (E1-4, E10-6): a fresh token per new connection."""

from psycopg.rows import dict_row

from support_assistant.config import Settings
from support_assistant.db import connection_kwargs, make_pool


def test_without_entra_the_kwargs_are_static():
    kw = connection_kwargs(None)
    assert kw == {"row_factory": dict_row, "autocommit": False}


async def test_with_entra_every_new_connection_gets_a_fresh_token():
    tokens = iter(["tok-1", "tok-2"])

    async def provider() -> str:
        return next(tokens)

    make_kwargs = connection_kwargs(provider)
    first, second = await make_kwargs(), await make_kwargs()
    assert first["password"] == "tok-1" and second["password"] == "tok-2"
    assert first["sslmode"] == "require" and first["row_factory"] is dict_row


def test_pool_recycles_connections_before_tokens_expire():
    s = Settings(_env_file=None, database_url="postgresql://u@h/db", redis_url="redis://x")

    async def provider() -> str:
        return "t"

    pool = make_pool(s, provider)
    assert pool.max_lifetime <= 60 * 60 and callable(pool.kwargs)
