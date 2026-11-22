"""Integration tests against REAL Postgres (pgvector) and Redis Stack.

They run in CI only, against throwaway service containers (see the `integration` job in ci.yml).
Nothing runs locally: without the env vars below, every test here is skipped.

    IT_DATABASE_URL  postgresql://postgres:postgres@localhost:5432/it
    IT_REDIS_URL     redis://localhost:6379/0   (Redis Stack: RediSearch + RedisJSON)
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DB_URL = os.environ.get("IT_DATABASE_URL")
REDIS_URL = os.environ.get("IT_REDIS_URL")
EMBEDDING_DIM = 8  # small vectors: the FakeEmbedder produces this size

needs_db = pytest.mark.skipif(not DB_URL, reason="IT_DATABASE_URL not set (runs in CI only)")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="IT_REDIS_URL not set (runs in CI only)")

TABLES = (
    "audit_events, transaction_history, bill_line_items, bills, devices, lines, payment_methods, "
    "plans, session_conversations, customer_episodes, knowledge_base_chunks, customers"
)


@pytest.fixture(scope="session")
def migrated() -> str:
    """Run the real Alembic migrations once against the empty test database."""
    if not DB_URL:
        pytest.skip("IT_DATABASE_URL not set")
    env = {**os.environ, "DATABASE_URL": DB_URL, "EMBEDDING_DIM": str(EMBEDDING_DIM)}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
    )
    return DB_URL


@pytest.fixture
async def pool(migrated: str) -> AsyncIterator[object]:
    from support_assistant.config import Settings
    from support_assistant.db import make_pool

    settings = Settings(_env_file=None, database_url=migrated, redis_url=REDIS_URL or "redis://x")
    p = make_pool(settings)
    await p.open()
    async with p.connection() as conn:  # every test starts from empty tables
        await conn.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
        await conn.commit()
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
async def seeded(pool: object) -> object:
    """The example seed (C1002 owes B1005), loaded through the real loader."""
    from support_assistant.seed import load_seed, parse_seed

    text = (ROOT / "seed" / "example_seed.toml").read_text(encoding="utf-8")
    await load_seed(pool, parse_seed(text))  # type: ignore[arg-type]
    return pool


@pytest.fixture
async def redis() -> AsyncIterator[object]:
    if not REDIS_URL:
        pytest.skip("IT_REDIS_URL not set")
    from redis.asyncio import Redis

    client = Redis.from_url(REDIS_URL)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.aclose()
