"""Real Postgres: migrations, seed, tenant scoping, payments under concurrency, pgvector search,
episodes, audit, memory and retention. Skipped unless IT_DATABASE_URL is set (CI only)."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import psycopg
import pytest

from support_assistant import repositories as repo
from support_assistant.audit import PgAuditSink
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ConflictError, DuplicateOperation, NotFoundError
from support_assistant.governance import erase_customer_memory, prune_sessions
from support_assistant.indexer import build_index_chain
from support_assistant.memory import SessionMemory
from support_assistant.seed import load_seed, parse_seed
from support_assistant.services import episodes, payments
from support_assistant.services.retrieval import search_kb

from .conftest import EMBEDDING_DIM, ROOT, needs_db

pytestmark = needs_db


# --- concurrent first writes (the lost update memory.py guards against) ------------------------------
class _NoCache:
    async def set(self, *a, **k):
        return None


async def test_many_first_writes_to_one_session_lose_no_message(seeded):
    memory = SessionMemory(seeded, _NoCache(), window=100, ttl_s=60)
    await asyncio.gather(
        *(
            memory.append("C1001", "concurrent-first", [{"role": "user", "content": str(i)}])
            for i in range(8)
        )
    )
    async with seeded.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT messages FROM session_conversations WHERE session_id = 'concurrent-first'"
            )
        ).fetchone()
    assert sorted(m["content"] for m in row["messages"]) == [str(i) for i in range(8)]


# --- schema and seed ----------------------------------------------------------------------------------
async def test_seed_loads_and_reloading_it_changes_nothing(seeded):
    async with seeded.connection() as conn:
        before = await (await conn.execute("SELECT count(*) AS n FROM bills")).fetchone()
    text = (ROOT / "seed" / "example_seed.toml").read_text(encoding="utf-8")
    await load_seed(seeded, parse_seed(text))  # idempotent
    async with seeded.connection() as conn:
        after = await (await conn.execute("SELECT count(*) AS n FROM bills")).fetchone()
        items = await (await conn.execute("SELECT count(*) AS n FROM bill_line_items")).fetchone()
    assert before["n"] == after["n"] == 2 and items["n"] == 1


async def test_the_database_allows_only_one_awaiting_payment_bill_per_customer(seeded):
    async with seeded.connection() as conn:
        await conn.execute("UPDATE bills SET status = 'AWAITING_PAYMENT' WHERE id = 'B1005'")
        with pytest.raises(psycopg.errors.UniqueViolation):
            await conn.execute(
                "INSERT INTO bills (id, customer_id, amount_due, due_date, status) "
                "VALUES ('B9', 'C1002', 1, '2026-12-01', 'AWAITING_PAYMENT')"
            )


async def test_an_unknown_bill_status_is_rejected_by_the_database(seeded):
    async with seeded.connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute("UPDATE bills SET status = 'MAYBE' WHERE id = 'B1005'")


# --- tenant scoping with real SQL ----------------------------------------------------------------------------
async def test_a_customer_never_sees_another_customers_rows(seeded):
    async with seeded.connection() as conn:
        assert [b["id"] for b in await repo.list_bills(conn, "C1001")] == ["B2001"]
        assert [ln["id"] for ln in await repo.list_lines(conn, "C1002")] == ["L2"]
        with pytest.raises(NotFoundError):
            await repo.get_bill(conn, "C1001", "B1005")  # exists, but not theirs


async def test_overdue_count_and_contract_end_from_real_data(seeded):
    async with seeded.connection() as conn:
        assert await repo.count_overdue_bills(conn, "C1002", date(2026, 11, 21)) == 1
        assert await repo.count_overdue_bills(conn, "C1001", date(2026, 11, 21)) == 0
        assert await repo.latest_contract_end(conn, "C1001") == date(2027, 6, 1)


# --- payments: real row locks and constraints ----------------------------------------------------------------
async def _pay(pool, key: str, amount: str = "42.50", gateway=None):
    return await payments.pay_bill(
        pool,
        gateway or payments.StubGateway(),
        customer_id="C1002",
        bill_id="B1005",
        amount=Decimal(amount),
        idempotency_key=key,
    )


async def test_a_payment_marks_the_bill_paid_and_records_one_transaction(seeded):
    res = await _pay(seeded, "K1")
    async with seeded.connection() as conn:
        bill = await repo.get_bill(conn, "C1002", "B1005")
        tx = await repo.find_transaction(conn, "K1")
        n = await repo.recent_payment_count(conn, "C1002", "B1005", 10)
    assert (
        bill["status"] == "PAID" and tx["status"] == "SUCCEEDED" and tx["id"] == res.transaction_id
    )
    assert n == 1


async def test_replaying_the_same_key_returns_the_original_and_never_charges_twice(seeded):
    first = await _pay(seeded, "K1")
    again = await _pay(seeded, "K1")
    async with seeded.connection() as conn:
        n = await (await conn.execute("SELECT count(*) AS n FROM transaction_history")).fetchone()
    assert again.replayed and again.transaction_id == first.transaction_id and n["n"] == 1


async def test_two_concurrent_payments_for_one_bill_exactly_one_succeeds(seeded):
    """The real FOR UPDATE row lock and state machine decide; no double charge."""
    results = await asyncio.gather(_pay(seeded, "KA"), _pay(seeded, "KB"), return_exceptions=True)
    ok = [r for r in results if isinstance(r, payments.PaymentResult)]
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(ok) == 1 and len(errors) == 1
    assert isinstance(errors[0], (ConflictError, DuplicateOperation))
    async with seeded.connection() as conn:
        rows = await (
            await conn.execute("SELECT status FROM transaction_history ORDER BY id")
        ).fetchall()
    assert [r["status"] for r in rows].count("SUCCEEDED") == 1


async def test_a_paid_bill_cannot_be_paid_again(seeded):
    await _pay(seeded, "K1")
    with pytest.raises(DuplicateOperation):
        await _pay(seeded, "K2")


# --- pgvector + keyword hybrid search, through the real indexer chain -----------------------------------------
MANUAL = """# Guide

## No signal

Toggle airplane mode off and on, then restart the phone.

## Mobile data

If mobile data fails, reset the APN settings to the carrier default.
"""


async def test_indexing_then_searching_finds_the_right_section(pool, monkeypatch):
    emb = FakeEmbedder(EMBEDDING_DIM)
    chain = build_index_chain(pool, LcEmbeddings(emb), "fake")
    first = await chain.ainvoke({"source": "guide.md", "text": MANUAL})
    second = await chain.ainvoke({"source": "guide.md", "text": MANUAL})
    assert first["added"] == first["total"] >= 2 and second["added"] == 0  # idempotent

    # The keyword half of hybrid search must find the APN section by its exact term.
    hits = await search_kb(pool, emb, "APN settings", top_k=2, candidates=5)
    assert hits and "APN" in hits[0].content and hits[0].heading_path == "Guide > Mobile data"


# --- episodes, audit, memory, governance ---------------------------------------------------------------------
async def test_episodes_are_stored_once_and_searched_per_customer(seeded):
    emb = FakeEmbedder(EMBEDDING_DIM)
    kw = {"summary": "no signal fixed by reboot", "outcome": "resolved", "model": "fake"}
    assert await episodes.save_episode(seeded, emb, customer_id="C1001", **kw)
    assert not await episodes.save_episode(seeded, emb, customer_id="C1001", **kw)  # duplicate
    mine = await episodes.search_episodes(seeded, emb, customer_id="C1001", query="no signal")
    theirs = await episodes.search_episodes(seeded, emb, customer_id="C1002", query="no signal")
    assert len(mine) == 1 and theirs == []


async def test_audit_rows_are_written(seeded):
    await PgAuditSink(seeded).write(
        "S1", "C1001", [{"agent": "director", "tool": "get_bills", "ok": True}]
    )
    async with seeded.connection() as conn:
        row = await (await conn.execute("SELECT agent, tool, ok FROM audit_events")).fetchone()
    assert row == {"agent": "director", "tool": "get_bills", "ok": True}


async def test_erasure_removes_memory_but_keeps_the_ledger(seeded):
    async with seeded.connection() as conn:
        await conn.execute(
            "INSERT INTO session_conversations (session_id, customer_id) VALUES ('S1', 'C1001')"
        )
        await conn.commit()
    out = await erase_customer_memory(seeded, [], "C1001")
    async with seeded.connection() as conn:
        bills = await (
            await conn.execute("SELECT count(*) AS n FROM bills WHERE customer_id = 'C1001'")
        ).fetchone()
    assert out["sessions"] == 1 and bills["n"] == 1


async def test_retention_prunes_only_old_sessions(seeded):
    async with seeded.connection() as conn:
        await conn.execute(
            "INSERT INTO session_conversations (session_id, customer_id, updated_at) VALUES "
            "('old', 'C1001', now() - interval '100 days'), ('new', 'C1001', now())"
        )
        await conn.commit()
    assert await prune_sessions(seeded, 90) == 1
