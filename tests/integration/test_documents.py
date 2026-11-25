"""Real Postgres: document mirroring, customer uploads, and the least-privilege grants.

Skipped unless IT_DATABASE_URL is set (CI only). These check the SQL itself, which the unit tests'
fakes cannot: type handling in `= ANY(...)`, the CHECK constraints, and what Postgres actually
lets each role do.
"""

from __future__ import annotations

import psycopg
import pytest

from support_assistant import dbroles, indexer
from support_assistant.blobstore import InMemoryBlobStore
from support_assistant.docsource import StoreDocumentSource
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import NotFoundError, UploadRejected
from support_assistant.governance import erase_customer_memory
from support_assistant.uploads import SCAN_TAG, UploadService, UploadStatus

from .conftest import EMBEDDING_DIM, needs_db

pytestmark = needs_db

DOC = "# Guide\n\n## No signal\n\nRestart the phone.\n\n## Slow data\n\nCheck usage.\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _embeddings() -> LcEmbeddings:
    return LcEmbeddings(FakeEmbedder(EMBEDDING_DIM))


async def _texts(pool, source: str) -> set[str]:
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT content FROM knowledge_base_chunks WHERE source = %s", (source,)
            )
        ).fetchall()
    return {r["content"] for r in rows}


async def _store(files: dict[str, str]) -> InMemoryBlobStore:
    s = InMemoryBlobStore()
    for name, text in files.items():
        await s.put(name, text.encode(), content_type="text/markdown")
    return s


# --- documents: the knowledge base mirrors its source ------------------------------------------------------
async def test_an_edited_document_loses_its_old_text_in_the_real_database(pool):
    chain = indexer.build_index_chain(pool, _embeddings(), "fake")
    await chain.ainvoke({"source": "kb.md", "text": DOC})
    assert "Restart the phone." in await _texts(pool, "kb.md")
    result = await chain.ainvoke({"source": "kb.md", "text": DOC.replace("Restart", "Reboot")})
    texts = await _texts(pool, "kb.md")
    assert result["added"] == 1 and result["removed"] == 1
    assert "Reboot the phone." in texts and "Restart the phone." not in texts


async def test_a_deleted_document_disappears_and_an_empty_listing_deletes_nothing(pool):
    emb = _embeddings()
    both = StoreDocumentSource(await _store({"a.md": DOC, "b.md": "# B\n\nkeep me"}))
    await indexer.index_source(pool, emb, both, "fake")
    assert await _texts(pool, "a.md") and await _texts(pool, "b.md")

    only_b = StoreDocumentSource(await _store({"b.md": "# B\n\nkeep me"}))
    report = await indexer.index_source(pool, emb, only_b, "fake")
    assert await _texts(pool, "a.md") == set() and await _texts(pool, "b.md")
    assert report.sources_removed > 0

    empty = StoreDocumentSource(await _store({}))  # storage outage / wrong container
    await indexer.index_source(pool, emb, empty, "fake")
    assert await _texts(pool, "b.md")  # nothing was wiped


# --- customer uploads: real constraints and queries ------------------------------------------------------
def _svc(pool, blobs=None, daily=10):
    blobs = blobs or InMemoryBlobStore()
    return UploadService(pool, blobs, max_bytes=1_000_000, daily_limit=daily), blobs


async def test_an_upload_is_recorded_scanned_and_private_to_its_customer(seeded):
    svc, blobs = _svc(seeded)
    rec = await svc.accept("C1001", "S1", "shot.png", PNG)
    assert (await svc.status("C1001", rec.id)).status is UploadStatus.PENDING_SCAN

    (name,) = blobs.blobs
    blobs.set_tags(name, {SCAN_TAG: "No threats found"})
    assert (await svc.status("C1001", rec.id)).status is UploadStatus.CLEAN
    # persisted: a fresh service instance reads CLEAN from the database, no tag needed
    fresh, _ = _svc(seeded, InMemoryBlobStore())
    assert (await fresh.status("C1001", rec.id)).status is UploadStatus.CLEAN

    with pytest.raises(NotFoundError):
        await svc.status("C1002", rec.id)  # another customer's upload id: not found
    assert await svc.list_for("C1002") == []


async def test_a_malicious_file_is_deleted_but_the_audit_row_stays(seeded):
    svc, blobs = _svc(seeded)
    rec = await svc.accept("C1001", "S1", "a.png", PNG)
    (name,) = blobs.blobs
    blobs.set_tags(name, {SCAN_TAG: "Malicious"})
    assert (await svc.status("C1001", rec.id)).status is UploadStatus.REJECTED
    assert blobs.blobs == {}
    assert (await svc.list_for("C1001"))[0].status is UploadStatus.REJECTED


async def test_the_daily_limit_counts_real_rows(seeded):
    svc, _ = _svc(seeded, daily=2)
    await svc.accept("C1001", "S", "a.png", PNG)
    await svc.accept("C1001", "S", "b.png", PNG)
    with pytest.raises(UploadRejected) as e:
        await svc.accept("C1001", "S", "c.png", PNG)
    assert e.value.http_status == 429
    await svc.accept("C1002", "S", "a.png", PNG)  # another customer is unaffected


async def test_the_database_itself_refuses_a_bad_status_and_an_empty_file(seeded):
    async with seeded.connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO customer_uploads (id, customer_id, session_id, blob_name, original_name,"
                " content_type, size_bytes, sha256, status) VALUES (gen_random_uuid(), 'C1001', 'S',"
                " 'x/1', 'a.png', 'image/png', 10, repeat('a', 64), 'WHATEVER')"
            )
    async with seeded.connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO customer_uploads (id, customer_id, session_id, blob_name, original_name,"
                " content_type, size_bytes, sha256, status) VALUES (gen_random_uuid(), 'C1001', 'S',"
                " 'x/2', 'a.png', 'image/png', 0, repeat('a', 64), 'CLEAN')"
            )


async def test_erasure_removes_the_customers_files_and_rows_only(seeded):
    svc, blobs = _svc(seeded)
    await svc.accept("C1001", "S", "a.png", PNG)
    await svc.accept("C1002", "S", "b.png", PNG)
    out = await erase_customer_memory(seeded, [], "C1001", uploads=svc)
    assert out["uploads"] == 1
    assert [n.split("/")[0] for n in blobs.blobs] == ["C1002"]
    assert await svc.list_for("C1001") == [] and len(await svc.list_for("C1002")) == 1


# --- least privilege: ask Postgres what each role can really do ---------------------------------------------
async def _grant(pool, workload: str) -> str:
    role = f"it-{workload}"
    async with pool.connection() as conn:
        await conn.execute(
            f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
            f'THEN CREATE ROLE "{role}" NOLOGIN; END IF; END $$'
        )
        for stmt in dbroles.grant_statements(role, workload):
            await conn.execute(stmt)
        await conn.commit()
    return role


async def _can(pool, role: str, table: str, privilege: str) -> bool:
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT has_table_privilege(%s, %s, %s) AS ok", (role, table, privilege)
            )
        ).fetchone()
    return bool(row["ok"])


async def test_each_service_can_do_exactly_what_it_needs_and_no_more(pool):
    mcp = await _grant(pool, "mcp-servers")
    assert await _can(pool, mcp, "bills", "UPDATE") and await _can(
        pool, mcp, "transaction_history", "INSERT"
    )
    assert not await _can(pool, mcp, "bills", "DELETE") and not await _can(
        pool, mcp, "bills", "INSERT"
    )
    assert not await _can(pool, mcp, "knowledge_base_chunks", "SELECT")

    orch = await _grant(pool, "orchestrator")
    assert await _can(pool, orch, "audit_events", "INSERT")
    assert not await _can(pool, orch, "knowledge_base_chunks", "SELECT")
    assert not await _can(pool, orch, "audit_events", "UPDATE") and not await _can(
        pool, orch, "audit_events", "DELETE"
    )
    assert not await _can(pool, orch, "bills", "SELECT")

    ret = await _grant(pool, "retrieval")
    assert await _can(pool, ret, "knowledge_base_chunks", "SELECT") and not await _can(
        pool, ret, "knowledge_base_chunks", "INSERT"
    )

    ix = await _grant(pool, "indexer")
    assert await _can(pool, ix, "knowledge_base_chunks", "DELETE") and not await _can(
        pool, ix, "customers", "SELECT"
    )

    be = await _grant(pool, "fat-backend")
    assert await _can(pool, be, "customer_uploads", "DELETE") and await _can(
        pool, be, "session_conversations", "UPDATE"
    )
    assert not await _can(pool, be, "bills", "SELECT") and not await _can(
        pool, be, "transaction_history", "SELECT"
    )

    for blind in ("agent-director", "agent-troubleshooter", "agent-evaluator"):
        role = await _grant(pool, blind)
        assert not await _can(pool, role, "customers", "SELECT")  # they reach data only through MCP


async def test_a_writing_role_can_use_the_sequences_it_needs_and_a_reader_cannot(pool):
    orch = await _grant(pool, "orchestrator")  # inserts into audit_events (BIGSERIAL)
    trb = await _grant(pool, "retrieval")
    async with pool.connection() as conn:
        w = await (
            await conn.execute(
                "SELECT has_sequence_privilege(%s, 'audit_events_id_seq', 'USAGE') AS ok", (orch,)
            )
        ).fetchone()
        r = await (
            await conn.execute(
                "SELECT has_sequence_privilege(%s, 'audit_events_id_seq', 'USAGE') AS ok", (trb,)
            )
        ).fetchone()
    assert w["ok"] is True and r["ok"] is False


async def test_regranting_removes_access_that_is_no_longer_listed(pool):
    role = await _grant(pool, "mcp-servers")
    async with pool.connection() as conn:  # simulate drift: someone granted too much by hand
        await conn.execute(f'GRANT DELETE ON "bills" TO "{role}"')
        await conn.commit()
    assert await _can(pool, role, "bills", "DELETE")
    await _grant(pool, "mcp-servers")  # the next deploy re-applies the matrix
    assert not await _can(pool, role, "bills", "DELETE")
