"""The LCEL indexing chain end to end with a fake pool and the fake embedder.

The fake database implements just enough SQL semantics to prove the mirroring behaviour: new
chunks are added, unchanged ones are skipped, and chunks no longer in a document are removed.
"""

from contextlib import asynccontextmanager

from support_assistant import indexer
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings

DOC = "# Guide\n\nIntro.\n\n## No signal\n\nRestart the phone.\n\n## Slow data\n\nCheck usage.\n"


class FakeCursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows, self.rowcount = rows or [], rowcount

    async def fetchall(self):
        return self._rows


class FakeConn:
    """rows: content_hash -> source."""

    def __init__(self, store):
        self.store = store

    async def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        rows: dict[str, str] = self.store["rows"]
        if sql.startswith("SELECT content_hash"):
            return FakeCursor([{"content_hash": h} for h in rows])
        if sql.startswith("INSERT"):
            source, content_hash = params[0], params[4]
            self.store["inserts"].append(params)
            rows.setdefault(content_hash, source)
            return FakeCursor()
        if sql.startswith("DELETE FROM knowledge_base_chunks WHERE source = %s AND NOT"):
            source, keep = params
            gone = [h for h, s in rows.items() if s == source and h not in keep]
            for h in gone:
                del rows[h]
            return FakeCursor(rowcount=len(gone))
        if sql.startswith("DELETE FROM knowledge_base_chunks WHERE NOT (source = ANY"):
            (sources,) = params
            gone = [h for h, s in rows.items() if s not in sources]
            for h in gone:
                del rows[h]
            return FakeCursor(rowcount=len(gone))
        raise AssertionError(f"unexpected SQL: {sql}")

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self):
        self.store = {"rows": {}, "inserts": []}

    @asynccontextmanager
    async def connection(self):
        yield FakeConn(self.store)


def _chain(pool, monkeypatch):
    async def no_register(_conn):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", no_register)
    return indexer.build_index_chain(pool, LcEmbeddings(FakeEmbedder(dim=8)), "fake-model")


async def test_chain_indexes_then_skips_unchanged(monkeypatch):
    pool = FakePool()
    chain = _chain(pool, monkeypatch)

    first = await chain.ainvoke({"source": "kb.md", "text": DOC})
    assert first["added"] == first["total"] >= 3 and first["removed"] == 0
    row = pool.store["inserts"][0]
    assert row[0] == "kb.md" and len(row[5]) == 8 and row[6] == "fake-model"

    second = await chain.ainvoke({"source": "kb.md", "text": DOC})
    assert second["added"] == 0 and second["removed"] == 0  # idempotent
    assert len(pool.store["inserts"]) == first["added"]


async def test_chain_embeds_only_changed_chunks(monkeypatch):
    pool = FakePool()
    chain = _chain(pool, monkeypatch)
    await chain.ainvoke({"source": "kb.md", "text": DOC})
    changed = await chain.ainvoke({"source": "kb.md", "text": DOC.replace("Restart", "Reboot")})
    assert changed["added"] == 1


async def test_an_edited_document_loses_its_old_text(monkeypatch):
    """The bug this guards against: the assistant quoting text that was corrected or removed."""
    pool = FakePool()
    chain = _chain(pool, monkeypatch)
    await chain.ainvoke({"source": "kb.md", "text": DOC})
    edited = await chain.ainvoke({"source": "kb.md", "text": DOC.replace("Restart", "Reboot")})
    assert edited["added"] == 1 and edited["removed"] == 1  # new chunk in, old chunk out
    assert len(pool.store["rows"]) == 3  # still one row per chunk, none stale


async def test_removing_a_section_removes_its_chunk(monkeypatch):
    pool = FakePool()
    chain = _chain(pool, monkeypatch)
    await chain.ainvoke({"source": "kb.md", "text": DOC})
    shorter = DOC.split("## Slow data")[0]
    result = await chain.ainvoke({"source": "kb.md", "text": shorter})
    assert result["removed"] == 1 and result["added"] == 0


async def test_editing_one_document_never_touches_another(monkeypatch):
    pool = FakePool()
    chain = _chain(pool, monkeypatch)
    await chain.ainvoke({"source": "a.md", "text": DOC})
    await chain.ainvoke({"source": "b.md", "text": "# Other\n\nSomething else.\n"})
    before = {h for h, s in pool.store["rows"].items() if s == "b.md"}
    await chain.ainvoke({"source": "a.md", "text": "# Guide\n\nCompletely new.\n"})
    assert {h for h, s in pool.store["rows"].items() if s == "b.md"} == before
