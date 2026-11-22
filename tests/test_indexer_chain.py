"""The LCEL indexing chain end to end with a fake pool and the fake embedder."""

from contextlib import asynccontextmanager

from support_assistant import indexer
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings

DOC = "# Guide\n\nIntro.\n\n## No signal\n\nRestart the phone.\n\n## Slow data\n\nCheck usage.\n"


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, store):
        self.store = store

    async def execute(self, sql, params=None):
        if sql.lstrip().startswith("SELECT content_hash"):
            return FakeCursor([{"content_hash": h} for h in self.store["hashes"]])
        self.store["inserts"].append(params)
        self.store["hashes"].add(params[4])
        return FakeCursor([])

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self):
        self.store = {"hashes": set(), "inserts": []}

    @asynccontextmanager
    async def connection(self):
        yield FakeConn(self.store)


async def test_chain_indexes_then_skips_unchanged(monkeypatch):
    async def no_register(_conn):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", no_register)
    pool = FakePool()
    chain = indexer.build_index_chain(pool, LcEmbeddings(FakeEmbedder(dim=8)), "fake-model")

    first = await chain.ainvoke({"source": "kb.md", "text": DOC})
    assert first["added"] == first["total"] >= 3
    row = pool.store["inserts"][0]
    assert row[0] == "kb.md" and len(row[5]) == 8 and row[6] == "fake-model"

    second = await chain.ainvoke({"source": "kb.md", "text": DOC})
    assert second["added"] == 0  # idempotent: nothing new to embed or write
    assert len(pool.store["inserts"]) == first["added"]


async def test_chain_embeds_only_changed_chunks(monkeypatch):
    async def no_register(_conn):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", no_register)
    pool = FakePool()
    chain = indexer.build_index_chain(pool, LcEmbeddings(FakeEmbedder(dim=8)), "m")
    await chain.ainvoke({"source": "kb.md", "text": DOC})
    changed = await chain.ainvoke({"source": "kb.md", "text": DOC.replace("Restart", "Reboot")})
    assert changed["added"] == 1
