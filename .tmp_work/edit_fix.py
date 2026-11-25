from lib import cut, save, sub

f = "src/support_assistant/registry.py"
sub(f, 'COLLECTION = "company_knowledge"  # the one knowledge collection today\n', 'COLLECTION: Final = "company_knowledge"  # the one knowledge collection today\n')
sub(f, "from typing import Any, Literal, Protocol, TypeVar\n", "from typing import Any, Final, Literal, Protocol, TypeVar\n")

sub(
    "src/support_assistant/evaluation.py",
    """            hits = await search_kb(
                pool, embedder, q, candidates=s.rerank_candidates, reranker=reranker
            )
""",
    """            hits = await search_kb(
                pool,
                embedder,
                q,
                candidates=s.rerank_candidates,
                reranker=reranker,
                embedding=(s.embedding_model, s.embedding_version),
            )
""",
)

# --- one fake of the chunk table that enforces UNIQUE (source, content_hash) ----------------------
f = "tests/test_indexer_chain.py"
cut(
    f,
    "class FakeConn:\n",
    "def _chain(pool, monkeypatch):",
    '''class FakeConn:
    """store["rows"]: (source, content_hash) -> row. The key IS the table's unique constraint."""

    def __init__(self, store):
        self.store = store

    async def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        rows: dict = self.store["rows"]
        if sql.startswith("SELECT content_hash FROM knowledge_base_chunks WHERE source"):
            source, model, version = params
            return FakeCursor(
                [
                    {"content_hash": h}
                    for (s, h), row in rows.items()
                    if s == source and (row["model"], row["version"]) == (model, version)
                ]
            )
        if sql.startswith("SELECT content_hash, embedding"):
            wanted, model, version = params
            return FakeCursor(
                [
                    {"content_hash": h, "embedding": row["embedding"]}
                    for (_, h), row in rows.items()
                    if h in wanted and (row["model"], row["version"]) == (model, version)
                ]
            )
        if sql.startswith("INSERT"):
            assert "ON CONFLICT (source, content_hash) DO UPDATE" in sql
            source, section_id, _, _, content_hash, embedding, model, version = params
            self.store["inserts"].append(params)
            rows[(source, content_hash)] = {
                "section_id": section_id,
                "embedding": embedding,
                "model": model,
                "version": version,
            }
            return FakeCursor()
        if sql.startswith("UPDATE knowledge_base_chunks SET section_id"):
            section_id, source, content_hash, _ = params
            rows[(source, content_hash)]["section_id"] = section_id
            return FakeCursor()
        if sql.startswith("DELETE FROM knowledge_base_chunks WHERE source = %s AND NOT"):
            source, keep = params
            return self._delete(lambda s, h: s == source and h not in keep)
        if sql.startswith("DELETE FROM knowledge_base_chunks WHERE NOT (source = ANY"):
            (sources,) = params
            return self._delete(lambda s, h: s not in sources)
        raise AssertionError(f"unexpected SQL: {sql}")

    def _delete(self, wanted):
        gone = [key for key in self.store["rows"] if wanted(*key)]
        for key in gone:
            del self.store["rows"][key]
        return FakeCursor(rowcount=len(gone))

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self):
        self.store = {"rows": {}, "inserts": []}

    @asynccontextmanager
    async def connection(self):
        yield FakeConn(self.store)


''',
)
sub(
    f,
    '    before = {h for h, s in pool.store["rows"].items() if s == "b.md"}\n',
    '    before = {h for s, h in pool.store["rows"] if s == "b.md"}\n',
)
sub(
    f,
    '    assert {h for h, s in pool.store["rows"].items() if s == "b.md"} == before\n',
    '    assert {h for s, h in pool.store["rows"] if s == "b.md"} == before\n',
)

f = "tests/test_documents_uploads.py"
sub(
    f,
    '    sources = lambda: {s for s in pool.store["rows"].values()}  # noqa: E731\n',
    '    sources = lambda: {s for s, _ in pool.store["rows"]}  # noqa: E731\n',
)

f = "tests/test_reliability.py"
cut(
    f,
    "async def test_a_new_embedding_model_embeds_every_chunk_again(monkeypatch):",
    "# --- correlation and leaks",
    '''async def test_a_new_embedding_model_embeds_every_chunk_again(monkeypatch):
    pool = KbPool()

    async def register(_):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", register)

    async def index(model: str) -> dict:
        chain = indexer.build_index_chain(pool, LcEmbeddings(FakeEmbedder(dim=8)), model)
        return await chain.ainvoke({"source": "kb.md", "text": "# G\\n\\nRestart.\\n"})

    first = await index("model-a")
    assert first["embedded"] >= 1 and (await index("model-a"))["embedded"] == 0  # unchanged
    again = await index("model-b")  # vectors of another model are never reused
    assert again["embedded"] == first["embedded"]
    assert {r["model"] for r in pool.store["rows"].values()} == {"model-b"}


''',
)
sub(f, "from test_security import STRONG, _setup\n", "from test_indexer_chain import FakePool as KbPool\nfrom test_security import STRONG, _setup\n")
save()
