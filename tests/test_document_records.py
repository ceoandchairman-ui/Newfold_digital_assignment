"""Document hashing: unchanged documents are skipped, using the records kept beside the chunks.

The two tables are the fake from test_indexer_chain, whose chunk key is the real unique
constraint. Nothing here proves behaviour against a real Postgres.
"""

import re
from pathlib import Path
from typing import Any

import pytest

from support_assistant import indexer
from support_assistant.config import Settings
from support_assistant.dbroles import GRANTS
from support_assistant.docsource import LoadedDocs
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from test_indexer_chain import FakePool

ROOT = Path(__file__).resolve().parents[1]
APN = "# Guide\n\nIntro.\n\n## APN\n\nReset the APN.\n"
BILLING = "# Billing\n\nLate fees apply.\n"


def _settings(**extra: Any) -> Settings:
    return Settings(database_url="postgresql://x", redis_url="redis://x", **extra)


class Docs:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents

    async def load(self) -> LoadedDocs:
        return LoadedDocs(documents=dict(self.documents))


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=8)
        self.texts = 0

    async def embed(self, texts):
        self.texts += len(texts)
        return await super().embed(texts)


class Kb:
    """One knowledge base: the chunk and document tables, and the embedder."""

    def __init__(self, monkeypatch) -> None:
        async def register(_):
            return None

        monkeypatch.setattr(indexer, "register_vector_async", register)
        self.pool = FakePool()
        self.embedder = CountingEmbedder()

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return self.pool.store["documents"]

    async def index(self, documents: dict[str, str], model="m", fingerprint="settings-1"):
        return await indexer.index_source(
            self.pool,
            LcEmbeddings(self.embedder),
            Docs(documents),
            model,
            fingerprint=fingerprint,
        )

    def chunks(self, source: str) -> int:
        return sum(s == source for s, _ in self.pool.store["rows"])


@pytest.fixture
def kb(monkeypatch) -> Kb:
    return Kb(monkeypatch)


# --- document hashing ---------------------------------------------------------------------------
async def test_each_document_is_recorded_with_its_hash_and_chunks(kb):
    await kb.index({"apn.md": APN})
    record = kb.records["apn.md"]
    assert record["document_hash"] == indexer.document_hash(APN)
    assert record["indexing_fingerprint"] == "settings-1" and len(record["chunk_hashes"]) == 2
    assert record["size_bytes"] == len(APN.encode())


async def test_an_unchanged_document_is_not_even_split(kb, monkeypatch):
    await kb.index({"apn.md": APN})
    embedded, writes = kb.embedder.texts, kb.pool.store["record_writes"]

    def must_not_run(*a, **k):
        raise AssertionError("an unchanged document was split")

    monkeypatch.setattr(indexer, "chunk_markdown", must_not_run)
    report = await kb.index({"apn.md": APN})
    assert report.documents_unchanged == 1 and report.chunks_added == 0
    assert (kb.embedder.texts, kb.pool.store["record_writes"]) == (embedded, writes)


@pytest.mark.parametrize(
    ("text", "model", "fingerprint"),
    [
        (APN.replace("Reset", "Check"), "m", "settings-1"),  # the text changed
        (APN, "m", "settings-2"),  # the chunking or embedding settings changed
        (APN, "other-model", "settings-2"),
    ],
)
async def test_a_change_of_text_or_settings_processes_the_document_again(
    kb, text, model, fingerprint
):
    await kb.index({"apn.md": APN})
    report = await kb.index({"apn.md": text}, model=model, fingerprint=fingerprint)
    assert report.documents_unchanged == 0
    record = kb.records["apn.md"]
    assert record["indexing_fingerprint"] == fingerprint
    assert record["document_hash"] == indexer.document_hash(text)


async def test_a_matching_record_is_not_believed_when_the_database_disagrees(kb):
    """Chunks may have been restored or edited since the record was written."""
    await kb.index({"apn.md": APN})
    lost = next(iter(kb.pool.store["rows"]))
    del kb.pool.store["rows"][lost]
    report = await kb.index({"apn.md": APN})
    assert report.documents_unchanged == 0 and report.chunks_added == 1
    assert kb.chunks("apn.md") == 2


async def test_deleting_a_document_removes_its_record_and_an_empty_listing_removes_nothing(kb):
    await kb.index({"apn.md": APN, "billing.md": BILLING})
    await kb.index({})
    assert sorted(kb.records) == ["apn.md", "billing.md"]
    await kb.index({"billing.md": BILLING})
    assert sorted(kb.records) == ["billing.md"] and kb.chunks("apn.md") == 0


# --- the bugs unique-per-document fixes -------------------------------------------------------------
async def test_a_renamed_document_keeps_every_chunk_and_embeds_nothing_again(kb):
    await kb.index({"apn.md": APN, "billing.md": BILLING})
    embedded = kb.embedder.texts
    report = await kb.index({"apn-guide.md": APN, "billing.md": BILLING})
    assert kb.chunks("apn-guide.md") == 2 and kb.chunks("apn.md") == 0
    assert kb.embedder.texts == embedded and report.chunks_embedded == 0
    assert sorted(kb.records) == ["apn-guide.md", "billing.md"]


@pytest.mark.parametrize(("owner", "other"), [("a.md", "z.md"), ("z.md", "a.md")])
async def test_two_documents_can_share_text_and_one_changing_never_costs_the_other(
    kb, owner, other
):
    await kb.index({owner: APN})
    await kb.index({owner: APN, other: APN})
    assert kb.chunks(owner) == kb.chunks(other) == 2
    await kb.index({owner: BILLING, other: APN})  # whichever is processed first
    assert kb.chunks(other) == 2


async def test_a_moved_chunk_gets_its_new_position_without_being_embedded_again(kb):
    await kb.index({"apn.md": APN})
    await kb.index({"apn.md": APN.replace("## APN", "## New\n\nText.\n\n## APN")})
    positions = sorted(r["section_id"] for r in kb.pool.store["rows"].values())
    assert positions == ["0", "1", "2"]  # no stale or repeated numbers
    assert kb.embedder.texts == 3


async def test_text_repeated_inside_one_document_is_stored_once(kb):
    await kb.index({"d.md": "# A\n\nSame.\n\n# B\n\nOther.\n\n# A\n\nSame.\n"})
    assert kb.chunks("d.md") == 2


# --- settings and access ------------------------------------------------------------------------
def test_the_fingerprint_changes_only_when_stored_text_would_differ():
    base = indexer.indexing_fingerprint(_settings())
    assert base == indexer.indexing_fingerprint(_settings())
    for change in ({"embedding_model": "x"}, {"embedding_version": "2"}, {"embedding_dim": 8}):
        assert indexer.indexing_fingerprint(_settings(**change)) != base, change


def test_only_the_indexers_may_use_the_document_records():
    holders = {w for w, g in GRANTS.items() if any("knowledge_documents" in t for _, t in g)}
    assert holders == {"indexer", "index-worker"}
    migration = (ROOT / "migrations" / "versions" / "0008_knowledge_documents.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r"CREATE TABLE IF NOT EXISTS knowledge_documents", migration)
