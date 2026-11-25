"""Document hashing, the metadata registry, the shared-settings releases, and the Cosmos DB roles.

The chunk table is the fake from test_indexer_chain, whose key is the real unique constraint. The
Cosmos client is faked too: nothing here proves behaviour against a real Cosmos DB account.
"""

import re
from pathlib import Path
from typing import Any

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError
from fastapi import HTTPException

from support_assistant import indexer
from support_assistant.agents import service as agent_service
from support_assistant.config import Settings
from support_assistant.docsource import LoadedDocs
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.mcp_servers.main import require_release
from support_assistant.registry import (
    CosmosRegistry,
    DocumentRecord,
    InMemoryRegistry,
    build_registry,
    document_id,
)
from support_assistant.releases import (
    Release,
    ReleaseGate,
    publish,
    search_release,
    tools_release,
)
from support_assistant.retrieval_service import app as retrieval_service
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
    """One knowledge base: the chunk table, the registry and the embedder."""

    def __init__(self, monkeypatch) -> None:
        async def register(_):
            return None

        monkeypatch.setattr(indexer, "register_vector_async", register)
        self.pool, self.registry = FakePool(), InMemoryRegistry()
        self.embedder = CountingEmbedder()

    async def index(self, documents: dict[str, str], model="m", fingerprint="release-1"):
        return await indexer.index_source(
            self.pool,
            LcEmbeddings(self.embedder),
            Docs(documents),
            model,
            registry=self.registry,
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
    record = await kb.registry.get_document("apn.md")
    assert record.document_hash == indexer.document_hash(APN) and len(record.document_hash) == 64
    assert record.indexing_fingerprint == "release-1" and len(record.chunk_hashes) == 2
    assert record.size_bytes == len(APN.encode())


async def test_an_unchanged_document_is_not_even_split(kb, monkeypatch):
    await kb.index({"apn.md": APN})
    embedded, writes = kb.embedder.texts, kb.registry.writes

    def must_not_run(*a, **k):
        raise AssertionError("an unchanged document was split")

    monkeypatch.setattr(indexer, "chunk_markdown", must_not_run)
    report = await kb.index({"apn.md": APN})
    assert report.documents_unchanged == 1 and report.chunks_added == 0
    assert (kb.embedder.texts, kb.registry.writes) == (embedded, writes)


@pytest.mark.parametrize(
    ("text", "model", "fingerprint"),
    [
        (APN.replace("Reset", "Check"), "m", "release-1"),  # the text changed
        (APN, "m", "release-2"),  # the chunking or embedding settings changed
        (APN, "other-model", "release-2"),
    ],
)
async def test_a_change_of_text_or_settings_processes_the_document_again(
    kb, text, model, fingerprint
):
    await kb.index({"apn.md": APN})
    report = await kb.index({"apn.md": text}, model=model, fingerprint=fingerprint)
    assert report.documents_unchanged == 0
    record = await kb.registry.get_document("apn.md")
    assert record.indexing_fingerprint == fingerprint
    assert record.document_hash == indexer.document_hash(text)


async def test_a_matching_record_is_not_believed_when_the_database_disagrees(kb):
    """Cosmos and Postgres share no transaction: after a restore, the record can be ahead."""
    await kb.index({"apn.md": APN})
    lost = next(iter(kb.pool.store["rows"]))
    del kb.pool.store["rows"][lost]
    report = await kb.index({"apn.md": APN})
    assert report.documents_unchanged == 0 and report.chunks_added == 1
    assert kb.chunks("apn.md") == 2


async def test_deleting_a_document_removes_its_record_and_an_empty_listing_removes_nothing(kb):
    await kb.index({"apn.md": APN, "billing.md": BILLING})
    await kb.index({})
    assert await kb.registry.document_sources() == ["apn.md", "billing.md"]
    await kb.index({"billing.md": BILLING})
    assert await kb.registry.document_sources() == ["billing.md"] and kb.chunks("apn.md") == 0


# --- the bugs unique-per-document fixes -------------------------------------------------------------
async def test_a_renamed_document_keeps_every_chunk_and_embeds_nothing_again(kb):
    await kb.index({"apn.md": APN, "billing.md": BILLING})
    embedded = kb.embedder.texts
    report = await kb.index({"apn-guide.md": APN, "billing.md": BILLING})
    assert kb.chunks("apn-guide.md") == 2 and kb.chunks("apn.md") == 0
    assert kb.embedder.texts == embedded and report.chunks_embedded == 0
    assert await kb.registry.document_sources() == ["apn-guide.md", "billing.md"]


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


# --- registry outages ---------------------------------------------------------------------------
async def test_a_registry_outage_still_updates_search_and_the_next_run_repairs_the_records(kb):
    kb.registry.down = True
    report = await kb.index({"apn.md": APN, "billing.md": BILLING})
    assert report.registry_errors == 1  # left alone after the first failure, not once per file
    assert kb.chunks("apn.md") == 2 and kb.chunks("billing.md") == 1
    kb.registry.down = False
    repaired = await kb.index({"apn.md": APN, "billing.md": BILLING})
    assert repaired.registry_errors == 0 and repaired.chunks_embedded == 0
    assert await kb.registry.document_sources() == ["apn.md", "billing.md"]


async def test_a_registry_without_a_fingerprint_is_refused(kb):
    with pytest.raises(ValueError):
        await kb.index({"apn.md": APN}, fingerprint="")


# --- releases -------------------------------------------------------------------------------------
def test_a_release_version_changes_only_when_its_content_does():
    a, b = search_release(_settings()), search_release(_settings())
    assert a.version == b.version
    for change in ({"embedding_model": "x"}, {"embedding_version": "2"}, {"embedding_dim": 8}):
        assert search_release(_settings(**change)).version != a.version, change


def test_the_tool_release_lists_servers_write_tools_and_each_agents_grants():
    body = tools_release().body
    assert "process_payment" in body["servers"]["database"]["tools"]
    assert "process_payment" in body["write_tools"]
    assert "process_payment" not in body["agent_tools"]["director"]


async def test_publishing_twice_is_safe_and_old_and_new_versions_both_verify():
    registry = InMemoryRegistry()
    old, new = Release("search", {"model": "a"}), Release("search", {"model": "b"})
    await publish(registry, [old])
    await publish(registry, [old])
    await publish(registry, [new])  # a rolling upgrade: both sets of pods are running
    assert await ReleaseGate(registry, old).check() and await ReleaseGate(registry, new).check()


async def test_settings_nobody_published_or_a_tampered_release_keep_the_gate_closed():
    registry, release = InMemoryRegistry(), Release("tools", {"a": 1})
    assert not await ReleaseGate(registry, release).check()
    registry.releases[("tools", release.version)] = {"a": 2}
    assert not await ReleaseGate(registry, release).check()
    with pytest.raises(RuntimeError):
        await publish(registry, [release])


async def test_a_verified_pod_stays_ready_through_an_outage_and_a_new_one_does_not_become_ready():
    registry, release = InMemoryRegistry(), tools_release()
    await publish(registry, [release])
    running = ReleaseGate(registry, release)
    assert await running.check()
    registry.down = True
    assert await running.check()
    assert not await ReleaseGate(registry, release).check()


async def test_without_a_registry_the_gate_is_open_and_with_it_required_startup_fails():
    assert await ReleaseGate(None, tools_release()).check()
    assert build_registry(_settings()) is None
    with pytest.raises(RuntimeError, match="COSMOS_ENDPOINT"):
        build_registry(_settings(registry_required=True))


# --- each workload that must stay in sync -----------------------------------------------------------
async def test_the_mcp_servers_wait_for_the_release_then_refuse_to_serve():
    naps: list[float] = []

    async def sleep(s: float) -> None:
        naps.append(s)

    closed = ReleaseGate(InMemoryRegistry(), tools_release())
    with pytest.raises(RuntimeError, match="not verified"):
        await require_release(closed, attempts=3, sleep=sleep)
    assert len(naps) == 2 and all(n >= 1.0 for n in naps)


async def test_an_agent_is_not_ready_until_its_tool_release_is_published():
    class Mcp:
        async def verify(self, tools):
            return []

    registry = InMemoryRegistry()
    saved = dict(agent_service.app.state._state)
    try:
        agent_service.app.state.mcp = Mcp()
        agent_service.app.state.release_gate = ReleaseGate(registry, tools_release())
        with pytest.raises(HTTPException) as refused:
            await agent_service.readyz()
        assert refused.value.status_code == 503
        await publish(registry, [tools_release()])
        assert (await agent_service.readyz())["status"] == "ready"
    finally:
        agent_service.app.state._state.clear()
        agent_service.app.state._state.update(saved)


async def test_retrieval_is_not_ready_with_search_settings_nobody_published():
    class Conn:
        async def execute(self, *a):
            return None

    class Pool:
        def connection(self):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def cm():
                yield Conn()

            return cm()

    saved = dict(retrieval_service.app.state._state)
    try:
        retrieval_service.app.state.pool = Pool()
        retrieval_service.app.state.release_gate = ReleaseGate(
            InMemoryRegistry(), search_release(_settings())
        )
        with pytest.raises(HTTPException) as refused:
            await retrieval_service.readyz()
        assert refused.value.status_code == 503
    finally:
        retrieval_service.app.state._state.clear()
        retrieval_service.app.state._state.update(saved)


# --- the Cosmos adapter ---------------------------------------------------------------------------
class Container:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}
        self.error: Exception | None = None

    def _check(self):
        if self.error:
            raise self.error

    async def read_item(self, item, partition_key):
        self._check()
        if (partition_key, item) not in self.items:
            raise CosmosResourceNotFoundError(message="missing")
        return self.items[(partition_key, item)]

    async def upsert_item(self, body):
        self._check()
        self.items[(body["collection"], body["id"])] = body

    async def create_item(self, body):
        self._check()
        if (body["kind"], body["id"]) in self.items:
            raise CosmosResourceExistsError(message="exists")
        self.items[(body["kind"], body["id"])] = body

    async def delete_item(self, item, partition_key):
        self._check()
        if self.items.pop((partition_key, item), None) is None:
            raise CosmosResourceNotFoundError(message="missing")

    def query_items(self, query, partition_key):
        async def rows():
            self._check()
            for (pk, _), body in self.items.items():
                if pk == partition_key:
                    yield body["source"]

        return rows()


class Client:
    def __init__(self) -> None:
        self.containers = {"documents": Container(), "catalog": Container()}

    def get_database_client(self, name):
        assert name == "registry"
        client = self

        class Db:
            def get_container_client(self, container):
                return client.containers[container]

        return Db()

    async def close(self):
        return None


def _record(source: str) -> DocumentRecord:
    return DocumentRecord(
        source=source,
        document_hash="a" * 64,
        indexing_fingerprint="f",
        chunk_hashes=["h"],
        size_bytes=1,
        indexed_at="2026-01-01T00:00:00+00:00",
    )


async def test_cosmos_documents_round_trip_with_safe_ids_in_the_right_partition():
    client = Client()
    registry = CosmosRegistry("https://x", "registry", client=client)
    name = "policies/roaming?v=2#a.md"  # characters a Cosmos id may not hold
    await registry.put_document(_record(name))
    ((partition, item),) = client.containers["documents"].items
    assert partition == "company_knowledge" and item == document_id(name)
    assert not set("/\\?#") & set(item)
    assert (await registry.get_document(name)).source == name
    assert await registry.document_sources() == [name]
    await registry.delete_document(name)
    await registry.delete_document(name)  # already gone: not an error
    assert await registry.get_document(name) is None


async def test_cosmos_releases_are_create_only():
    registry = CosmosRegistry("https://x", "registry", client=Client())
    assert await registry.create_release("tools", "v1", {"a": 1}) is True
    assert await registry.create_release("tools", "v1", {"a": 2}) is False
    assert await registry.get_release("tools", "v1") == {"a": 1}
    assert await registry.get_release("tools", "v2") is None


async def test_any_cosmos_failure_is_one_unavailable_error_and_a_bad_record_is_ignored():
    client = Client()
    registry = CosmosRegistry("https://x", "registry", client=client)
    documents = client.containers["documents"]
    documents.items[("company_knowledge", document_id("a.md"))] = {"source": "a.md"}
    assert await registry.get_document("a.md") is None
    documents.error = PermissionError("403: no role assignment")
    for call in (
        registry.get_document("a.md"),
        registry.put_document(_record("a.md")),
        registry.delete_document("a.md"),
        registry.document_sources(),
    ):
        with pytest.raises(ExternalServiceUnavailable):
            await call


# --- the roles in Terraform ------------------------------------------------------------------------
def _terraform(name: str) -> str:
    return (ROOT / "infra" / "terraform" / name).read_text(encoding="utf-8")


def _access() -> dict[str, dict[str, str]]:
    block = _terraform("cosmos.tf").split("registry_access = {")[1].split("\n  }\n")[0]
    found = {}
    for workload, grants in re.findall(r'"([a-z-]+)"\s*=\s*\{([^}]*)\}', block):
        found[workload] = dict(re.findall(r'(\w+)\s*=\s*"(\w+)"', grants))
    return found


def test_only_the_indexers_write_documents_and_only_the_deploy_hook_publishes():
    access = _access()
    writers = {w for w, g in access.items() if g.get("documents") == "writer"}
    publishers = {w for w, g in access.items() if g.get("catalog") == "publisher"}
    assert writers == {"indexer", "index-worker"} and publishers == {"db-bootstrap"}
    assert {w for w, g in access.items() if "documents" in g} == writers
    assert not {"fat-backend", "orchestrator", "notification-worker"} & set(access)


def test_every_workload_that_checks_or_writes_has_a_role_and_a_real_identity():
    assert set(_access()) == {
        "indexer",
        "index-worker",
        "retrieval",
        "mcp-servers",
        "agent-director",
        "agent-troubleshooter",
        "agent-evaluator",
        "db-bootstrap",
    }
    for workload in _access():
        assert f'"{workload}"' in _terraform("aks.tf"), workload


def test_no_role_can_change_or_delete_a_published_release():
    text = _terraform("cosmos.tf")
    publisher = text.split("publisher = concat(")[1].split("])")[0]
    assert "items/create" in publisher
    assert not [v for v in ("replace", "upsert", "delete", "*") if f"items/{v}" in publisher]


def test_the_account_has_no_keys_and_no_public_endpoint():
    text = _terraform("cosmos.tf")
    for setting, value in (
        ("local_authentication_enabled", "false"),
        ("public_network_access_enabled", "false"),
    ):
        assert re.search(rf"{setting}\s*=\s*{value}", text), setting
    assert "privatelink.documents.azure.com" in _terraform("network.tf")


def test_the_deploy_hook_publishes_and_production_requires_the_registry():
    helm = ROOT / "deploy" / "helm" / "support"
    hook = (helm / "templates" / "db-bootstrap.yaml").read_text(encoding="utf-8")
    assert "python -m support_assistant.releases publish" in hook
    assert ".Values.endpoints" in hook  # the hook needs COSMOS_ENDPOINT too
    assert 'REGISTRY_REQUIRED: "true"' in (helm / "values.yaml").read_text(encoding="utf-8")
