from lib import cut, save, sub

P = "src/support_assistant/"
f = P + "indexer.py"

sub(
    f,
    '''Idempotent: chunks are keyed by sha256(heading path + content), so re-running only embeds new or
changed text. We keep our own table (hash, model/version columns, tsvector for hybrid search), so
the write step is plain SQL rather than a LangChain vector-store class.
''',
    '''Two levels of hashing make re-running cheap and safe:
  document  SHA-256 of the file. With the metadata registry, an unchanged document indexed with
            unchanged settings is skipped without being split. The record is only believed if
            Postgres holds exactly the chunks it lists.
  chunk     SHA-256 of heading path + content, unique PER DOCUMENT. Only new text is embedded, and
            identical text already embedded with the same model (a renamed or copied document)
            reuses that vector.
We keep our own table (hash, model/version columns, tsvector for hybrid search), so the write step
is plain SQL rather than a LangChain vector-store class.
''',
)
sub(
    f,
    '''MAX_CHARS = 1500
''',
    '''MAX_CHARS = 1500
# Change this whenever splitting or hashing changes, so every document is processed again.
CHUNKER_VERSION = "markdown-headings-1"
''',
)

cut(
    f,
    "def build_index_chain(",
    "INDEX_LOCK_KEY = 7_302_001",
    '''def document_hash(text: str) -> str:
    """SHA-256 of the file's bytes. Sources only yield valid UTF-8, so encoding gives them back."""
    return hashlib.sha256(text.encode()).hexdigest()


async def _chunk_hashes_in_db(pool: Pool, source: str, model: str, version: str) -> set[str]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT content_hash FROM knowledge_base_chunks"
            " WHERE source = %s AND embedding_model = %s AND embedding_version = %s",
            (source, model, version),
        )
        return {r["content_hash"] for r in await cur.fetchall()}


def build_index_chain(
    pool: Pool, embeddings: LcEmbeddings, model: str, version: str = "1"
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """The indexing pipeline for one document. Input: {"source": str, "text": str}."""

    async def split(inp: dict[str, Any]) -> dict[str, Any]:
        distinct: dict[str, RawChunk] = {}
        for chunk in chunk_markdown(inp["text"]):
            distinct.setdefault(chunk.digest, chunk)  # a repeat inside one document is one chunk
        return {**inp, "chunks": list(distinct.values())}

    async def find_known(inp: dict[str, Any]) -> dict[str, Any]:
        # "Already indexed" means this document, this text AND this embedding model: vectors from
        # two models cannot be compared, so a model change must embed every chunk again.
        known = await _chunk_hashes_in_db(pool, inp["source"], model, version)
        fresh = [c for c in inp["chunks"] if c.digest not in known]
        reusable: dict[str, Any] = {}
        if fresh:
            async with pool.connection() as conn:
                await register_vector_async(conn)
                cur = await conn.execute(
                    "SELECT content_hash, embedding FROM knowledge_base_chunks"
                    " WHERE content_hash = ANY(%s)"
                    " AND embedding_model = %s AND embedding_version = %s",
                    ([c.digest for c in fresh], model, version),
                )
                reusable = {r["content_hash"]: r["embedding"] for r in await cur.fetchall()}
        return {**inp, "fresh": fresh, "reusable": reusable}

    async def embed(inp: dict[str, Any]) -> dict[str, Any]:
        missing: list[RawChunk] = [c for c in inp["fresh"] if c.digest not in inp["reusable"]]
        docs = [Document(page_content=c.embed_text) for c in missing]
        vectors = await embeddings.aembed_documents([d.page_content for d in docs]) if docs else []
        made = {c.digest: v for c, v in zip(missing, vectors, strict=True)}
        return {**inp, "vectors": {**inp["reusable"], **made}, "embedded": len(made)}

    async def upsert(inp: dict[str, Any]) -> dict[str, Any]:
        fresh: list[RawChunk] = inp["fresh"]
        written = {c.digest for c in fresh}
        keep = [c.digest for c in inp["chunks"]]
        async with pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.transaction():
                for c in fresh:
                    await conn.execute(
                        """
                        INSERT INTO knowledge_base_chunks
                          (source, section_id, heading_path, content, content_hash,
                           embedding, embedding_model, embedding_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source, content_hash) DO UPDATE
                           SET section_id = EXCLUDED.section_id,
                               embedding = EXCLUDED.embedding,
                               embedding_model = EXCLUDED.embedding_model,
                               embedding_version = EXCLUDED.embedding_version
                        """,
                        (
                            inp["source"],
                            c.section_id,
                            c.heading_path,
                            c.content,
                            c.digest,
                            inp["vectors"][c.digest],
                            model,
                            version,
                        ),
                    )
                # Unchanged text can still have moved: keep its position number current.
                for c in inp["chunks"]:
                    if c.digest not in written:
                        await conn.execute(
                            "UPDATE knowledge_base_chunks SET section_id = %s"
                            " WHERE source = %s AND content_hash = %s AND section_id <> %s",
                            (c.section_id, inp["source"], c.digest, c.section_id),
                        )
                # Mirror the document: drop this source's chunks that are no longer in its text.
                cur = await conn.execute(
                    "DELETE FROM knowledge_base_chunks WHERE source = %s AND NOT (content_hash = ANY(%s))",
                    (inp["source"], keep),
                )
                removed = cur.rowcount
        return {
            "added": len(fresh),
            "embedded": inp["embedded"],
            "total": len(keep),
            "removed": removed,
            "chunk_hashes": sorted(keep),
        }

    return (
        RunnableLambda(split)
        | RunnableLambda(find_known)
        | RunnableLambda(embed)
        | RunnableLambda(upsert)
    )


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks_added: int
    chunks_removed: int
    sources_removed: int
    skipped: dict[str, str]
    documents_unchanged: int = 0
    chunks_embedded: int = 0
    # Registry calls that failed. Search is still up to date; the records are not, so the run
    # should be repeated.
    registry_errors: int = 0


class _Records:
    """The registry as the indexer uses it: after one failure it is left alone for the rest of the
    run, so an outage costs one timeout and never stops the knowledge base from updating."""

    def __init__(self, registry: Registry | None) -> None:
        self._registry = registry
        self.errors = 0

    async def run(self, operation: Callable[[Registry], Awaitable[T]]) -> T | None:
        if self._registry is None or self.errors:
            return None
        try:
            return await operation(self._registry)
        except ExternalServiceUnavailable as exc:
            self.errors += 1
            log.error("document_registry_unavailable", **error_fields(exc))
            return None


async def index_source(
    pool: Pool,
    embeddings: LcEmbeddings,
    source: DocumentSource,
    model: str,
    version: str = "1",
    registry: Registry | None = None,
    fingerprint: str = "",
) -> IndexReport:
    """Index every document in `source` and make the knowledge base match it.

    Safety: if the source yields NO documents, nothing is deleted. An empty or failed listing is
    far more likely a storage problem than an intent to erase the whole knowledge base.

    Order matters: documents are written BEFORE vanished ones are removed, so a renamed document
    is searchable throughout. Postgres is written before the registry: if the run stops between
    the two, the stale record no longer matches and the document is simply processed again.
    """
    if registry is not None and not fingerprint:
        raise ValueError("a registry needs the indexing fingerprint (releases.search_release)")
    loaded = await source.load()
    chain = build_index_chain(pool, embeddings, model, version)
    records = _Records(registry)
    added = removed = embedded = unchanged = 0
    for name in loaded.names:
        text = loaded.documents[name]
        digest = document_hash(text)
        record = await records.run(lambda r, name=name: r.get_document(name))  # type: ignore[misc]
        if (
            record is not None
            and record.document_hash == digest
            and record.indexing_fingerprint == fingerprint
            # Never trust the record alone: the database may have been restored or edited.
            and await _chunk_hashes_in_db(pool, name, model, version) == set(record.chunk_hashes)
        ):
            unchanged += 1
            continue
        result = await chain.ainvoke({"source": name, "text": text})
        added += result["added"]
        removed += result["removed"]
        embedded += result["embedded"]
        if registry is not None:
            fresh_record = DocumentRecord(
                source=name,
                document_hash=digest,
                indexing_fingerprint=fingerprint,
                chunk_hashes=result["chunk_hashes"],
                size_bytes=len(text.encode()),
                indexed_at=datetime.now(UTC).isoformat(),
            )
            await records.run(lambda r, rec=fresh_record: r.put_document(rec))  # type: ignore[misc]

    sources_removed = 0
    if loaded.names:  # never prune on an empty listing
        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "DELETE FROM knowledge_base_chunks WHERE NOT (source = ANY(%s))",
                    (loaded.names,),
                )
                sources_removed = cur.rowcount
        listed = set(loaded.names)
        recorded = await records.run(lambda r: r.document_sources()) or []
        for gone in (s for s in recorded if s not in listed):
            await records.run(lambda r, gone=gone: r.delete_document(gone))  # type: ignore[misc]
    log.info(
        "knowledge_documents_indexed",
        documents=len(loaded.names),
        unchanged=unchanged,
        chunks_added=added,
        chunks_embedded=embedded,
        chunks_removed=removed,
        registry_errors=records.errors,
    )
    return IndexReport(
        len(loaded.names),
        added,
        removed,
        sources_removed,
        loaded.skipped,
        unchanged,
        embedded,
        records.errors,
    )


''',
)

sub(
    f,
    """    pool = make_pool(s)
    await pool.open()
    try:
        async with exclusive_index_run(pool) as got:
            if not got:
                print("another indexing run is active; nothing to do")
                return 0
            report = await index_source(pool, LcEmbeddings(embedder), source, s.embedding_model)
        print(
            f"{report.documents} documents: +{report.chunks_added} chunks, "
            f"-{report.chunks_removed} stale, -{report.sources_removed} from deleted documents"
        )
        for name, why in report.skipped.items():
            print(f"skipped {name}: {why}")
        return 0
    finally:
        await embedder.aclose()
        await pool.close()
""",
    """    registry = build_registry(s)
    release = search_release(s)
    pool = make_pool(s)
    await pool.open()
    try:
        # Indexing with settings nobody deployed could mix incompatible vectors into search.
        if not await ReleaseGate(registry, release).check():
            print("the search settings of this release are not verified; not indexing")
            return 1
        async with exclusive_index_run(pool) as got:
            if not got:
                print("another indexing run is active; nothing to do")
                return 0
            report = await index_source(
                pool,
                LcEmbeddings(embedder),
                source,
                s.embedding_model,
                s.embedding_version,
                registry=registry,
                fingerprint=release.version,
            )
        print(
            f"{report.documents} documents ({report.documents_unchanged} unchanged): "
            f"+{report.chunks_added} chunks, -{report.chunks_removed} stale, "
            f"-{report.sources_removed} from deleted documents"
        )
        for name, why in report.skipped.items():
            print(f"skipped {name}: {why}")
        if report.registry_errors:
            print("document records could not be updated; run again", file=sys.stderr)
            return 1
        return 0
    finally:
        await embedder.aclose()
        await pool.close()
        if registry is not None:
            await registry.aclose()
""",
)
sub(
    f,
    "from collections.abc import AsyncIterator\n",
    "from collections.abc import AsyncIterator, Awaitable, Callable\n",
)
sub(f, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nfrom datetime import UTC, datetime\n")
sub(f, "from typing import Any\n", "from typing import Any, TypeVar\n")
sub(f, "from langchain_core.documents import Document\n", "import structlog\nfrom langchain_core.documents import Document\n")
sub(
    f,
    "from support_assistant.embeddings_lc import LcEmbeddings\n",
    """from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields
from support_assistant.registry import DocumentRecord, Registry, build_registry
from support_assistant.releases import ReleaseGate, search_release

log = structlog.get_logger()
T = TypeVar("T")
""",
)

# releases.py imports the chunking constants lazily, so there is no import cycle

# --- index worker --------------------------------------------------------------------------------
f = P + "index_worker.py"
sub(
    f,
    """    pool = make_pool(s)
    await pool.open()

    async def reindex() -> bool:
        async with exclusive_index_run(pool) as got:
            if not got:
                return False
            report = await index_source(
                pool, LcEmbeddings(embedder), StoreDocumentSource(store), s.embedding_model
            )
""",
    """    registry = build_registry(s)
    release = search_release(s)
    gate = ReleaseGate(registry, release)
    pool = make_pool(s)
    await pool.open()

    async def reindex() -> bool:
        if not await gate.check():
            return False  # settings not verified: look again at the next tick
        async with exclusive_index_run(pool) as got:
            if not got:
                return False
            report = await index_source(
                pool,
                LcEmbeddings(embedder),
                StoreDocumentSource(store),
                s.embedding_model,
                s.embedding_version,
                registry=registry,
                fingerprint=release.version,
            )
""",
)
sub(
    f,
    """            skipped=len(report.skipped),
        )
        return True
""",
    """            skipped=len(report.skipped),
        )
        # Records that could not be written are repaired by running again at the next tick.
        return report.registry_errors == 0
""",
)
sub(f, "from support_assistant.config import get_settings\n", "from support_assistant.config import get_settings\nfrom support_assistant.registry import build_registry\nfrom support_assistant.releases import ReleaseGate, search_release\n")

# --- search: only compare vectors from the same model; drop duplicate text -------------------------
f = P + "services/retrieval.py"
sub(
    f,
    """    min_similarity: float | None = None,
) -> list[Chunk]:""",
    """    min_similarity: float | None = None,
    embedding: tuple[str, str] | None = None,
) -> list[Chunk]:""",
)
sub(
    f,
    '''    None means no floor."""''',
    '''    None means no floor.

    `embedding` is the (model, version) that embedded the query. Vector matches are then limited to
    chunks embedded the same way: distances between two models' vectors mean nothing. Keyword
    matches do not depend on the model, so they keep working while documents are re-embedded."""''',
)
sub(
    f,
    '''                SELECT id, section_id, heading_path, content, content_hash,
                       embedding <=> %s::vector AS distance
                  FROM knowledge_base_chunks
                 ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (vec, vec, candidates),
''',
    '''                SELECT id, section_id, heading_path, content, content_hash,
                       embedding <=> %s::vector AS distance
                  FROM knowledge_base_chunks
                 WHERE (%s::text IS NULL OR embedding_model = %s)
                   AND (%s::text IS NULL OR embedding_version = %s)
                 ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (vec, model, model, version, version, vec, candidates),
''',
)
sub(f, "    (vec,) = await embedder.embed([query])\n", "    (vec,) = await embedder.embed([query])\n    model, version = embedding or (None, None)\n")
sub(
    f,
    "    ranked = sorted(fused, key=fused.__getitem__, reverse=True)\n",
    "    ranked = _distinct_text(sorted(fused, key=fused.__getitem__, reverse=True), by_id)\n",
)
sub(
    f,
    "def _chunk(row: dict[str, Any], score: float) -> Chunk:",
    '''def _distinct_text(ranked: list[int], by_id: dict[int, Any]) -> list[int]:
    """Two documents may now hold the same chunk: return that text once, at its best rank."""
    seen: set[str] = set()
    kept: list[int] = []
    for i in ranked:
        key = str(by_id[i].get("content_hash") or f"id-{i}")
        if key not in seen:
            seen.add(key)
            kept.append(i)
    return kept


def _chunk(row: dict[str, Any], score: float) -> Chunk:''',
)

f = P + "retrieval_service/app.py"
sub(
    f,
    "                min_similarity=s.retrieval_min_similarity,\n",
    "                min_similarity=s.retrieval_min_similarity,\n                embedding=(s.embedding_model, s.embedding_version),\n",
)
sub(
    f,
    """        log.error("not_ready", dependency="postgres", **error_fields(exc))
        raise HTTPException(503, detail="knowledge store unavailable") from None
""",
    """        log.error("not_ready", dependency="postgres", **error_fields(exc))
        raise HTTPException(503, detail="knowledge store unavailable") from None
    # A pod whose embedding settings were never published would search with the wrong vectors.
    gate: ReleaseGate | None = getattr(app.state, "release_gate", None)
    if gate is not None and not await gate.check():
        raise HTTPException(503, detail="search settings not verified")
""",
)
sub(
    f,
    "    app.state.settings = settings\n    yield\n",
    """    app.state.settings = settings
    registry = build_registry(settings)
    app.state.release_gate = ReleaseGate(registry, search_release(settings))
    yield
    if registry is not None:
        await registry.aclose()
""",
)
sub(
    f,
    "from support_assistant.observability import RequestContextMiddleware\n",
    "from support_assistant.observability import RequestContextMiddleware\nfrom support_assistant.registry import build_registry\nfrom support_assistant.releases import ReleaseGate, search_release\n",
)

# --- agents: ready only with the published tool release --------------------------------------------
f = P + "agents/service.py"
sub(
    f,
    """    yield
    await mcp.aclose()
""",
    """    metadata = build_registry(settings)
    app.state.release_gate = ReleaseGate(metadata, tools_release())
    yield
    if metadata is not None:
        await metadata.aclose()
    await mcp.aclose()
""",
)
sub(
    f,
    """        raise HTTPException(503, detail=f"MCP tools unavailable: {', '.join(missing)}")
""",
    """        raise HTTPException(503, detail=f"MCP tools unavailable: {', '.join(missing)}")
    gate: ReleaseGate | None = getattr(app.state, "release_gate", None)
    if gate is not None and not await gate.check():
        raise HTTPException(503, detail="tool release not verified")
""",
)
sub(
    f,
    "from support_assistant.observability import RequestContextMiddleware\n",
    "from support_assistant.observability import RequestContextMiddleware\nfrom support_assistant.registry import build_registry\nfrom support_assistant.releases import ReleaseGate, tools_release\n",
)

# --- MCP servers have no readiness probe, so they check before serving ---------------------------
f = P + "mcp_servers/main.py"
sub(
    f,
    """async def _serve(server: FastMCP, port: int) -> None:""",
    '''async def require_release(
    gate: ReleaseGate,
    attempts: int = 6,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Do not serve tools until this image's tool release is verified. Gives up after a few
    tries, so the pod restarts and the failed rollout is visible."""
    for attempt in range(1, attempts + 1):
        if await gate.check():
            return
        if attempt < attempts:
            await sleep(backoff_delay(attempt, 2.0, 20.0) + 1.0)
    raise RuntimeError("the tool release of this image is not verified in the registry")


async def _serve(server: FastMCP, port: int) -> None:''',
)
sub(
    f,
    """    setup_telemetry("mcp-servers")
""",
    """    setup_telemetry("mcp-servers")
    registry = build_registry(settings)
    try:
        await require_release(ReleaseGate(registry, tools_release()))
    finally:
        if registry is not None:
            await registry.aclose()
""",
)
sub(
    f,
    "from support_assistant.telemetry import instrument_asgi, setup_telemetry\n",
    "from support_assistant.registry import build_registry\nfrom support_assistant.releases import ReleaseGate, tools_release\nfrom support_assistant.resilience import backoff_delay\nfrom support_assistant.telemetry import instrument_asgi, setup_telemetry\n",
)

# --- settings -------------------------------------------------------------------------------------
f = P + "config.py"
sub(
    f,
    "    embedding_dim: int = 1024\n",
    """    embedding_dim: int = 1024
    # Raise this to force every document to be embedded again with the same model name.
    embedding_version: str = "1"
    # The metadata registry (registry.py), Azure Cosmos DB with Entra auth only. Unset = no
    # registry: nothing is checked or recorded. Production sets REGISTRY_REQUIRED so that a
    # missing endpoint stops a service instead of silently skipping the checks.
    cosmos_endpoint: str | None = None
    cosmos_database: str = "registry"
    registry_required: bool = False
""",
)

f = P + "evaluation.py"
sub(
    f,
    "            for c in await search_kb(pool, embedder, q, candidates=s.rerank_candidates)\n",
    "            for c in await search_kb(\n                pool,\n                embedder,\n                q,\n                candidates=s.rerank_candidates,\n                embedding=(s.embedding_model, s.embedding_version),\n            )\n",
)
save()
