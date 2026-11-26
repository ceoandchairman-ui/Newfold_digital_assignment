"""Knowledge-base indexer (E2-2, E2-3) as a LangChain (LCEL) chain.

    markdown text -> split (MarkdownHeader + Recursive splitters) -> drop known chunks
                  -> embed (LangChain Embeddings) -> upsert into pgvector

Two levels of hashing make re-running cheap and safe:
  document  SHA-256 of the file. With the metadata registry, an unchanged document indexed with
            unchanged settings is skipped without being split. The record is only believed if
            Postgres holds exactly the chunks it lists.
  chunk     SHA-256 of heading path + content, unique PER DOCUMENT. Only new text is embedded, and
            identical text already embedded with the same model (a renamed or copied document)
            reuses that vector.
We keep our own table (hash, model/version columns, tsvector for hybrid search), so the write step
is plain SQL rather than a LangChain vector-store class.

The knowledge base MIRRORS the source: when a document is edited, its old chunks are removed; when a
document is deleted from the source, all its chunks are removed. Otherwise the assistant would keep
quoting text that was corrected or withdrawn. A safety guard stops a failed or empty listing from
wiping everything (see `index_source`).

Run as a Kubernetes CronJob (reads the `knowledge-base` blob container):
    python -m support_assistant.indexer                # DOCS_STORAGE_ACCOUNT + DOCS_CONTAINER
    python -m support_assistant.indexer ./some/folder  # a local folder or file (dev, job images)
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import structlog
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector_async

from support_assistant.config import get_settings
from support_assistant.db import Pool, make_pool
from support_assistant.docsource import DocumentSource, LocalDirSource, StoreDocumentSource
from support_assistant.embeddings import azure_embedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields
from support_assistant.registry import DocumentRecord, Registry, build_registry
from support_assistant.releases import ReleaseGate, search_release

log = structlog.get_logger()
T = TypeVar("T")

MAX_CHARS = 1500
# Change this whenever splitting or hashing changes, so every document is processed again.
CHUNKER_VERSION = "markdown-headings-1"
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


@dataclass(frozen=True)
class RawChunk:
    section_id: str
    heading_path: str
    content: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(f"{self.heading_path}\n{self.content}".encode()).hexdigest()

    @property
    def embed_text(self) -> str:
        """The heading path is part of the embedded text so short chunks keep their context."""
        return f"{self.heading_path}\n{self.content}"


def chunk_markdown(text: str, max_chars: int = MAX_CHARS) -> list[RawChunk]:
    """Split on headings (keeping the heading path), then split long sections by size."""
    sections = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=True).split_text(text)
    sizer = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=min(100, max_chars // 10)
    )
    chunks: list[RawChunk] = []
    for n, doc in enumerate(sizer.split_documents(sections)):
        path = " > ".join(doc.metadata[k] for _, k in _HEADERS if k in doc.metadata) or "(preamble)"
        chunks.append(RawChunk(str(n), path, doc.page_content.strip()))
    return [c for c in chunks if c.content]


def document_hash(text: str) -> str:
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


INDEX_LOCK_KEY = 7_302_001  # Postgres advisory lock shared by the CronJob and the index-worker


@asynccontextmanager
async def exclusive_index_run(pool: Pool) -> AsyncIterator[bool]:
    """Yield True if this process now holds the one indexing slot, False if another run has it.

    The CronJob and the index-worker both mirror the whole container, and a run that listed the
    documents earlier must not prune what a later run just added. A session-level advisory lock
    held on one connection makes them take turns; it is released on exit or if the pod dies.
    """
    async with pool.connection() as conn:
        row = await (
            await conn.execute("SELECT pg_try_advisory_lock(%s) AS got", (INDEX_LOCK_KEY,))
        ).fetchone()
        got = bool(row and row["got"])
        try:
            yield got
        finally:
            if got:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (INDEX_LOCK_KEY,))


async def _main(path: str | None) -> int:
    from support_assistant.blobstore import AzureBlobStore

    s = get_settings()
    embedder = azure_embedder(s)
    store: AzureBlobStore | None = None
    source: DocumentSource
    if path:
        source = LocalDirSource(Path(path))
    elif s.docs_storage_account:
        store = AzureBlobStore(s.docs_storage_account, s.docs_container)
        source = StoreDocumentSource(store)
    else:
        print("give a path, or set DOCS_STORAGE_ACCOUNT", file=sys.stderr)
        return 2
    registry = build_registry(s)
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
        if store is not None:
            await store.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else None)))
