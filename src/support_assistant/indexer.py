"""Knowledge-base indexer (E2-2, E2-3) as a LangChain (LCEL) chain.

    markdown text -> split (MarkdownHeader + Recursive splitters) -> drop known chunks
                  -> embed (LangChain Embeddings) -> upsert into pgvector

Idempotent: chunks are keyed by sha256(heading path + content), so re-running only embeds new or
changed text. We keep our own table (hash, model/version columns, tsvector for hybrid search), so
the write step is plain SQL rather than a LangChain vector-store class.

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
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector_async

from support_assistant.config import get_settings
from support_assistant.db import Pool, make_pool
from support_assistant.docsource import DocumentSource, LocalDirSource, StoreDocumentSource
from support_assistant.embeddings import HttpEmbedder
from support_assistant.embeddings_lc import LcEmbeddings

MAX_CHARS = 1500
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


def build_index_chain(
    pool: Pool, embeddings: LcEmbeddings, model: str, version: str = "1"
) -> Runnable[dict[str, Any], dict[str, int]]:
    """The indexing pipeline as one LCEL chain. Input: {"source": str, "text": str}."""

    async def split(inp: dict[str, Any]) -> dict[str, Any]:
        return {**inp, "chunks": chunk_markdown(inp["text"])}

    async def drop_known(inp: dict[str, Any]) -> dict[str, Any]:
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT content_hash FROM knowledge_base_chunks")
            known = {r["content_hash"] for r in await cur.fetchall()}
        return {**inp, "fresh": [c for c in inp["chunks"] if c.digest not in known]}

    async def embed(inp: dict[str, Any]) -> dict[str, Any]:
        fresh: list[RawChunk] = inp["fresh"]
        docs = [Document(page_content=c.embed_text) for c in fresh]
        vectors = await embeddings.aembed_documents([d.page_content for d in docs]) if docs else []
        return {**inp, "vectors": vectors}

    async def upsert(inp: dict[str, Any]) -> dict[str, int]:
        fresh: list[RawChunk] = inp["fresh"]
        async with pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.transaction():
                for c, v in zip(fresh, inp["vectors"], strict=True):
                    await conn.execute(
                        """
                        INSERT INTO knowledge_base_chunks
                          (source, section_id, heading_path, content, content_hash,
                           embedding, embedding_model, embedding_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO NOTHING
                        """,
                        (
                            inp["source"],
                            c.section_id,
                            c.heading_path,
                            c.content,
                            c.digest,
                            v,
                            model,
                            version,
                        ),
                    )
                # Mirror the document: drop this source's chunks that are no longer in its text.
                keep = [c.digest for c in inp["chunks"]]
                cur = await conn.execute(
                    "DELETE FROM knowledge_base_chunks WHERE source = %s AND NOT (content_hash = ANY(%s))",
                    (inp["source"], keep),
                )
                removed = cur.rowcount
        return {"added": len(fresh), "total": len(inp["chunks"]), "removed": removed}

    return (
        RunnableLambda(split)
        | RunnableLambda(drop_known)
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


async def index_source(
    pool: Pool, embeddings: LcEmbeddings, source: DocumentSource, model: str, version: str = "1"
) -> IndexReport:
    """Index every document in `source` and make the knowledge base match it.

    Safety: if the source yields NO documents, nothing is deleted. An empty or failed listing is
    far more likely a storage problem than an intent to erase the whole knowledge base.
    """
    loaded = await source.load()
    chain = build_index_chain(pool, embeddings, model, version)
    added = removed = 0
    for name in loaded.names:
        result = await chain.ainvoke({"source": name, "text": loaded.documents[name]})
        added += result["added"]
        removed += result["removed"]

    sources_removed = 0
    if loaded.names:  # never prune on an empty listing
        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "DELETE FROM knowledge_base_chunks WHERE NOT (source = ANY(%s))",
                    (loaded.names,),
                )
                sources_removed = cur.rowcount
    return IndexReport(len(loaded.names), added, removed, sources_removed, loaded.skipped)


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
    key = os.environ.get("EMBEDDER_API_KEY", "")

    async def token() -> str:
        return key

    embedder = HttpEmbedder(
        s.embedder_url, s.embedding_model, dim=s.embedding_dim, token_provider=token
    )
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
    pool = make_pool(s)
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
        if store is not None:
            await store.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else None)))
