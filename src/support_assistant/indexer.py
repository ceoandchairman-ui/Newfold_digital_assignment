"""Knowledge-base indexer (E2-2, E2-3) as a LangChain (LCEL) chain.

    markdown text -> split (MarkdownHeader + Recursive splitters) -> drop known chunks
                  -> embed (LangChain Embeddings) -> upsert into pgvector

Idempotent: chunks are keyed by sha256(heading path + content), so re-running only embeds new or
changed text. We keep our own table (hash, model/version columns, tsvector for hybrid search), so
the write step is plain SQL rather than a LangChain vector-store class.

Run as a Kubernetes Job:  python -m support_assistant.indexer /data/knowledge_base.md
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector_async

from support_assistant.config import get_settings
from support_assistant.db import Pool, make_pool
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
        return {"added": len(fresh), "total": len(inp["chunks"])}

    return (
        RunnableLambda(split)
        | RunnableLambda(drop_known)
        | RunnableLambda(embed)
        | RunnableLambda(upsert)
    )


async def _main(path: str) -> None:
    s = get_settings()
    key = os.environ.get("EMBEDDER_API_KEY", "")

    async def token() -> str:
        return key

    embedder = HttpEmbedder(
        s.embedder_url, s.embedding_model, dim=s.embedding_dim, token_provider=token
    )
    pool = make_pool(s)
    await pool.open()
    try:
        chain = build_index_chain(pool, LcEmbeddings(embedder), s.embedding_model)
        file = Path(path)
        result = await chain.ainvoke(
            {"source": file.name, "text": file.read_text(encoding="utf-8")}
        )
        print(f"indexed {result['added']} new of {result['total']} chunks")
    finally:
        await embedder.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1]))
