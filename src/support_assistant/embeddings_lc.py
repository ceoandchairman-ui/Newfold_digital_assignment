"""LangChain `Embeddings` adapter over our `Embedder` (E2-3).

LangChain components (indexer chain, retrievers) take an `Embeddings`. The adapter delegates to our
`Embedder`, so batching, jittered retries, the dimension check and the fake embedder used in tests
all still apply. Async only: the services are async end to end.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from support_assistant.embeddings import Embedder


class LcEmbeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    @property
    def dim(self) -> int:
        return self._embedder.dim

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        (vec,) = await self._embedder.embed([text])
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("use aembed_documents; this adapter is async-only")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("use aembed_query; this adapter is async-only")
