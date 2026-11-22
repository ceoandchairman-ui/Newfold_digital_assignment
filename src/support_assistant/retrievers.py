"""LangChain retriever over our pgvector knowledge base (E2-4, E2-7).

A standard `BaseRetriever`, so graph nodes, chains and the evaluation harness can all use it with
`await retriever.ainvoke(query)`. Search itself is `search_kb` (vector + keyword + RRF, optional
reranker); this class only adapts it to LangChain `Document`s.
"""

from __future__ import annotations

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, SkipValidation

from support_assistant.db import Pool
from support_assistant.embeddings import Embedder
from support_assistant.reranker import Reranker
from support_assistant.services.retrieval import search_kb


class HybridPgRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Protocols and the pool type cannot be validated by pydantic (no isinstance check), so
    # validation is skipped for them; mypy still checks the types.
    pool: SkipValidation[Pool]
    embedder: SkipValidation[Embedder]
    reranker: SkipValidation[Reranker | None] = None
    top_k: int = 5
    candidates: int = 20

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = await search_kb(
            self.pool,
            self.embedder,
            query,
            top_k=self.top_k,
            candidates=self.candidates,
            reranker=self.reranker,
        )
        return [
            Document(
                page_content=c.content,
                metadata={
                    "id": c.id,
                    "section_id": c.section_id,
                    "heading_path": c.heading_path,
                    "score": c.score,
                },
            )
            for c in chunks
        ]

    # LangChain inspects THIS signature to decide whether to pass `run_manager` to the async
    # method too. Leaving it out makes `ainvoke` crash with a TypeError. Keep it.
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("use ainvoke; retrieval is async-only")
