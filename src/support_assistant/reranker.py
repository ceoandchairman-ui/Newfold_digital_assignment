"""Reranker behind an interface (D18). Reorders the fused top candidates; never the only retriever.

If the reranker fails, callers keep the fused (RRF) order, so retrieval degrades instead of failing.
The request shape is the common `/rerank` one (query, documents, top_n) -> results[{index, score}].
Confirm it against the chosen provider in the first integration run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from support_assistant.errors import ExternalServiceUnavailable


class Reranker(Protocol):
    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[int]:
        """Return document indexes, best first."""
        ...


class NoopReranker:
    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[int]:
        return list(range(min(top_n, len(documents))))


class HttpReranker:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[int]:
        if not documents:
            return []
        try:
            resp = await self._http.post(
                "/rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ExternalServiceUnavailable("reranker unavailable") from exc
        ordered = sorted(results, key=lambda r: r["score"], reverse=True)
        return [int(r["index"]) for r in ordered[:top_n]]

    async def aclose(self) -> None:
        await self._http.aclose()
