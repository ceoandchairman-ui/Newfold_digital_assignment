"""How every other component reaches the retrieval service.

Every failure (unreachable, timeout, error status, unreadable answer) is one exception,
`ExternalServiceUnavailable`, so a caller has one thing to handle. "Found nothing" is not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from support_assistant.contracts import (
    Evidence,
    EvidenceRef,
    FetchRequest,
    FetchResponse,
    SearchRequest,
    SearchResponse,
)
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import correlation_headers
from support_assistant.resilience import is_transport_error, retrying

MAX_QUERY_CHARS = 4000


class Retrieval(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        """Company knowledge relevant to the query, best first. May be empty."""
        ...

    async def fetch(self, refs: Sequence[EvidenceRef]) -> FetchResponse:
        """The referenced chunks again. Gone or changed ones are listed in `missing`."""
        ...


class HttpRetrievalClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    # Both calls only read, so one retry after a connection problem is safe.
    @retrying("retrieval", attempts=2, base=0.2, cap=1.0, retry_on=is_transport_error)
    async def _send(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return await self._http.post(path, json=body, headers=correlation_headers())

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            resp = await self._send(path, body)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceUnavailable("retrieval service unavailable") from exc

    async def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not query.strip():
            return []
        body = SearchRequest(store="company_knowledge", query=query[:MAX_QUERY_CHARS], top_k=top_k)
        data = await self._post("/v1/search", body.model_dump())
        try:
            return SearchResponse.model_validate(data).results
        except ValueError as exc:
            raise ExternalServiceUnavailable("unreadable retrieval answer") from exc

    async def fetch(self, refs: Sequence[EvidenceRef]) -> FetchResponse:
        if not refs:
            return FetchResponse()
        data = await self._post("/v1/fetch", FetchRequest(refs=list(refs)).model_dump())
        try:
            return FetchResponse.model_validate(data)
        except ValueError as exc:
            raise ExternalServiceUnavailable("unreadable retrieval answer") from exc

    async def aclose(self) -> None:
        await self._http.aclose()
