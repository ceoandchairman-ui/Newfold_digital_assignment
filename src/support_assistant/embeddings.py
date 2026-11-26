"""Embedding client behind an interface (E2-3).

Production calls Azure OpenAI's OpenAI-compatible `/openai/v1/embeddings` endpoint with the
pod's own Entra identity (no API key). The dimension comes from configuration and is sent with
every request, so `vector(N)` and the model always agree.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import httpx

from support_assistant.config import Settings
from support_assistant.errors import ExternalServiceUnavailable, ValidationError
from support_assistant.resilience import retrying

AZURE_AI_SCOPE = "https://cognitiveservices.azure.com/.default"


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, hash-seeded unit vectors for offline tests."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        raw: list[float] = []
        counter = 0
        while len(raw) < self._dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            raw.extend((b - 127.5) / 127.5 for b in digest)
            counter += 1
        vec = raw[: self._dim]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class HttpEmbedder:
    """OpenAI-compatible embeddings API client with batching and retry with jitter."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        token_provider: Callable[[], Awaitable[str]] | None = None,
        batch_size: int = 32,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._token_provider = token_provider
        self._on_close = on_close
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(await self._embed_batch(list(texts[i : i + self._batch_size])))
        return out

    # Embedding has no side effect, so any transient failure may be repeated.
    @retrying("embed", attempts=4, base=0.2, cap=5.0, retry_on=_is_retryable)
    async def _call(self, batch: list[str]) -> httpx.Response:
        headers = {}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {await self._token_provider()}"
        body = {"model": self._model, "input": batch, "dimensions": self._dim}
        resp = await self._client.post("/embeddings", json=body, headers=headers)
        resp.raise_for_status()
        return resp

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = await self._call(batch)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise ExternalServiceUnavailable("embedder API unavailable") from exc
        data = resp.json()["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        for vec in vectors:
            if len(vec) != self._dim:
                raise ValidationError(f"embedding dimension {len(vec)} != expected {self._dim}")
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._on_close is not None:
            await self._on_close()


def azure_embedder(settings: Settings) -> HttpEmbedder:
    """The embedding deployment on this environment's Azure OpenAI account.

    The address is derived from `azure_openai_endpoint` and cannot be set separately, so the
    Entra token is only ever sent to that account. `embedding_model` is the deployment name.
    """
    if not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for embeddings")

    from azure.identity.aio import DefaultAzureCredential

    credential = DefaultAzureCredential()

    async def token() -> str:
        return (await credential.get_token(AZURE_AI_SCOPE)).token

    return HttpEmbedder(
        settings.azure_openai_endpoint.rstrip("/") + "/openai/v1",
        settings.embedding_model,
        dim=settings.embedding_dim,
        token_provider=token,
        on_close=credential.close,
    )
