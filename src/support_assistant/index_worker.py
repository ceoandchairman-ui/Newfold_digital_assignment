"""Index worker: keeps the knowledge base in step with the `knowledge-base` container, on its own.

    team uploads, edits or deletes a document  ->  within about a minute the change is searchable

It looks at the container every INDEX_POLL_SECONDS (a cheap listing of names and etags over the
private endpoint). When the listing differs from what was last indexed AND has stayed the same for
INDEX_SETTLE_POLLS looks in a row, it runs the same idempotent mirror the CronJob runs. Waiting for
the listing to settle means a 20-file upload is indexed once, after it finishes, not 20 times.

The 6-hour CronJob stays as a safety net. Both take the same Postgres advisory lock, so they never
run at the same time. Nothing here needs cluster access: anyone who may write to the container
(Entra group docs_writers_object_id) can publish a document just by uploading it.

Why polling and not Event Grid: delivering blob events to the private Service Bus namespace would
need that namespace opened to trusted public services. A listing every 30 s costs almost nothing and
keeps every service private.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from enum import StrEnum

import structlog

from support_assistant.config import get_settings

log = structlog.get_logger()


class Tick(StrEnum):
    IDLE = "idle"  # nothing changed since the last successful index
    WAITING = "waiting"  # a change was seen; waiting for the container to stop changing
    INDEXED = "indexed"
    FAILED = "failed"  # listing or indexing failed; the next tick tries again


def fingerprint(etags: dict[str, str]) -> str:
    """One value that changes when a document is added, edited or removed."""
    raw = "\n".join(f"{name}\t{tag}" for name, tag in sorted(etags.items()))
    return hashlib.sha256(raw.encode()).hexdigest()


class Watcher:
    """The decision logic, free of Azure and Postgres so it can be tested directly."""

    def __init__(
        self,
        list_etags: Callable[[], Awaitable[dict[str, str]]],
        reindex: Callable[[], Awaitable[bool]],
        *,
        settle_polls: int = 2,
    ) -> None:
        self._list, self._reindex, self._settle = list_etags, reindex, max(1, settle_polls)
        self._indexed: str | None = None  # None until the first successful run
        self._pending: str | None = None
        self._seen = 0

    async def tick(self) -> Tick:
        try:
            current = fingerprint(await self._list())
        except Exception:  # noqa: BLE001 - storage hiccup: keep state and look again
            log.warning("index_listing_failed", exc_info=True)
            return Tick.FAILED
        if current == self._indexed:
            self._pending, self._seen = None, 0
            return Tick.IDLE
        if current == self._pending:
            self._seen += 1
        else:
            self._pending, self._seen = current, 1
        if self._seen < self._settle:
            return Tick.WAITING
        try:
            ran = await self._reindex()
        except Exception:  # noqa: BLE001 - embedder or database down: retry on the next tick
            log.warning("index_run_failed", exc_info=True)
            return Tick.FAILED
        if not ran:  # another run (the CronJob) holds the slot; try again next tick
            return Tick.WAITING
        self._indexed, self._pending, self._seen = current, None, 0
        return Tick.INDEXED


async def run() -> None:
    from support_assistant.blobstore import AzureBlobStore
    from support_assistant.db import make_pool
    from support_assistant.docsource import StoreDocumentSource
    from support_assistant.embeddings import HttpEmbedder
    from support_assistant.embeddings_lc import LcEmbeddings
    from support_assistant.indexer import exclusive_index_run, index_source

    s = get_settings()
    if not s.docs_storage_account:
        raise RuntimeError("DOCS_STORAGE_ACCOUNT is required")
    key = os.environ.get("EMBEDDER_API_KEY", "")

    async def token() -> str:
        return key

    embedder = HttpEmbedder(
        s.embedder_url, s.embedding_model, dim=s.embedding_dim, token_provider=token
    )
    store = AzureBlobStore(s.docs_storage_account, s.docs_container)
    pool = make_pool(s)
    await pool.open()

    async def reindex() -> bool:
        async with exclusive_index_run(pool) as got:
            if not got:
                return False
            report = await index_source(
                pool, LcEmbeddings(embedder), StoreDocumentSource(store), s.embedding_model
            )
        log.info(
            "knowledge_base_indexed",
            documents=report.documents,
            chunks_added=report.chunks_added,
            chunks_removed=report.chunks_removed,
            documents_removed=report.sources_removed,
            skipped=len(report.skipped),
        )
        return True

    watcher = Watcher(store.etags, reindex, settle_polls=s.index_settle_polls)
    log.info("index_worker_started", poll_seconds=s.index_poll_seconds)
    try:
        while True:
            result = await watcher.tick()
            if result is not Tick.IDLE:
                log.info("index_tick", result=result.value)
            await asyncio.sleep(s.index_poll_seconds)
    finally:
        await embedder.aclose()
        await pool.close()
        await store.aclose()


if __name__ == "__main__":
    asyncio.run(run())
