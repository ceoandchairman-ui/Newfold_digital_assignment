"""The index worker notices document changes by itself, indexes them once the upload settles, and
never runs at the same time as the CronJob."""

from support_assistant.blobstore import InMemoryBlobStore
from support_assistant.index_worker import Tick, Watcher, fingerprint
from support_assistant.indexer import INDEX_LOCK_KEY, exclusive_index_run


class Harness:
    def __init__(self, settle: int = 2) -> None:
        self.store = InMemoryBlobStore()
        self.runs = 0
        self.slot_free = True
        self.fail_next = 0
        self.list_fails = False
        self.watcher = Watcher(self._list, self._reindex, settle_polls=settle)

    async def _list(self) -> dict[str, str]:
        if self.list_fails:
            raise RuntimeError("storage down")
        return await self.store.etags()

    async def _reindex(self) -> bool:
        if not self.slot_free:
            return False
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("embedder down")
        self.runs += 1
        return True

    async def upload(self, name: str, text: str) -> None:
        self.store.blobs.pop(name, None)
        await self.store.put(name, text.encode(), content_type="text/markdown")


async def test_an_upload_is_indexed_after_it_settles_and_not_before():
    h = Harness()
    await h.upload("manual.md", "# Reset\nhold the button")
    assert await h.watcher.tick() is Tick.WAITING  # seen once: the upload may still be going
    assert h.runs == 0
    assert await h.watcher.tick() is Tick.INDEXED  # unchanged since the last look
    assert h.runs == 1


async def test_nothing_runs_when_nothing_changed():
    h = Harness()
    await h.upload("manual.md", "a")
    await h.watcher.tick()
    await h.watcher.tick()
    assert [await h.watcher.tick() for _ in range(3)] == [Tick.IDLE] * 3
    assert h.runs == 1


async def test_many_files_uploaded_over_several_looks_cause_one_run():
    h = Harness()
    for n in range(5):
        await h.upload(f"doc{n}.md", f"text {n}")
        assert await h.watcher.tick() is Tick.WAITING  # the listing keeps changing
    assert await h.watcher.tick() is Tick.INDEXED
    assert h.runs == 1


async def test_an_edit_is_noticed_because_the_etag_changes_even_if_the_name_does_not():
    h = Harness()
    await h.upload("manual.md", "version one")
    await h.watcher.tick()
    await h.watcher.tick()
    await h.upload("manual.md", "version two")
    await h.watcher.tick()
    assert await h.watcher.tick() is Tick.INDEXED
    assert h.runs == 2


async def test_a_deleted_document_triggers_a_run_so_its_chunks_are_removed():
    h = Harness()
    await h.upload("a.md", "a")
    await h.upload("b.md", "b")
    await h.watcher.tick()
    await h.watcher.tick()
    del h.store.blobs["b.md"]
    await h.watcher.tick()
    assert await h.watcher.tick() is Tick.INDEXED
    assert h.runs == 2


async def test_a_failed_run_is_retried_on_the_next_look_and_nothing_is_marked_done():
    h = Harness()
    await h.upload("manual.md", "a")
    h.fail_next = 1
    await h.watcher.tick()
    assert await h.watcher.tick() is Tick.FAILED
    assert await h.watcher.tick() is Tick.INDEXED  # same listing, tries again
    assert h.runs == 1


async def test_a_storage_listing_error_changes_nothing():
    h = Harness()
    await h.upload("manual.md", "a")
    await h.watcher.tick()
    await h.watcher.tick()
    h.list_fails = True
    assert await h.watcher.tick() is Tick.FAILED
    h.list_fails = False
    assert await h.watcher.tick() is Tick.IDLE  # still what was indexed: no needless run
    assert h.runs == 1


async def test_when_the_cronjob_holds_the_slot_the_worker_waits_and_runs_later():
    h = Harness()
    await h.upload("manual.md", "a")
    h.slot_free = False
    await h.watcher.tick()
    assert await h.watcher.tick() is Tick.WAITING
    assert h.runs == 0
    h.slot_free = True
    assert await h.watcher.tick() is Tick.INDEXED
    assert h.runs == 1


async def test_the_first_look_at_an_empty_container_runs_once_and_then_idles():
    h = Harness(settle=1)
    assert await h.watcher.tick() is Tick.INDEXED  # nothing was ever indexed
    assert await h.watcher.tick() is Tick.IDLE


def test_the_fingerprint_depends_on_names_and_etags_not_on_listing_order():
    assert fingerprint({"a": "1", "b": "2"}) == fingerprint({"b": "2", "a": "1"})
    assert fingerprint({"a": "1"}) != fingerprint({"a": "2"})
    assert fingerprint({"a": "1"}) != fingerprint({"b": "1"})


# --- the advisory lock shared with the CronJob ----------------------------------------------------
class _Cursor:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, pool):
        self._pool = pool

    async def execute(self, sql, params=None):
        if "pg_try_advisory_lock" in sql:
            assert params == (INDEX_LOCK_KEY,)
            if self._pool.held:
                return _Cursor({"got": False})
            self._pool.held = True
            return _Cursor({"got": True})
        if "pg_advisory_unlock" in sql:
            self._pool.held = False
            self._pool.unlocks += 1
        return _Cursor(None)


class _ConnCtx:
    def __init__(self, pool):
        self._pool = pool

    async def __aenter__(self):
        return _Conn(self._pool)

    async def __aexit__(self, *exc):
        return False


class FakeLockPool:
    def __init__(self):
        self.held = False
        self.unlocks = 0

    def connection(self):
        return _ConnCtx(self)


async def test_only_one_run_holds_the_indexing_slot_and_it_is_released():
    pool = FakeLockPool()
    async with exclusive_index_run(pool) as first:
        assert first is True
        async with exclusive_index_run(pool) as second:
            assert second is False  # the CronJob would exit here, the worker would wait
        assert pool.unlocks == 0  # the loser must not release the winner's lock
    assert pool.held is False and pool.unlocks == 1


async def test_the_slot_is_released_even_when_the_run_fails():
    pool = FakeLockPool()
    try:
        async with exclusive_index_run(pool):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert pool.held is False
