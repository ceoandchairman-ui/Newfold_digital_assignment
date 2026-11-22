"""Episodic memory, governance, context compaction, MCP connection reuse and the rule checker."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from support_assistant import governance
from support_assistant.agents.mcp_client import McpConnection
from support_assistant.context import compact_history, episode_context
from support_assistant.embeddings import FakeEmbedder
from support_assistant.errors import ExternalServiceUnavailable, ValidationError
from support_assistant.evaluation_rules import check_transcript
from support_assistant.services import episodes


# --- a recording fake pool shared by the SQL-shape tests -------------------------------------------
class Cur:
    def __init__(self, rows=None, rowcount=0):
        self._rows, self.rowcount = rows or [], rowcount

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class RecConn:
    def __init__(self, log, rows=None, rowcount=3):
        self.log, self.rows, self.rowcount = log, rows, rowcount

    async def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))
        return Cur(self.rows, self.rowcount)

    @asynccontextmanager
    async def transaction(self):
        yield


class RecPool:
    def __init__(self, rows=None, rowcount=3):
        self.log: list[tuple[str, Any]] = []
        self._rows, self._rowcount = rows, rowcount

    @asynccontextmanager
    async def connection(self):
        yield RecConn(self.log, self._rows, self._rowcount)


@pytest.fixture(autouse=True)
def _no_vector_registration(monkeypatch):
    async def noop(_conn):
        return None

    monkeypatch.setattr(episodes, "register_vector_async", noop)


# --- episodic memory (E2-5) ----------------------------------------------------------------------------
async def test_episode_search_is_always_filtered_by_customer():
    pool = RecPool(rows=[{"summary": "s", "outcome": "ok", "created_at": "t"}])
    out = await episodes.search_episodes(pool, FakeEmbedder(4), customer_id="C1", query="q")
    sql, params = pool.log[0]
    assert "WHERE customer_id = %s" in sql and params[0] == "C1" and out[0]["outcome"] == "ok"


async def test_episode_save_is_idempotent_and_validated():
    pool = RecPool(rows=[{"id": 1}])
    assert await episodes.save_episode(
        pool, FakeEmbedder(4), customer_id="C1", summary="x", outcome="o", model="m"
    )
    assert "ON CONFLICT (content_hash) DO NOTHING" in pool.log[0][0]
    dup = RecPool(rows=[])  # conflict: RETURNING gives no row
    assert not await episodes.save_episode(
        dup, FakeEmbedder(4), customer_id="C1", summary="x", outcome="o", model="m"
    )
    with pytest.raises(ValidationError):
        await episodes.save_episode(
            pool, FakeEmbedder(4), customer_id="C1", summary="  ", outcome="o", model="m"
        )


def test_same_episode_hashes_the_same_but_differs_per_customer():
    assert episodes.episode_hash("C1", "a", "b") == episodes.episode_hash("C1", "a", "b")
    assert episodes.episode_hash("C1", "a", "b") != episodes.episode_hash("C2", "a", "b")


# --- governance (E5-8) --------------------------------------------------------------------------------------
class ScanRedis:
    def __init__(self, keys):
        self.keys, self.deleted = keys, []

    async def scan_iter(self, match):
        import fnmatch

        for k in self.keys:
            if fnmatch.fnmatch(k, match):
                yield k

    async def delete(self, key):
        self.deleted.append(key)
        return 1


async def test_erasure_removes_memory_and_caches_but_never_the_ledger():
    pool = RecPool(rowcount=2)
    redis = ScanRedis(["sess:C1:a", "sess:C2:a", "tool:C1", "tool:C2"])
    out = await governance.erase_customer_memory(pool, [redis], "C1")
    assert out == {"sessions": 2, "episodes": 2, "cache_keys": 2}
    assert sorted(redis.deleted) == ["sess:C1:a", "tool:C1"]  # C2 untouched
    touched = " ".join(sql for sql, _ in pool.log).lower()
    assert "bills" not in touched and "transaction_history" not in touched


async def test_prune_uses_a_positive_age_and_rejects_nonsense():
    pool = RecPool(rowcount=5)
    assert await governance.prune_sessions(pool, 30) == 5
    assert pool.log[0][1] == (30,)
    with pytest.raises(ValueError):
        await governance.prune_sessions(pool, 0)


# --- context compaction (E6-2) --------------------------------------------------------------------------------
def _msgs(n, size=500):
    return [{"role": "user", "content": f"{i}:" + "x" * size} for i in range(n)]


def test_recent_turns_stay_verbatim_and_old_ones_are_shortened():
    out = compact_history(_msgs(10), keep_recent=3, old_chars=50)
    assert len(out[-1]["content"]) > 400 and len(out[0]["content"]) == 50
    assert out[-3:] == _msgs(10)[-3:]


def test_total_size_is_capped_by_dropping_oldest_first_but_never_recent_turns():
    out = compact_history(_msgs(30), keep_recent=4, old_chars=200, max_chars=2500)
    assert out[-4:] == _msgs(30)[-4:] and sum(len(m["content"]) for m in out) <= 2500 + 4 * 510


def test_short_history_is_unchanged_and_bad_args_rejected():
    assert compact_history(_msgs(2)) == _msgs(2)
    with pytest.raises(ValueError):
        compact_history([], max_chars=0)


def test_episode_context_text():
    assert episode_context([]) == ""
    assert "outcome: resolved" in episode_context([{"summary": "no signal", "outcome": "resolved"}])


# --- MCP connection reuse (E8-2) -------------------------------------------------------------------------------
class FakeSession:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    async def call_tool(self, tool, args):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError("dropped")
        return {"tool": tool, "args": args}


def _opener(sessions: list[FakeSession], fail_connect: bool = False):
    @asynccontextmanager
    async def open_(url):
        if fail_connect:
            raise ConnectionError("refused")
        s = sessions.pop(0)
        yield s
        await asyncio.sleep(0)

    return open_


async def test_many_calls_share_one_connection():
    conn = McpConnection("u", _opener([FakeSession()]))
    results = await asyncio.gather(*(conn.call("get_bills", {"i": i}) for i in range(10)))
    assert len(results) == 10 and conn.connects == 1
    await conn.aclose()


async def test_a_dropped_connection_is_reopened_and_the_call_retried():
    conn = McpConnection("u", _opener([FakeSession(fail_after=1), FakeSession()]))
    assert (await conn.call("t", {}))["tool"] == "t"
    assert (await conn.call("t", {}))["tool"] == "t"  # first session fails here, second serves it
    assert conn.connects == 2
    await conn.aclose()


async def test_unreachable_server_raises_service_unavailable():
    conn = McpConnection("u", _opener([], fail_connect=True))
    with pytest.raises(ExternalServiceUnavailable):
        await conn.call("t", {})
    await conn.aclose()


# --- rule checker (E9-5) ---------------------------------------------------------------------------------------------
GOOD = [
    {"type": "auth", "verified": True},
    {"type": "user", "text": "pay my bill"},
    {"type": "approval", "approved": True},
    {"type": "tool", "name": "process_payment", "ok": True, "customer_id": "C1"},
    {"type": "assistant", "text": "Your payment was successful."},
]


def _rules(events):
    return {v.rule for v in check_transcript(events, customer_id="C1")}


def test_a_clean_payment_transcript_has_no_violations():
    assert _rules(GOOD) == set()


def test_payment_without_approval_is_flagged():
    bad = [e for e in GOOD if e["type"] != "approval"]
    assert "payment_without_approval" in _rules(bad)


def test_an_approval_covers_exactly_one_payment():
    twice = [
        *GOOD[:4],
        {"type": "tool", "name": "process_payment", "ok": True, "customer_id": "C1"},
    ]
    assert "payment_without_approval" in _rules(twice)


def test_tool_before_auth_and_cross_tenant_are_flagged():
    events = [{"type": "tool", "name": "get_bills", "ok": True, "customer_id": "C9"}]
    assert {"tool_before_auth", "cross_tenant_tool"} <= _rules(events)


def test_claiming_payment_without_a_paid_result_is_flagged():
    events = [
        {"type": "auth", "verified": True},
        {"type": "assistant", "text": "Payment received, thanks!"},
    ]
    assert "unverified_payment_claim" in _rules(events)


def test_wrong_escalation_wording_and_double_reply_are_flagged():
    events = [
        {"type": "auth", "verified": True},
        {"type": "assistant", "text": "you are being transferred to a human, bye"},
        {"type": "assistant", "text": "anything else?"},
    ]
    assert {"escalation_wording", "two_replies_in_a_row"} <= _rules(events)
