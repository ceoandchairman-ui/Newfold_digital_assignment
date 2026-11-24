"""The fat backend offers the customer's last few days of conversations on a session's first turn."""

import json
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import pytest

from support_assistant.backend import app as backend
from support_assistant.config import Settings
from support_assistant.context import history_context


class Cur:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class Pool:
    def __init__(self, sessions=None, fail=False):
        self.sessions, self.fail, self.queries = sessions or [], fail, []

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Conn:
            async def execute(self, sql, params=None):
                pool.queries.append((" ".join(sql.split()), params))
                if pool.fail:
                    raise ConnectionError("db down")
                return Cur(pool.sessions if "FROM session_conversations" in sql else [])

        yield Conn()


class KV:
    def __init__(self):
        self.data = {"verified:S1": "C1"}

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v, ex=None, nx=False):
        self.data[k] = v
        return True

    async def delete(self, k):
        self.data.pop(k, None)


class Memory:
    def __init__(self, history):
        self.history = history

    async def load(self, cid, sid):
        return self.history

    async def append(self, cid, sid, new):
        return new


def _session(sid, user, answer, day):
    return {
        "session_id": sid,
        "updated_at": datetime(2026, 11, day),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def _setup(pool, history=(), **settings):
    sent: list[dict] = []

    def orch(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json={"reply": "ok", "pending_approval": None, "escalated": False}
        )

    backend.app.state.s = Settings(database_url="postgresql://x", redis_url="redis://x", **settings)
    backend.app.state.redis, backend.app.state.pool = KV(), pool
    backend.app.state.memory, backend.app.state.rewrite = Memory(list(history)), None
    backend.app.state.http = httpx.AsyncClient(
        base_url="http://orch", transport=httpx.MockTransport(orch)
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://t")
    return client, sent


async def _chat(client):
    return await client.post("/chat", json={"session_id": "S1", "message": "my data is slow again"})


async def test_the_first_turn_carries_a_note_about_the_last_days_conversations():
    pool = Pool([_session("OLD1", "sim not working", "Reseat the SIM", 22)])
    client, sent = _setup(pool)
    await _chat(client)
    first = sent[0]["messages"][0]
    assert first["role"] == "system" and "Earlier conversations" in first["content"]
    assert "sim not working" in first["content"] and "2026-11-22" in first["content"]
    sql, params = pool.queries[0]
    assert "customer_id = %s" in sql and params == ("C1", 5, 6)  # this customer, 5 days, one spare


async def test_the_current_session_is_not_repeated_back_to_itself():
    pool = Pool(
        [_session("S1", "this very session", "x", 23), _session("OLD1", "older issue", "y", 21)]
    )
    client, sent = _setup(pool)
    await _chat(client)
    note = sent[0]["messages"][0]["content"]
    assert "older issue" in note and "this very session" not in note


async def test_later_turns_do_not_reload_the_history():
    pool = Pool([_session("OLD1", "older issue", "y", 21)])
    client, sent = _setup(pool, history=[{"role": "user", "content": "earlier in this session"}])
    await _chat(client)
    assert pool.queries == []  # only the first turn of a session asks
    assert all(m["role"] != "system" for m in sent[0]["messages"])


async def test_history_can_be_turned_off():
    pool = Pool([_session("OLD1", "older issue", "y", 21)])
    client, sent = _setup(pool, history_days=0)
    await _chat(client)
    assert pool.queries == [] and sent[0]["messages"][0]["role"] == "user"


async def test_a_database_error_never_costs_the_turn():
    client, sent = _setup(Pool(fail=True))
    r = await _chat(client)
    assert r.status_code == 200 and sent[0]["messages"][0]["role"] == "user"


async def test_the_window_and_the_number_of_sessions_come_from_settings():
    pool = Pool([])
    client, _ = _setup(pool, history_days=3, history_max_sessions=2)
    await _chat(client)
    assert pool.queries[0][1] == ("C1", 3, 3)


# --- the note itself ----------------------------------------------------------------------------
def test_the_note_is_capped_so_it_cannot_crowd_out_the_turn():
    many = [_session(f"S{i}", "x" * 300, "y" * 300, 20) for i in range(10)]
    assert len(history_context(many, max_chars=500)) <= 500


def test_sessions_without_any_customer_message_add_nothing():
    assert (
        history_context([{"session_id": "S", "messages": [], "updated_at": datetime(2026, 11, 1)}])
        == ""
    )
    assert history_context([]) == ""


@pytest.mark.parametrize("bad", [None, [], [{"role": "assistant", "content": "hi"}]])
def test_malformed_sessions_are_skipped(bad):
    assert history_context([{"session_id": "S", "messages": bad, "updated_at": "2026-11-01"}]) == ""
