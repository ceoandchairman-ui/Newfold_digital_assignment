"""Loading a customer's earlier conversations, and the note built from them."""

from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from support_assistant.config import Settings
from support_assistant.context import history_context
from support_assistant.services import customer_memory
from support_assistant.services.customer_memory import load_customer_memory


class Cur:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class Pool:
    def __init__(self, sessions=None, *, returning=None, fail=()):
        self.sessions, self.fail, self.queries = sessions or [], fail, []
        self.returning = bool(self.sessions) if returning is None else returning

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Conn:
            async def execute(self, sql, params=None):
                sql = " ".join(sql.split())
                pool.queries.append((sql, params))
                kind = "exists" if "AS found" in sql else "sessions"
                if kind in pool.fail:
                    raise ConnectionError("db down")
                return Cur([{"found": pool.returning}] if kind == "exists" else pool.sessions)

        yield Conn()


def _session(sid, user, answer, day):
    return {
        "session_id": sid,
        "updated_at": datetime(2026, 11, day),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def _settings(**values):
    return Settings(database_url="postgresql://x", redis_url="redis://x", **values)


async def _load(pool, embedder=None, **settings):
    return await load_customer_memory(
        pool, embedder, _settings(**settings), "C1", "S1", "my data is slow again"
    )


async def test_a_returning_customer_gets_a_note_about_the_last_days_conversations():
    pool = Pool([_session("OLD1", "sim not working", "Reseat the SIM", 22)])
    memory = await _load(pool)
    assert memory.returning and memory.complete
    assert "Earlier conversations" in memory.note
    assert "sim not working" in memory.note and "2026-11-22" in memory.note
    sql, params = pool.queries[1]
    assert "customer_id = %s" in sql and params == ("C1", 5, 6)  # this customer, 5 days, one spare


async def test_a_new_customer_costs_one_query_and_no_embedding_call():
    pool = Pool([])
    memory = await _load(pool, embedder=object())  # an object() would fail if it were used
    assert not memory.returning and memory.note == "" and memory.complete
    assert len(pool.queries) == 1 and pool.queries[0][1] == ("C1", "S1", "C1")


async def test_past_episodes_are_added_for_a_returning_customer(monkeypatch):
    async def search(pool, embedder, *, customer_id, query, top_k=3):
        assert (customer_id, query) == ("C1", "my data is slow again")
        return [{"summary": "no signal last week", "outcome": "resolved"}]

    monkeypatch.setattr(customer_memory, "search_episodes", search)
    memory = await _load(Pool([], returning=True), embedder=object())
    assert "no signal last week" in memory.note and memory.complete


async def test_the_current_session_is_not_repeated_back_to_itself():
    pool = Pool(
        [_session("S1", "this very session", "x", 23), _session("OLD1", "older issue", "y", 21)]
    )
    note = (await _load(pool)).note
    assert "older issue" in note and "this very session" not in note


async def test_history_can_be_turned_off():
    pool = Pool([_session("OLD1", "older issue", "y", 21)])
    memory = await _load(pool, history_days=0)
    assert memory.returning and memory.note == "" and len(pool.queries) == 1


async def test_a_failed_fetch_is_marked_incomplete_so_it_is_tried_again(monkeypatch):
    async def search(*a, **kw):
        raise ConnectionError("embeddings down")

    monkeypatch.setattr(customer_memory, "search_episodes", search)
    pool = Pool([_session("OLD1", "older issue", "y", 21)])
    memory = await _load(pool, embedder=object())
    assert "older issue" in memory.note and not memory.complete
    assert not (await _load(Pool([], returning=True, fail=("sessions",)))).complete


async def test_not_knowing_whether_the_customer_is_returning_is_an_error_not_a_new_customer():
    with pytest.raises(ConnectionError):
        await _load(Pool(fail=("exists",)))


async def test_the_window_and_the_number_of_sessions_come_from_settings():
    pool = Pool([], returning=True)
    await _load(pool, history_days=3, history_max_sessions=2)
    assert pool.queries[1][1] == ("C1", 3, 3)


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
