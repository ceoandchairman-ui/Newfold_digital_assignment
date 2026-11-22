"""Backend endpoints (E3-1, E6-3), session memory (E2-6) and tenant scoping of every query (E5-2)."""

import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest

from support_assistant import repositories as repo
from support_assistant.backend import app as backend
from support_assistant.memory import SessionMemory


# --- shared fakes ---------------------------------------------------------------------------------------
class Cur:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class Conn:
    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    async def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))
        return Cur(self.rows)

    @asynccontextmanager
    async def transaction(self):
        yield


class Pool:
    def __init__(self, rows=None):
        self.rows, self.log = rows or [], []

    @asynccontextmanager
    async def connection(self):
        yield Conn(self.rows, self.log)


class KV:
    def __init__(self, fail=False):
        self.data: dict[str, Any] = {}
        self.fail = fail

    async def get(self, k):
        if self.fail:
            raise ConnectionError
        return self.data.get(k)

    async def set(self, k, v, ex=None):
        if self.fail:
            raise ConnectionError
        self.data[k] = v

    async def delete(self, k):
        self.data.pop(k, None)


# --- tenant scoping: every customer-scoped query filters by customer_id (E5-2) ----------------
@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (repo.list_lines, ("C1",)),
        (repo.list_bills, ("C1",)),
        (repo.count_overdue_bills, ("C1", date(2026, 1, 1))),
        (repo.latest_contract_end, ("C1",)),
        (repo.list_devices, ("C1",)),
        (repo.recent_payment_count, ("C1", "B1", 10)),
        (repo.lock_bill, ("C1", "B1")),
        (repo.get_bill, ("C1", "B1")),
    ],
)
async def test_every_customer_scoped_query_filters_by_customer(fn, args):
    conn = Conn([{"id": "B1", "n": 0, "d": None}], [])
    try:
        await fn(conn, *args)
    except Exception:  # noqa: BLE001 - shape of the result does not matter here
        pass
    sql, params = conn.log[0]
    assert "customer_id = %s" in sql and "C1" in params  # the tenant is always bound


# --- session memory (E2-6) ---------------------------------------------------------------------------------
async def test_memory_reads_cache_first_then_postgres_and_refills():
    redis, pool = KV(), Pool([{"messages": [{"role": "user", "content": "hi"}]}])
    mem = SessionMemory(pool, redis, window=20, ttl_s=60)
    assert (await mem.load("C1", "S1"))[0]["content"] == "hi"  # miss: Postgres
    assert "sess:C1:S1" in redis.data  # refilled
    pool.log.clear()
    await mem.load("C1", "S1")
    assert pool.log == []  # hit: no Postgres query


async def test_memory_survives_a_redis_outage():
    mem = SessionMemory(
        Pool([{"messages": [{"role": "user", "content": "x"}]}]), KV(fail=True), 20, 60
    )
    assert (await mem.load("C1", "S1"))[0]["content"] == "x"


async def test_memory_window_is_bounded_and_keys_are_per_customer():
    msgs = [{"role": "user", "content": str(i)} for i in range(30)]
    redis = KV()
    mem = SessionMemory(Pool([{"messages": msgs}]), redis, window=5, ttl_s=60)
    out = await mem.append("C1", "S1", [{"role": "user", "content": "new"}])
    assert len(out) == 5 and out[-1]["content"] == "new"
    assert any(k.startswith("sess:C1:") for k in redis.data) and not any(
        "C2" in k for k in redis.data
    )


# --- backend endpoints (E3-1, E6-3) -------------------------------------------------------------------------
class FakeMemory:
    def __init__(self):
        self.appended: list[tuple] = []

    async def load(self, cid, sid):
        return []

    async def append(self, cid, sid, new):
        self.appended.append((cid, sid, new))
        return new


@pytest.fixture
def client():
    orch_calls: list[tuple[str, dict]] = []

    def orch(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        orch_calls.append((request.url.path, body))
        return httpx.Response(
            200, json={"reply": "ok", "pending_approval": None, "escalated": False}
        )

    backend.app.state.redis = KV()
    backend.app.state.pool = Pool([{"id": "C1", "date_of_birth": date(1990, 5, 1)}])
    backend.app.state.memory = FakeMemory()
    backend.app.state.rewrite = None
    backend.app.state.http = httpx.AsyncClient(
        base_url="http://orch", transport=httpx.MockTransport(orch)
    )
    transport = httpx.ASGITransport(app=backend.app)
    c = httpx.AsyncClient(transport=transport, base_url="http://test")
    c.orch_calls = orch_calls  # type: ignore[attr-defined]
    return c


async def test_verify_with_matching_phone_and_dob_then_chat_works(client):
    r = await client.post(
        "/verify", json={"session_id": "S1", "phone": "+1", "date_of_birth": "1990-05-01"}
    )
    assert r.status_code == 200
    r = await client.post("/chat", json={"session_id": "S1", "message": "how much data"})
    assert r.status_code == 200 and client.orch_calls[0][1]["customer_id"] == "C1"


async def test_wrong_dob_fails_with_the_same_answer_as_unknown_phone(client):
    wrong = await client.post(
        "/verify", json={"session_id": "S1", "phone": "+1", "date_of_birth": "2000-01-01"}
    )
    backend.app.state.pool = Pool([])  # unknown phone
    unknown = await client.post(
        "/verify", json={"session_id": "S1", "phone": "+9", "date_of_birth": "1990-05-01"}
    )
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()


async def test_chat_without_verification_never_reaches_the_orchestrator(client):
    r = await client.post("/chat", json={"session_id": "S1", "message": "hi"})
    assert r.status_code == 401 and client.orch_calls == []


async def test_troubleshooting_messages_are_routed_and_billing_ones_are_not(client):
    await client.post(
        "/verify", json={"session_id": "S1", "phone": "+1", "date_of_birth": "1990-05-01"}
    )
    await client.post("/chat", json={"session_id": "S1", "message": "my phone has no signal"})
    await client.post("/chat", json={"session_id": "S1", "message": "what is my bill"})
    assert [c[1]["route"] for c in client.orch_calls] == ["troubleshoot", "billing"]


async def test_chat_rejects_empty_and_oversized_messages(client):
    await client.post(
        "/verify", json={"session_id": "S1", "phone": "+1", "date_of_birth": "1990-05-01"}
    )
    assert (await client.post("/chat", json={"session_id": "S1", "message": ""})).status_code == 422
    assert (
        await client.post("/chat", json={"session_id": "S1", "message": "x" * 2001})
    ).status_code == 422


async def test_approve_resumes_and_records_the_reply(client):
    await client.post(
        "/verify", json={"session_id": "S1", "phone": "+1", "date_of_birth": "1990-05-01"}
    )
    r = await client.post("/approve", json={"session_id": "S1", "approved": True})
    assert r.status_code == 200 and client.orch_calls[-1][0] == "/resume"
    assert backend.app.state.memory.appended[-1][2][0]["role"] == "assistant"
