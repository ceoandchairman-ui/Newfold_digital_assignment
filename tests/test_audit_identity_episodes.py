"""E5-5 audit persistence, E7-3 state hygiene, E3-1 name verification, E2-5/E6-1 episodes."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from fakes import World, install_repo, make_tools
from support_assistant.agents.llm import TemplateLLM
from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.audit import BackgroundAudit, PgAuditSink
from support_assistant.backend import app as backend
from support_assistant.context import summarize_session
from support_assistant.orchestrator import service as orch_service
from support_assistant.orchestrator.client import InProcessAgentClient
from support_assistant.orchestrator.graph import build_graph
from support_assistant.services.gateway_sim import SimulatedGateway


# --- the per-turn audit list through a real pause and resume (E5-5) -----------------------------------------
@pytest.fixture
def graph_env(monkeypatch):
    world = World()
    install_repo(monkeypatch, world)
    tools = make_tools(world, SimulatedGateway())
    reg = StaticRegistry(tools)
    client = InProcessAgentClient(
        registries={"director": reg, "evaluator": reg, "troubleshooter": reg},
        llms={"director": TemplateLLM(), "troubleshooter": TemplateLLM()},
    )

    async def pay(cid, bill, amount, key):
        return await tools["process_payment"](
            customer_id=cid, bill_id=bill, amount=amount, idempotency_key=key
        )

    return build_graph(client, pay, MemorySaver())


def _turn(text: str) -> dict[str, Any]:
    return {
        "customer_id": "C1002",
        "session_id": "S",
        "identity_verified": True,
        "route_hint": "BILLING",
        "messages": [{"role": "user", "content": text}],
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
        "audit": None,
    }


async def test_each_turn_records_its_own_tool_calls_exactly_once(graph_env):
    cfg = {"configurable": {"thread_id": "S"}}
    paused = await graph_env.ainvoke(_turn("pay my bill"), cfg)
    first = [(e["agent"], e["tool"]) for e in paused["audit"]]
    assert ("director", "get_bills") in first and ("evaluator", "check_recent_payments") in first

    resumed = await graph_env.ainvoke(
        Command(resume={"approved": True}, update={"audit": None}), cfg
    )
    second = [(e["agent"], e["tool"]) for e in resumed["audit"]]
    assert second == [("orchestrator", "process_payment")]  # the paused entries are not repeated


async def test_a_new_turn_starts_a_fresh_audit_list(graph_env):
    cfg = {"configurable": {"thread_id": "S"}}
    await graph_env.ainvoke(_turn("how much data have I used"), cfg)
    out = await graph_env.ainvoke(_turn("how much data have I used"), cfg)
    assert [e["tool"] for e in out["audit"]] == ["get_account_summary"]


# --- the sink and the background writer -------------------------------------------------------------------------
class ManyCursor:
    def __init__(self, log):
        self.log = log

    async def executemany(self, sql, rows):
        self.log.append((sql, rows))


class ManyConn:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def cursor(self):
        if self.fail:
            raise ConnectionError("db down")
        return ManyCursor(self.log)

    @asynccontextmanager
    async def transaction(self):
        yield


class ManyPool:
    def __init__(self, fail=False):
        self.log: list = []
        self.fail = fail

    @asynccontextmanager
    async def connection(self):
        yield ManyConn(self.log, self.fail)


async def test_the_sink_writes_one_row_per_entry():
    pool = ManyPool()
    await PgAuditSink(pool).write("S", "C1", [{"agent": "a", "tool": "t", "ok": True}] * 2)
    sql, rows = pool.log[0]
    assert "INSERT INTO audit_events" in sql and rows == [("S", "C1", "a", "t", True)] * 2


async def test_background_writes_happen_later_and_failures_never_raise():
    ok_pool, bad_pool = ManyPool(), ManyPool(fail=True)
    good, bad = BackgroundAudit(PgAuditSink(ok_pool)), BackgroundAudit(PgAuditSink(bad_pool))
    good.submit("S", "C1", [{"agent": "a", "tool": "t", "ok": True}])
    bad.submit("S", "C1", [{"agent": "a", "tool": "t", "ok": True}])
    good.submit("S", "C1", [])  # nothing to write: no task
    await good.drain()
    await bad.drain()  # the failure is logged, not raised
    assert len(ok_pool.log) == 1


# --- state hygiene (E7-3) -------------------------------------------------------------------------------------------
def test_oversized_state_is_measured():
    small = orch_service.check_state_size({"reply": "hi"}, "S")
    big = orch_service.check_state_size({"blob": "x" * (orch_service.MAX_STATE_BYTES + 10)}, "S")
    assert small < orch_service.MAX_STATE_BYTES < big


# --- backend: name verification, episodes, session end (E3-1, E2-5, E6-1) ------------------------------------------
class Cur:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class Conn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params=None):
        return Cur(self.rows)

    @asynccontextmanager
    async def transaction(self):
        yield


class Pool:
    def __init__(self, rows):
        self.rows = rows

    @asynccontextmanager
    async def connection(self):
        yield Conn(self.rows)


class KV:
    def __init__(self):
        self.data: dict[str, Any] = {}

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v, ex=None):
        self.data[k] = v

    async def delete(self, k):
        self.data.pop(k, None)


class Memory:
    def __init__(self, messages):
        self.messages = messages

    async def load(self, cid, sid):
        return self.messages

    async def append(self, cid, sid, new):
        return new


@pytest.fixture
def api(monkeypatch):
    calls: list[dict] = []
    saved: list[dict] = []

    def orch(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    async def fake_save(pool, embedder, **kw):
        saved.append(kw)
        return True

    monkeypatch.setattr(backend, "save_episode", fake_save)
    backend.app.state.redis = KV()
    backend.app.state.pool = Pool(
        [{"id": "C1", "full_name": "Ada Lovelace", "date_of_birth": date(1990, 5, 1)}]
    )
    backend.app.state.memory = Memory([])
    backend.app.state.rewrite = None
    backend.app.state.budget = None
    backend.app.state.embedder = object()
    backend.app.state.s = type("S", (), {"embedding_model": "m"})()
    backend.app.state.http = httpx.AsyncClient(
        base_url="http://o", transport=httpx.MockTransport(orch)
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://t")
    return client, calls, saved


async def test_verify_by_name_and_date_of_birth(api):
    client, _, _ = api
    r = await client.post(
        "/verify",
        json={"session_id": "S1", "full_name": "ada lovelace ", "date_of_birth": "1990-05-01"},
    )
    assert r.status_code == 200


async def test_an_ambiguous_name_is_refused_with_the_same_answer(api):
    client, _, _ = api
    twin = {"id": "C9", "full_name": "Ada Lovelace", "date_of_birth": date(1990, 5, 1)}
    backend.app.state.pool = Pool([backend.app.state.pool.rows[0], twin])
    r = await client.post(
        "/verify",
        json={"session_id": "S1", "full_name": "Ada Lovelace", "date_of_birth": "1990-05-01"},
    )
    assert r.status_code == 401


async def test_exactly_one_identifier_is_required(api):
    client, _, _ = api
    both = {"session_id": "S1", "phone": "+1", "full_name": "Ada", "date_of_birth": "1990-05-01"}
    neither = {"session_id": "S1", "date_of_birth": "1990-05-01"}
    assert (await client.post("/verify", json=both)).status_code == 422
    assert (await client.post("/verify", json=neither)).status_code == 422


async def test_ending_a_session_stores_an_episode_and_closes_the_session(api):
    client, _, saved = api
    await client.post(
        "/verify",
        json={"session_id": "S1", "full_name": "Ada Lovelace", "date_of_birth": "1990-05-01"},
    )
    backend.app.state.memory = Memory(
        [
            {"role": "user", "content": "my data is slow"},
            {"role": "assistant", "content": "Reset the APN settings."},
        ]
    )
    r = await client.post("/end", json={"session_id": "S1", "outcome": "resolved"})
    assert r.json() == {"stored": True} and saved[0]["outcome"] == "resolved"
    assert "my data is slow" in saved[0]["summary"]
    assert (
        await client.post("/chat", json={"session_id": "S1", "message": "hi"})
    ).status_code == 401


def test_session_summary_is_deterministic_and_bounded():
    msgs = [{"role": "user", "content": "q" * 1000}, {"role": "assistant", "content": "a" * 1000}]
    s = summarize_session(msgs)
    assert s.startswith("Customer asked:") and len(s) <= 600 and summarize_session([]) == ""


async def test_concurrent_background_writes_all_complete():
    pool = ManyPool()
    audit = BackgroundAudit(PgAuditSink(pool))
    for i in range(20):
        audit.submit(f"S{i}", "C1", [{"agent": "a", "tool": "t", "ok": True}])
    await asyncio.sleep(0)
    await audit.drain()
    assert len(pool.log) == 20
