"""Retries, idempotency, locking and transaction safety, plus one whole turn checked for log leaks.

These use fakes. What only a real database can prove (the row lock under true concurrency) is in
tests/integration/test_postgres.py, which needs Postgres.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from psycopg import errors as pg_errors

from fakes import World, install_repo, make_tools
from support_assistant import indexer, resilience, worker
from support_assistant.agents.llm import TemplateLLM
from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.audit import BackgroundAudit
from support_assistant.config import Settings
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.locks import RunLock
from support_assistant.logs import conversation, pseudonym
from support_assistant.memory import SessionMemory
from support_assistant.notifications import queue_message_id
from support_assistant.orchestrator import service as orchestrator
from support_assistant.orchestrator.client import HttpAgentClient, InProcessAgentClient
from support_assistant.orchestrator.graph import build_graph
from support_assistant.retrieval_client import HttpRetrievalClient
from support_assistant.services.gateway_sim import SimulatedGateway
from support_assistant.telemetry import RedactSessionInUrls
from test_indexer_chain import FakePool as KbPool
from test_security import STRONG, _setup


# --- the retry policy ---------------------------------------------------------------------------
def test_backoff_doubles_up_to_a_cap_and_is_fully_jittered():
    top = [resilience.backoff_delay(n, 1.0, 10.0, rand=lambda: 1.0) for n in range(1, 7)]
    assert top == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
    assert resilience.backoff_delay(5, 1.0, 10.0, rand=lambda: 0.0) == 0.0  # anywhere from zero


async def test_only_the_named_errors_are_retried_and_the_last_one_is_raised(log_lines):
    calls: list[int] = []

    @resilience.retrying(
        "op", attempts=3, base=0.001, cap=0.002, retry_on=lambda e: isinstance(e, KeyError)
    )
    async def flaky(error: type[Exception]) -> None:
        calls.append(1)
        raise error("x")

    with pytest.raises(KeyError):
        await flaky(KeyError)
    assert len(calls) == 3
    scheduled = [x for x in log_lines() if x["event"] == "retry_scheduled"]
    assert [x["attempt"] for x in scheduled] == [1, 2] and scheduled[0]["error.type"] == "KeyError"
    calls.clear()
    with pytest.raises(ValueError):
        await flaky(ValueError)
    assert len(calls) == 1


def test_only_failures_that_changed_nothing_count_as_safe():
    assert resilience.never_sent(httpx.ConnectError("x"))
    assert not resilience.never_sent(httpx.ReadTimeout("x"))  # the server may be working on it
    assert resilience.is_db_conflict(pg_errors.DeadlockDetected())
    assert resilience.is_db_conflict(pg_errors.SerializationFailure())
    assert not resilience.is_db_conflict(pg_errors.OperationalError())  # commit may have happened


@pytest.mark.parametrize(("error", "attempts"), [(httpx.ReadTimeout, 1), (httpx.ConnectError, 3)])
async def test_an_agent_run_is_repeated_only_if_it_never_reached_the_agent(error, attempts):
    seen: list[int] = []

    def fail(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        raise error("boom", request=request)

    settings = Settings(database_url="postgresql://x", redis_url="redis://x")
    client = HttpAgentClient(settings, transport=httpx.MockTransport(fail))
    req = AgentRequest(customer_id="C1", session_id="S", identity_verified=True)
    with pytest.raises(ExternalServiceUnavailable):
        await client.run("troubleshooter", req)
    assert len(seen) == attempts


# --- conversation memory --------------------------------------------------------------------------
class MemoryDb:
    """Records the SQL, and fails the first `conflicts` transactions with a deadlock."""

    def __init__(self, conflicts: int = 0) -> None:
        self.sql: list[str] = []
        self.conflicts, self.saved = conflicts, None

    @asynccontextmanager
    async def connection(self):
        db = self

        class Cursor:
            async def fetchone(self):
                return {"messages": [{"role": "user", "content": "earlier"}]}

        class Conn:
            async def execute(self, sql, params=None):
                text = " ".join(sql.split())
                db.sql.append(text)
                if text.startswith("UPDATE"):
                    if db.conflicts:
                        db.conflicts -= 1
                        raise pg_errors.DeadlockDetected()
                    db.saved = params[0].obj
                return Cursor()

            @asynccontextmanager
            async def transaction(self):
                yield

        yield Conn()


class NoRedis:
    async def set(self, *a: Any, **k: Any) -> None:
        return None


async def test_a_first_write_creates_the_row_before_locking_it():
    db = MemoryDb()
    await SessionMemory(db, NoRedis(), window=20, ttl_s=60).append("C1", "S", [{"role": "x"}])
    kinds = [q.split()[0] for q in db.sql]
    assert kinds == ["INSERT", "SELECT", "UPDATE"]  # a missing row cannot be locked
    assert "ON CONFLICT (session_id) DO NOTHING" in db.sql[0] and "FOR UPDATE" in db.sql[1]
    assert [m["role"] for m in db.saved] == ["user", "x"]


async def test_a_deadlock_reruns_the_whole_memory_transaction_once():
    db = MemoryDb(conflicts=1)
    await SessionMemory(db, NoRedis(), window=20, ttl_s=60).append("C1", "S", [{"role": "x"}])
    assert [q.split()[0] for q in db.sql] == ["INSERT", "SELECT", "UPDATE"] * 2
    assert len(db.saved) == 2  # the message was added once, not once per attempt


# --- payment idempotency ----------------------------------------------------------------------------
class Proposes:
    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        if req.context.get("mode") == "route":
            return AgentResponse(route="BILLING")
        if agent == "evaluator":
            return AgentResponse(verdict="approve")
        return AgentResponse(
            proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "5.00"}
        )


def _turn(session: str = "S1") -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": session,
        "identity_verified": True,
        "messages": [{"role": "user", "content": "pay my bill"}],
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
    }


async def test_each_proposal_has_its_own_payment_key_so_a_decline_can_be_tried_again():
    keys: list[str] = []

    async def declined(customer_id, bill_id, amount, key):
        keys.append(key)
        return {"ok": False, "error": "payment_declined", "message": "Declined."}

    graph = build_graph(Proposes(), declined, MemorySaver())
    cfg = {"configurable": {"thread_id": "S1"}}
    for _ in range(2):
        paused = await graph.ainvoke(_turn(), cfg)
        proposed = paused["__interrupt__"][0].value["action"]["proposal_id"]
        await graph.ainvoke(Command(resume={"approved": True}), cfg)
        assert keys[-1] == f"S1:B1:{proposed}"  # the id saved before the pause is the one used
    assert keys[0] != keys[1]


class LockRedis:
    def __init__(self, renew: Any = 1, release_fails: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.renewals, self.renew, self.release_fails = 0, renew, release_fails

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def eval(self, script, numkeys, key, token, *args):
        if "pexpire" in script:
            self.renewals += 1
            return self.renew
        if self.release_fails:
            raise ConnectionError("redis gone")
        self.data.pop(key, None)
        return 1


async def test_approving_twice_pays_once_and_the_second_answer_says_so():
    paid: list[str] = []

    async def pay(customer_id, bill_id, amount, key):
        paid.append(key)
        return {"ok": True}

    class Sink:
        async def write(self, *a: Any) -> None:
            return None

    saved = dict(orchestrator.app.state._state)
    try:
        orchestrator.app.state.graph = build_graph(Proposes(), pay, MemorySaver())
        orchestrator.app.state.lock = RunLock(LockRedis())
        orchestrator.app.state.audit = BackgroundAudit(Sink())
        orchestrator.app.state.turn_timeout = None
        body = orchestrator.InvokeBody(
            customer_id="C1",
            session_id="S9",
            identity_verified=True,
            messages=[{"role": "user", "content": "pay my bill"}],
        )
        assert (await orchestrator.invoke(body)).pending_approval
        yes = orchestrator.ResumeBody(session_id="S9", approved=True)
        first = await orchestrator.resume(yes)
        second = await orchestrator.resume(yes)
    finally:
        orchestrator.app.state._state.clear()
        orchestrator.app.state._state.update(saved)
    assert "successful" in first.reply and len(paid) == 1
    assert second.reply == orchestrator.NOTHING_PENDING_REPLY


# --- the run lock ---------------------------------------------------------------------------------
async def test_a_long_run_keeps_renewing_its_lock_and_stops_when_it_ends():
    redis = LockRedis()
    async with RunLock(redis, ttl_ms=60).hold("S1"):
        await asyncio.sleep(0.15)
    during = redis.renewals
    await asyncio.sleep(0.08)
    assert during >= 2 and redis.renewals == during and redis.data == {}


async def test_losing_the_lock_is_logged_loudly(log_lines):
    async with RunLock(LockRedis(renew=0), ttl_ms=30).hold("S1"):
        await asyncio.sleep(0.08)
    assert [x["level"] for x in log_lines() if x["event"] == "run_lock_lost"] == ["error"]


async def test_a_failed_release_never_replaces_the_runs_own_result():
    async with RunLock(LockRedis(release_fails=True)).hold("S1"):
        done = True
    assert done
    with pytest.raises(KeyError):  # the run's own error, not the release's
        async with RunLock(LockRedis(release_fails=True)).hold("S2"):
            raise KeyError("run failed")


# --- repeated requests ----------------------------------------------------------------------------
async def test_a_message_id_reused_for_a_different_message_is_refused_not_answered():
    client, redis, sent = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"

    async def chat(message: str) -> httpx.Response:
        body = {"session_id": STRONG, "message": message, "message_id": "msg-0001"}
        return await client.post("/chat", json=body)

    first, same, other = [await chat(m) for m in ("what do I owe", "what do I owe", "pay it")]
    assert first.status_code == same.status_code == 200 and first.json() == same.json()
    assert other.status_code == 409 and len(sent) == 1


# --- notifications --------------------------------------------------------------------------------
def test_a_repeat_of_one_event_has_one_id_and_a_new_event_has_another():
    a = queue_message_id("C1", "sms", "payment_received", "tx-1")
    assert a == queue_message_id("C1", "sms", "payment_received", "tx-1")
    assert a != queue_message_id("C1", "sms", "payment_received", "tx-2")
    # with no event id nothing is merged: two real payments must both be announced
    none = {queue_message_id("C1", "sms", "payment_received", None) for _ in range(2)}
    assert len(none) == 2


class SentRedis:
    def __init__(self, down: bool = False) -> None:
        self.keys: dict[str, int] = {}
        self.down = down

    async def exists(self, key):
        if self.down:
            raise ConnectionError("redis")
        return key in self.keys

    async def set(self, key, value, ex):
        if self.down:
            raise ConnectionError("redis")
        self.keys[key] = ex


class Delivery:
    """Runs one message through the worker, recording pauses, attempts and dead letters."""

    def __init__(self, *outcomes, redis: SentRedis | None = None, dlq_down: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.handled: list[str] = []
        self.dead: list[bytes] = []
        self.pauses: list[float] = []
        self.sent = worker.SentLog(redis or SentRedis())
        self.dlq_down = dlq_down

    async def handle(self, raw: str) -> worker.Outcome:
        self.handled.append(raw)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def dead_letter(self, value, headers) -> None:
        if self.dlq_down:
            raise ConnectionError("broker")
        self.dead.append(value)

    async def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)

    async def run(self, message_id: str = "m1") -> worker.Outcome:
        return await worker.process_message(
            b"body",
            [("message_id", message_id.encode())],
            self.handle,
            self.dead_letter,
            self.sent,
            sleep=self.pause,
        )


async def test_a_repeated_message_is_sent_once():
    redis = SentRedis()
    first = Delivery(worker.Outcome.COMPLETE, redis=redis)
    again = Delivery(worker.Outcome.COMPLETE, redis=redis)
    assert await first.run() is await again.run() is worker.Outcome.COMPLETE
    assert first.handled == ["body"] and again.handled == []
    assert redis.keys == {"notification:sent:m1": worker.DEDUPE_TTL_S}


async def test_redis_being_down_never_blocks_a_notification():
    d = Delivery(worker.Outcome.COMPLETE, redis=SentRedis(down=True))
    assert await d.run() is worker.Outcome.COMPLETE
    assert d.handled == ["body"] and d.dead == []


async def test_a_crash_is_retried_with_a_pause_until_it_works():
    d = Delivery(ConnectionError("db"), worker.Outcome.RETRY, worker.Outcome.COMPLETE)
    assert await d.run() is worker.Outcome.COMPLETE
    assert len(d.handled) == 3 and d.dead == []
    assert len(d.pauses) == 2 and all(p >= worker.RETRY_BASE_S for p in d.pauses)


async def test_retries_are_bounded_then_the_message_is_dead_lettered():
    d = Delivery(worker.Outcome.RETRY)
    assert await d.run() is worker.Outcome.DEAD_LETTER
    assert len(d.handled) == worker.MAX_ATTEMPTS and len(d.pauses) == worker.MAX_ATTEMPTS - 1
    assert d.dead == [b"body"]


async def test_a_bad_message_is_dead_lettered_at_once_and_never_marked_sent():
    redis = SentRedis()
    d = Delivery(worker.Outcome.DEAD_LETTER, redis=redis)
    assert await d.run() is worker.Outcome.DEAD_LETTER
    assert d.dead == [b"body"] and d.pauses == [] and redis.keys == {}


async def test_a_failed_dead_letter_raises_so_the_offset_is_not_committed():
    with pytest.raises(ConnectionError):
        await Delivery(worker.Outcome.DEAD_LETTER, dlq_down=True).run()


# --- re-embedding ---------------------------------------------------------------------------------
async def test_a_new_embedding_model_embeds_every_chunk_again(monkeypatch):
    pool = KbPool()

    async def register(_):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", register)

    async def index(model: str) -> dict:
        chain = indexer.build_index_chain(pool, LcEmbeddings(FakeEmbedder(dim=8)), model)
        return await chain.ainvoke({"source": "kb.md", "text": "# G\n\nRestart.\n"})

    first = await index("model-a")
    assert first["embedded"] >= 1 and (await index("model-a"))["embedded"] == 0  # unchanged
    again = await index("model-b")  # vectors of another model are never reused
    assert again["embedded"] == first["embedded"]
    assert {r["model"] for r in pool.store["rows"].values()} == {"model-b"}


# --- correlation and leaks ------------------------------------------------------------------------
async def test_the_retrieval_client_passes_the_hashed_session_and_turn_on():
    seen: list[httpx.Headers] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"results": []})

    c = HttpRetrievalClient("http://retrieval", transport=httpx.MockTransport(answer))
    with conversation(STRONG, "C1", "turn-0001"):
        await c.search("apn")
    assert seen[0]["x-session-ref"] == pseudonym(STRONG) and seen[0]["x-turn-id"] == "turn-0001"
    assert STRONG not in str(dict(seen[0]))


def test_the_access_log_never_prints_a_session_id_from_a_url():
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1", "GET", f"/uploads?session_id={STRONG}&x=1", "1.1", 200),
        None,
    )
    assert RedactSessionInUrls().filter(record)
    assert STRONG not in record.getMessage()
    assert "session_id=[redacted]&x=1" in record.getMessage()


async def test_a_whole_payment_turn_is_traceable_and_leaks_nothing(monkeypatch, log_lines):
    world, gateway = World(), SimulatedGateway()
    install_repo(monkeypatch, world)
    tools = make_tools(world, gateway)
    registry = StaticRegistry(tools)
    agents = InProcessAgentClient(
        registries={"director": registry, "evaluator": registry},
        llms={"director": TemplateLLM()},
    )

    async def pay(customer_id, bill_id, amount, key):
        return await tools["process_payment"](
            customer_id=customer_id, bill_id=bill_id, amount=amount, idempotency_key=key
        )

    class Sink:
        async def write(self, *a: Any) -> None:
            return None

    secret_words = "please pay my bill, my pin is 99881"
    saved = dict(orchestrator.app.state._state)
    try:
        orchestrator.app.state.graph = build_graph(agents, pay, MemorySaver())
        orchestrator.app.state.lock = RunLock(LockRedis())
        orchestrator.app.state.audit = BackgroundAudit(Sink())
        orchestrator.app.state.turn_timeout = None
        body = orchestrator.InvokeBody(
            customer_id="C1002",
            session_id=STRONG,
            identity_verified=True,
            messages=[{"role": "user", "content": secret_words}],
            turn_id="turn-e2e-1",
        )
        assert (await orchestrator.invoke(body)).pending_approval
    finally:
        orchestrator.app.state._state.clear()
        orchestrator.app.state._state.update(saved)

    lines = log_lines()
    everything = json.dumps(lines)
    for secret in (STRONG, "C1002", "99881", "my pin"):
        assert secret not in everything, secret
    events = {x["event"] for x in lines}
    assert {
        "graph_node_completed",
        "graph_branch",
        "route_decided",
        "billing_intent_classified",
        "payment_proposed",
        "payment_checked",
        "tool_call",
        "graph_run_completed",
    } <= events
    # every line of the run can be found by conversation and by turn
    for line in lines:
        assert line["gen_ai.conversation.id"] == pseudonym(STRONG), line["event"]
        assert line["conversation.turn.id"] == "turn-e2e-1", line["event"]
