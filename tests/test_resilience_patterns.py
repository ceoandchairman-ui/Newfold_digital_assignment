"""Resilience patterns: a deadline and a fallback on every model call, a circuit breaker on every agent,
and a deadline on a whole turn that never cuts a payment short."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from support_assistant.agents.llm import ResilientLLM
from support_assistant.config import Settings
from support_assistant.contracts import AgentRequest
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.orchestrator import service
from support_assistant.orchestrator.client import HttpAgentClient


# --- a model call with a deadline and a fallback ------------------------------------------------------
class Model:
    def __init__(self, answer="", *, boom=False, delay=0.0):
        self.answer, self.boom, self.delay, self.calls = answer, boom, delay, 0

    async def respond(self, instructions, facts, user_message, history=""):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("429 quota")
        return self.answer


async def test_a_healthy_primary_answers_and_the_fallback_is_never_touched():
    primary, fallback = Model("primary"), Model("fallback")
    assert await ResilientLLM(primary, fallback, 1.0).respond("i", "f", "m") == "primary"
    assert fallback.calls == 0


async def test_a_throttled_primary_falls_back_instead_of_failing_the_turn():
    primary, fallback = Model(boom=True), Model("fallback")
    assert (
        await ResilientLLM(primary, fallback, 1.0, "planner").respond("i", "f", "m") == "fallback"
    )
    assert (primary.calls, fallback.calls) == (1, 1)


async def test_a_slow_primary_is_cut_off_at_the_deadline_and_the_fallback_answers():
    primary, fallback = Model("slow", delay=5), Model("fallback")
    out = await asyncio.wait_for(ResilientLLM(primary, fallback, 0.05).respond("i", "f", "m"), 2)
    assert out == "fallback"


async def test_with_no_fallback_the_error_reaches_the_caller_which_has_its_own_safe_path():
    with pytest.raises(RuntimeError):
        await ResilientLLM(Model(boom=True), None, 1.0).respond("i", "f", "m")
    with pytest.raises(TimeoutError):
        await ResilientLLM(Model(delay=5), None, 0.05).respond("i", "f", "m")


async def test_when_both_fail_the_error_is_raised_not_swallowed():
    with pytest.raises(RuntimeError):
        await ResilientLLM(Model(boom=True), Model(boom=True), 1.0).respond("i", "f", "m")


# --- a circuit breaker on every agent ----------------------------------------------------------------------
def _client(handler, **settings):
    s = Settings(database_url="postgresql://x", redis_url="redis://x", **settings)
    return HttpAgentClient(s, transport=httpx.MockTransport(handler))


def _req():
    return AgentRequest(customer_id="C1", session_id="S", identity_verified=True)


async def test_an_agent_that_keeps_failing_is_cut_off_without_waiting_for_more_timeouts():
    hits = []

    def handler(request):
        hits.append(request.url.host)
        return httpx.Response(503)

    client = _client(handler, agent_circuit_threshold=3, agent_circuit_cooldown_seconds=60)
    for _ in range(3):
        with pytest.raises(ExternalServiceUnavailable):
            await client.run("director", _req())
    before = len(hits)
    with pytest.raises(ExternalServiceUnavailable, match="circuit open"):
        await client.run("director", _req())
    assert len(hits) == before  # the open circuit made no network call


async def test_one_dead_agent_does_not_cut_off_the_others():
    def handler(request):
        if "agent-director" in request.url.host:
            return httpx.Response(503)
        return httpx.Response(200, json={"reply": "ok"})

    client = _client(handler, agent_circuit_threshold=2)
    for _ in range(2):
        with pytest.raises(ExternalServiceUnavailable):
            await client.run("director", _req())
    assert (await client.run("troubleshooter", _req())).reply == "ok"
    assert (await client.run("evaluator", _req())).reply == "ok"


async def test_a_client_error_never_opens_the_circuit():
    client = _client(
        lambda r: httpx.Response(422, json={"detail": "bad"}), agent_circuit_threshold=2
    )
    for _ in range(5):
        assert (await client.run("director", _req())).reply == "bad"  # still being called


async def test_after_the_cooldown_one_trial_call_closes_the_circuit_again():
    healthy = {"now": False}

    def handler(request):
        return (
            httpx.Response(200, json={"reply": "back"}) if healthy["now"] else httpx.Response(503)
        )

    client = _client(handler, agent_circuit_threshold=2, agent_circuit_cooldown_seconds=0.05)
    for _ in range(2):
        with pytest.raises(ExternalServiceUnavailable):
            await client.run("director", _req())
    healthy["now"] = True
    await asyncio.sleep(0.08)
    assert (await client.run("director", _req())).reply == "back"
    assert (await client.run("director", _req())).reply == "back"


async def test_a_transport_failure_counts_towards_the_circuit_too():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("no route")

    client = _client(handler, agent_circuit_threshold=2, agent_circuit_cooldown_seconds=60)
    for _ in range(2):
        with pytest.raises(ExternalServiceUnavailable):
            await client.run("evaluator", _req())
    n = len(calls)
    with pytest.raises(ExternalServiceUnavailable, match="circuit open"):
        await client.run("evaluator", _req())
    assert len(calls) == n


# --- a deadline on a whole turn ------------------------------------------------------------------------------
class Lock:
    def __init__(self):
        self.held = self.released = 0

    @asynccontextmanager
    async def hold(self, session_id):
        self.held += 1
        try:
            yield
        finally:
            self.released += 1


class Audit:
    def submit(self, *a, **k):
        pass


class Graph:
    def __init__(self, delay=0.0, out=None):
        self.delay, self.out = delay, out or {"reply": "done", "customer_id": "C1"}

    async def ainvoke(self, graph_input, cfg, **kw):
        await asyncio.sleep(self.delay)
        return self.out


@pytest.fixture
def orch(monkeypatch):
    lock = Lock()
    for name, value in (("lock", lock), ("audit", Audit()), ("graph", Graph())):
        monkeypatch.setattr(service.app.state, name, value, raising=False)
    return lock


async def test_a_turn_that_overruns_its_deadline_is_cancelled_and_the_customer_is_told_to_retry(
    orch,
):
    service.app.state.graph = Graph(delay=5)
    with pytest.raises(ExternalServiceUnavailable):
        await service._run("S", {}, deadline_s=0.05)
    assert (
        orch.held == orch.released == 1
    )  # the session lock is released, so a retry is not blocked


async def test_a_turn_within_its_deadline_returns_normally(orch):
    out = await service._run("S", {}, deadline_s=1.0)
    assert out.reply == "done"


async def test_with_no_deadline_a_slow_run_is_left_alone(orch):
    service.app.state.graph = Graph(delay=0.1)
    assert (await service._run("S", {}, deadline_s=None)).reply == "done"


async def test_only_a_new_message_gets_a_deadline_never_a_resume_that_may_pay(monkeypatch):
    seen: dict[str, float | None] = {}

    async def fake_run(session_id, graph_input, deadline_s=None):
        seen[session_id] = deadline_s
        return service.TurnResult(reply="ok")

    monkeypatch.setattr(service, "_run", fake_run)
    monkeypatch.setattr(service.app.state, "turn_timeout", 45.0, raising=False)
    await service.invoke(
        service.InvokeBody(customer_id="C1", session_id="NEW", identity_verified=True, messages=[])
    )
    await service.resume(service.ResumeBody(session_id="PAUSED", approved=True))
    assert seen == {"NEW": 45.0, "PAUSED": None}
