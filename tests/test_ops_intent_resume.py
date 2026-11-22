"""E0-5 correlation ids, E4-3 intent classification, E4-9 cross-session approval, E8-8 supervisor,
E7-2 retention job."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest
import structlog
from fastapi import FastAPI

from support_assistant import jobs
from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.intent import KeywordIntentClassifier, LlmIntentClassifier
from support_assistant.agents.llm import TemplateLLM
from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.backend import app as backend
from support_assistant.contracts import AgentRequest
from support_assistant.mcp_servers.main import supervise
from support_assistant.observability import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    current_request_id,
    outgoing_headers,
)


# --- correlation ids (E0-5) ---------------------------------------------------------------------------------
def _app_echoing_request_id() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/x")
    async def x() -> dict[str, Any]:
        return {
            "rid": current_request_id(),
            "forward": outgoing_headers(),
            "log": structlog.contextvars.get_contextvars().get("request_id"),
        }

    return app


async def test_an_incoming_x_request_id_is_not_trusted_for_correlation():
    """Correlation is W3C traceparent now (test_otel.py); a client-chosen id is ignored."""
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app_echoing_request_id()), base_url="http://t"
    )
    r = await client.get("/x", headers={REQUEST_ID_HEADER: "abc123"})
    assert r.json()["rid"] != "abc123" and r.headers[REQUEST_ID_HEADER] == r.json()["rid"]


async def test_a_request_id_is_generated_when_missing_and_cleared_afterwards():
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app_echoing_request_id()), base_url="http://t"
    )
    r = await client.get("/x")
    assert len(r.json()["rid"]) == 32 and r.headers[REQUEST_ID_HEADER] == r.json()["rid"]
    assert current_request_id() == ""  # nothing leaks between requests
    assert outgoing_headers() == {}  # propagation is done by OTel instrumentation (traceparent)


# --- intent classification (E4-3) ---------------------------------------------------------------------------
class Replying:
    def __init__(self, text: str, fail: bool = False):
        self.text, self.fail = text, fail

    async def respond(self, instructions, facts, user_message):
        if self.fail:
            raise RuntimeError("model down")
        return self.text


async def test_a_valid_confident_model_answer_is_used():
    c = LlmIntentClassifier(Replying('Sure: {"kind": "billing", "confidence": 0.9}'))
    assert (await c.classify("pay")).kind == "billing"  # the model, not the keyword, decides


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '{"kind": "refund_everyone", "confidence": 0.99}',
        '{"kind": "billing", "confidence": 3}',
        '{"kind": "billing", "confidence": 0.2}',
    ],
)
async def test_bad_or_unsure_model_answers_fall_back_to_keywords(reply):
    assert (
        await LlmIntentClassifier(Replying(reply)).classify("please pay my bill")
    ).kind == "payment"


async def test_a_model_outage_falls_back_to_keywords():
    assert (
        await LlmIntentClassifier(Replying("", fail=True)).classify("my data usage")
    ).kind == "account"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("pay my bill", "payment"),
        ("how much data left", "account"),
        ("what do I owe", "billing"),
        ("hello", "other"),
    ],
)
async def test_keyword_classifier(text, kind):
    assert (await KeywordIntentClassifier().classify(text)).kind == kind


async def test_the_director_follows_the_classifier():
    async def get_bills(customer_id):
        return {
            "ok": True,
            "bills": [{"id": "B1", "amount_due": "5", "due_date": "d", "status": "ISSUED"}],
        }

    req = AgentRequest(
        customer_id="C1",
        session_id="S",
        identity_verified=True,
        messages=[{"role": "user", "content": "settle it"}],
    )
    h = Harness("director", roles.DIRECTOR_TOOLS, StaticRegistry({"get_bills": get_bills}))
    says_pay = LlmIntentClassifier(Replying('{"kind": "payment", "confidence": 0.95}'))
    res = await roles.director(req, h, TemplateLLM(), says_pay)
    assert (
        res.proposed_action == {"type": "pay_bill", "bill_id": "B1", "amount": "5"}
        and res.reply == ""
    )


# --- approvals across sessions (E4-9) --------------------------------------------------------------------------------
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
    async def load(self, cid, sid):
        return []

    async def append(self, cid, sid, new):
        return new


class Pool:
    @asynccontextmanager
    async def connection(self):
        class C:
            async def execute(self, sql, params=None):
                class R:
                    async def fetchone(self_inner):
                        return {"id": "C1", "date_of_birth": date(1990, 5, 1)}

                return R()

        yield C()


@pytest.fixture
def api():
    calls: list[tuple[str, dict]] = []

    def orch(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/invoke":
            return httpx.Response(
                200, json={"reply": "", "pending_approval": {"question": "Pay 5 for B1?"}}
            )
        return httpx.Response(200, json={"reply": "Your payment of 5 was successful."})

    backend.app.state.redis = KV()
    backend.app.state.pool = Pool()
    backend.app.state.memory = Memory()
    backend.app.state.rewrite = None
    backend.app.state.budget = None
    backend.app.state.embedder = None
    backend.app.state.http = httpx.AsyncClient(
        base_url="http://o", transport=httpx.MockTransport(orch)
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend.app), base_url="http://t"
    ), calls


async def _verify(client, session):
    await client.post(
        "/verify", json={"session_id": session, "phone": "+1", "date_of_birth": "1990-05-01"}
    )


async def test_a_pending_payment_follows_the_customer_into_a_new_session(api):
    client, calls = api
    await _verify(client, "OLD")
    await client.post("/chat", json={"session_id": "OLD", "message": "pay my bill"})
    await _verify(client, "NEW")  # the customer comes back later in a fresh session
    r = await client.post("/chat", json={"session_id": "NEW", "message": "hi again"})
    assert r.json()["pending_approval"] == {"question": "Pay 5 for B1?"}
    assert [p for p, _ in calls] == ["/invoke"]  # no new graph run was started

    done = await client.post("/approve", json={"session_id": "NEW", "approved": True})
    assert calls[-1] == ("/resume", {"session_id": "OLD", "approved": True})  # the paused run
    assert "successful" in done.json()["reply"]
    assert "pending:C1" not in backend.app.state.redis.data


async def test_without_a_pending_payment_approve_resumes_the_current_session(api):
    client, calls = api
    await _verify(client, "S1")
    await client.post("/approve", json={"session_id": "S1", "approved": False})
    assert calls[-1] == ("/resume", {"session_id": "S1", "approved": False})


# --- MCP supervisor (E8-8) -------------------------------------------------------------------------------------------
async def test_a_crashing_server_is_restarted_with_growing_backoff():
    runs, sleeps = [], []

    async def flaky():
        runs.append(1)
        raise RuntimeError("crash")

    async def fake_sleep(s):
        sleeps.append(s)

    await supervise("database", flaky, sleep=fake_sleep, max_restarts=3, max_backoff_s=5)
    assert len(runs) == 4 and sleeps == [2, 4, 5, 5]


async def test_shutdown_cancellation_stops_supervision():
    async def forever():
        await asyncio.sleep(3600)

    task = asyncio.create_task(supervise("comms", forever))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_one_server_crashing_does_not_stop_the_others():
    healthy_ran = asyncio.Event()

    async def crashing():
        raise RuntimeError("crash")

    async def healthy():
        healthy_ran.set()
        await asyncio.sleep(3600)

    async def no_sleep(_):
        await asyncio.sleep(0)

    t1 = asyncio.create_task(supervise("db", crashing, sleep=no_sleep, max_restarts=5))
    t2 = asyncio.create_task(supervise("comms", healthy))
    await t1  # the crashing one gives up after its restarts...
    assert healthy_ran.is_set() and not t2.done()  # ...the healthy one keeps serving
    t2.cancel()


# --- retention job (E7-2) ------------------------------------------------------------------------------------------------
async def test_the_prune_job_reports_what_it_deleted(monkeypatch):
    seen = {}

    async def fake_prune(pool, days):
        seen["days"] = days
        return 7

    monkeypatch.setattr(jobs, "prune_sessions", fake_prune)
    assert await jobs.run_prune(object(), 90) == 7 and seen == {"days": 90}
