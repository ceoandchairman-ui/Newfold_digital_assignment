"""Security hardening of the fat backend: session ids, identity-guessing lockout, rate limits,
idempotent retries, load shedding, request limits and response headers."""

import asyncio
import json
from datetime import date

import httpx
import pytest

from support_assistant.backend import app as backend
from support_assistant.backend import channels
from support_assistant.config import Settings
from support_assistant.errors import Overloaded, RateLimited
from support_assistant.security import (
    Bulkhead,
    RateLimiter,
    channel_session_id,
    client_ip,
    subject_key,
    valid_session_id,
)

STRONG = "s" * 40  # a plausible high-entropy id, for tests only
OTHER = "o" * 40


class Redis:
    """The few Redis commands the limiter and the backend use, with a switch to make it fail."""

    def __init__(self):
        self.data, self.ttls, self.fail = {}, {}, False

    def _check(self):
        if self.fail:
            raise ConnectionError("redis down")

    async def get(self, k):
        self._check()
        return self.data.get(k)

    async def set(self, k, v, ex=None, nx=False):
        self._check()
        if nx and k in self.data:
            return None
        self.data[k] = v
        if ex is not None:
            self.ttls[k] = ex
        return True

    async def incr(self, k):
        self._check()
        self.data[k] = int(self.data.get(k, 0)) + 1
        return self.data[k]

    async def ttl(self, k):
        return self.ttls.get(k, -1)

    async def delete(self, k):
        self.data.pop(k, None)
        self.ttls.pop(k, None)


# --- session ids ------------------------------------------------------------------------------
def test_a_long_random_session_id_is_accepted_and_a_short_or_odd_one_is_not():
    assert valid_session_id(STRONG, 32)
    assert not valid_session_id("S1", 32)
    assert not valid_session_id("s" * 31, 32)
    assert not valid_session_id("s" * 40 + "/../etc", 32)
    assert not valid_session_id("a b" * 20, 32)
    assert not valid_session_id("s" * 200, 32)


def test_the_length_rule_can_be_turned_off_for_tests_but_an_empty_id_never_passes():
    assert valid_session_id("S1", 0) and not valid_session_id("", 0)


def test_a_channel_session_id_cannot_be_computed_from_a_phone_number_alone():
    a = channel_session_id("secret-1", "twilio", "+15550001")
    assert a == channel_session_id(
        "secret-1", "twilio", "+15550001"
    )  # stable for the same customer
    assert a != channel_session_id("secret-2", "twilio", "+15550001")  # depends on the secret
    assert a != channel_session_id("secret-1", "twilio", "+15550002")
    assert "+15550001" not in a and valid_session_id(a, 32)


def test_the_subject_key_ignores_case_and_spacing_and_hides_the_subject():
    assert subject_key(" +1555 ", "") == subject_key("+1555", "")
    assert subject_key("", "Ada Lovelace") == subject_key("", "ada lovelace")
    assert "+1555" not in subject_key("+1555")


# --- caller address -----------------------------------------------------------------------------
def test_forwarded_for_is_ignored_unless_we_run_the_proxies_ourselves():
    headers = {"x-forwarded-for": "6.6.6.6, 10.0.0.1"}
    assert (
        client_ip(headers, "10.0.0.9", trusted_hops=0) == "10.0.0.9"
    )  # anyone can send the header
    assert client_ip(headers, "10.0.0.9", trusted_hops=1) == "10.0.0.1"  # what our proxy saw
    assert client_ip(headers, "10.0.0.9", trusted_hops=2) == "6.6.6.6"
    assert client_ip({}, None, trusted_hops=1) == "unknown"
    assert client_ip({"x-forwarded-for": "6.6.6.6"}, "10.0.0.9", trusted_hops=3) == "10.0.0.9"


# --- the rate limiter ---------------------------------------------------------------------------
async def test_a_request_rate_is_capped_per_window_and_the_caller_is_told_when_to_return():
    limiter = RateLimiter(Redis())
    for _ in range(3):
        await limiter.hit("k", limit=3, window_s=60)
    with pytest.raises(RateLimited) as err:
        await limiter.hit("k", limit=3, window_s=60)
    assert err.value.retry_after == 60 and err.value.http_status == 429


async def test_the_window_is_set_once_so_a_busy_key_cannot_extend_itself_forever():
    redis = Redis()
    limiter = RateLimiter(redis)
    await limiter.hit("k", 5, 60)
    redis.ttls["rl:k"] = 10  # time has passed
    await limiter.hit("k", 5, 60)
    assert redis.ttls["rl:k"] == 10  # the second hit did not reset it


async def test_failures_lock_a_key_out_and_a_reset_clears_it():
    limiter = RateLimiter(Redis())
    assert await limiter.blocked("k", 2) is None
    await limiter.record("k", 900)
    assert await limiter.blocked("k", 2) is None
    await limiter.record("k", 900)
    assert await limiter.blocked("k", 2) == 900
    await limiter.reset("k")
    assert await limiter.blocked("k", 2) is None


# --- load shedding ------------------------------------------------------------------------------
async def test_a_full_bulkhead_refuses_at_once_and_a_finished_one_frees_its_place():
    b = Bulkhead(1)
    async with b.slot():
        with pytest.raises(Overloaded):
            async with b.slot():
                pass
    async with b.slot():  # released
        pass


async def test_a_failing_request_still_releases_its_place():
    b = Bulkhead(1)
    with pytest.raises(RuntimeError):
        async with b.slot():
            raise RuntimeError("boom")
    async with b.slot():
        pass


# --- the backend, end to end ----------------------------------------------------------------------
def _setup(*, strict_ids=False, redis=None, orchestrator=None, **settings):
    sent: list[tuple[str, dict]] = []

    async def orch(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        if orchestrator:
            await orchestrator()
        return httpx.Response(
            200, json={"reply": f"answer {len(sent)}", "pending_approval": None, "escalated": False}
        )

    class Cur:
        def __init__(self, row):
            self._row = row

        async def fetchone(self):
            return self._row

        async def fetchall(self):
            return []

    class Pool:
        def connection(self):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def conn():
                class C:
                    async def execute(self_, sql, params=None):
                        if "phone = %s" in sql and params and params[0] != "+1":
                            return Cur(None)  # an unknown phone
                        return Cur({"id": "C1", "date_of_birth": date(1990, 5, 1)})

                yield C()

            return conn()

    class Memory:
        async def load(self, cid, sid):
            return []

        async def append(self, cid, sid, new):
            return new

    if not strict_ids:
        settings.setdefault("min_session_id_length", 0)
    redis = redis or Redis()
    state = backend.app.state
    state.s = Settings(database_url="postgresql://x", redis_url="redis://x", **settings)
    state.redis, state.pool, state.memory = redis, Pool(), Memory()
    state.limiter, state.bulkhead = RateLimiter(redis), Bulkhead(state.s.max_inflight_turns)
    state.rewrite = state.budget = state.embedder = None
    state.http = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(orch))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://t")
    return client, redis, sent


async def _verify(client, session, dob="1990-05-01", phone="+1"):
    return await client.post(
        "/verify", json={"session_id": session, "phone": phone, "date_of_birth": dob}
    )


async def _chat(client, session, **extra):
    return await client.post("/chat", json={"session_id": session, "message": "hello", **extra})


# session ids
async def test_a_short_session_id_is_refused_everywhere_with_the_unverified_answer():
    client, redis, sent = _setup(strict_ids=True)
    assert (await _verify(client, "S1")).status_code == 401
    redis.data["verified:S1"] = "C1"  # even if somehow marked verified, a weak id is never honoured
    assert (await _chat(client, "S1")).status_code == 401
    assert sent == []


async def test_a_strong_session_id_verifies_and_chats():
    client, _, sent = _setup(strict_ids=True)
    assert (await _verify(client, STRONG)).status_code == 200
    assert (await _chat(client, STRONG)).status_code == 200 and sent


# identity-guessing lockout
async def test_repeated_wrong_dates_lock_the_session_out_even_for_the_right_answer():
    client, _, _ = _setup(verify_max_failures_session=3)
    for _ in range(3):
        assert (await _verify(client, STRONG, dob="2000-01-01")).status_code == 401
    locked = await _verify(client, STRONG, dob="1990-05-01")  # the correct date, but too late
    assert locked.status_code == 429 and locked.headers["retry-after"] == "900"


async def test_guessing_one_person_across_many_sessions_is_stopped_by_the_subject_limit():
    client, _, _ = _setup(verify_max_failures_subject=4, verify_max_failures_session=99)
    for i in range(4):
        await _verify(client, f"{i}" * 40, dob="2000-01-01")
    assert (await _verify(client, "9" * 40, dob="2000-01-01")).status_code == 429


async def test_one_address_guessing_many_people_is_stopped_by_the_ip_limit():
    client, _, _ = _setup(verify_max_failures_ip=3, verify_max_failures_subject=99)
    for i in range(3):
        await _verify(client, f"{i}" * 40, dob="2000-01-01", phone=f"+9{i}")
    assert (await _verify(client, "8" * 40, dob="2000-01-01", phone="+98")).status_code == 429


async def test_an_unknown_phone_is_counted_exactly_like_a_wrong_date_so_nothing_leaks():
    known, unknown = [], []
    for phone, out in (("+1", known), ("+9", unknown)):
        client, _, _ = _setup(verify_max_failures_session=2)
        for _ in range(3):
            out.append((await _verify(client, STRONG, dob="2000-01-01", phone=phone)).status_code)
    assert known == unknown == [401, 401, 429]


async def test_a_success_clears_the_session_and_person_counters():
    client, redis, _ = _setup(verify_max_failures_session=3)
    await _verify(client, STRONG, dob="2000-01-01")
    assert (await _verify(client, STRONG)).status_code == 200
    assert not any(k.startswith("rl:vf:s:") or k.startswith("rl:vf:p:") for k in redis.data)


async def test_if_the_counters_are_unavailable_verification_fails_closed():
    client, redis, _ = _setup()
    redis.fail = True
    assert (await _verify(client, STRONG)).status_code == 503


# chat rate limits
async def test_a_customer_cannot_exceed_the_message_rate_and_is_told_when_to_return():
    client, redis, _ = _setup(chat_rate_per_minute=3)
    redis.data[f"verified:{STRONG}"] = "C1"
    codes = [(await _chat(client, STRONG)).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert (await _chat(client, STRONG)).headers["retry-after"] == "60"


async def test_a_broken_limiter_does_not_stop_a_conversation():
    client, redis, _ = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    ok = await _chat(client, STRONG)

    async def broken(*a, **k):
        raise ConnectionError

    backend.app.state.limiter.hit = broken  # only the counters are down, not the verified flag
    assert ok.status_code == 200 and (await _chat(client, STRONG)).status_code == 200


# idempotent retries
async def test_a_retried_message_runs_once_and_the_retry_gets_the_same_answer():
    client, redis, sent = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    first = await _chat(client, STRONG, message_id="msg-0001")
    again = await _chat(client, STRONG, message_id="msg-0001")
    assert first.json() == again.json() and len(sent) == 1


async def test_two_different_messages_both_run():
    client, redis, sent = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    await _chat(client, STRONG, message_id="msg-0001")
    await _chat(client, STRONG, message_id="msg-0002")
    assert len(sent) == 2


async def test_an_id_is_per_customer_so_one_customer_cannot_replay_anothers_answer():
    client, redis, sent = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    redis.data[f"verified:{OTHER}"] = "C2"
    await _chat(client, STRONG, message_id="msg-0001")
    await _chat(client, OTHER, message_id="msg-0001")
    assert len(sent) == 2


async def test_a_message_still_in_flight_is_refused_not_run_twice():
    release = asyncio.Event()

    async def slow():
        await release.wait()

    client, redis, sent = _setup(orchestrator=slow)
    redis.data[f"verified:{STRONG}"] = "C1"
    first = asyncio.create_task(_chat(client, STRONG, message_id="msg-0001"))
    await asyncio.sleep(0.05)
    duplicate = await _chat(client, STRONG, message_id="msg-0001")
    release.set()
    assert duplicate.status_code == 409 and (await first).status_code == 200 and len(sent) == 1


async def test_a_failed_turn_can_be_retried_with_the_same_id():
    client, redis, sent = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    real = backend.app.state.http
    backend.app.state.http = httpx.AsyncClient(
        base_url="http://o", transport=httpx.MockTransport(lambda r: httpx.Response(503, json={}))
    )
    assert (await _chat(client, STRONG, message_id="msg-0001")).status_code == 503
    backend.app.state.http = real
    assert (await _chat(client, STRONG, message_id="msg-0001")).status_code == 200


# load shedding
async def test_a_pod_at_its_limit_sheds_load_with_a_retry_after_instead_of_queueing():
    release = asyncio.Event()

    async def slow():
        await release.wait()

    client, redis, _ = _setup(orchestrator=slow, max_inflight_turns=1)
    redis.data[f"verified:{STRONG}"] = "C1"
    redis.data[f"verified:{OTHER}"] = "C2"
    first = asyncio.create_task(_chat(client, STRONG))
    await asyncio.sleep(0.05)
    shed = await _chat(client, OTHER)
    release.set()
    assert shed.status_code == 503 and shed.headers["retry-after"] == "5"
    assert (await first).status_code == 200


# the Twilio channel's session ids
async def test_the_twilio_channel_never_uses_a_session_id_derived_from_the_phone_alone(monkeypatch):
    client, redis, _ = _setup(twilio_channel_enabled=True, twilio_auth_token="tok")
    seen: list[str] = []

    async def verify(body, ip):
        seen.append(body.session_id)
        raise channels.AuthenticationError("no")

    monkeypatch.setattr(backend, "verify_identity", verify)
    await channels.converse("+1", "1990-05-01", "SM1")
    assert seen and "+1" not in seen[0] and valid_session_id(seen[0], 32)


# --- request limits and headers ------------------------------------------------------------------
async def test_every_response_carries_the_hardening_headers_and_no_cors_header():
    client, _, _ = _setup()
    r = await client.get("/healthz", headers={"origin": "https://evil.example"})
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert "strict-transport-security" in r.headers
    assert "access-control-allow-origin" not in r.headers  # other origins are refused by default


async def test_an_error_response_is_hardened_too():
    client, _, _ = _setup(strict_ids=True)
    r = await _verify(client, "S1")
    assert r.status_code == 401 and r.headers["x-content-type-options"] == "nosniff"


async def test_a_huge_body_is_refused_before_it_is_read_and_still_carries_the_headers():
    client, _, sent = _setup()
    r = await client.post(
        "/chat", content=b"x" * 70_000, headers={"content-type": "application/json"}
    )
    assert r.status_code == 413 and r.headers["x-content-type-options"] == "nosniff"
    assert sent == []


async def test_a_post_that_does_not_say_how_big_it_is_is_refused():
    client, _, _ = _setup()

    async def chunks():
        yield b'{"session_id": "x"}'

    r = await client.post("/chat", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 411


async def test_the_upload_endpoint_keeps_its_own_larger_limit():
    client, redis, _ = _setup()
    redis.data[f"verified:{STRONG}"] = "C1"
    backend.app.state.uploads = None
    files = {"file": ("a.txt", b"x" * 100_000, "text/plain")}
    r = await client.post("/upload", data={"session_id": STRONG}, files=files)
    assert (
        r.status_code != 413
    )  # not cut off by the 64 KB JSON limit (it fails later, as unconfigured)
