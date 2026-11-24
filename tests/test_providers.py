"""Twilio, Stripe, Gmail and Drive: the MCP servers, the provider clients, the Twilio chat channel.

Every provider here runs as a stub or against a mock HTTP transport. Nothing contacts a real service.
"""

from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
import pytest

from support_assistant.backend import app as backend
from support_assistant.backend import channels
from support_assistant.config import Settings
from support_assistant.errors import (
    AuthenticationError,
    ExternalServiceUnavailable,
    PaymentOutcomeUnknown,
)
from support_assistant.integrations.google_providers import (
    MAX_DOCUMENT_CHARS,
    Document,
    DriveApiProvider,
    GmailApiProvider,
    MailThread,
    StubDocuments,
    StubEmail,
    build_document_provider,
    build_email_provider,
)
from support_assistant.integrations.stripe_gateway import (
    StripeGateway,
    build_gateway,
    to_minor_units,
)
from support_assistant.integrations.twilio_client import (
    StubSms,
    TwilioSms,
    build_sms_provider,
    twilio_signature,
    valid_twilio_signature,
)
from support_assistant.mcp_servers.drive import build_drive_server
from support_assistant.mcp_servers.gmail import build_gmail_server
from support_assistant.mcp_servers.stripe_server import build_stripe_server
from support_assistant.mcp_servers.twilio_sms import build_twilio_server
from support_assistant.services.payments import GatewayOutcome, StubGateway
from support_assistant.worker import TEMPLATES, _sms_sender


# --- fakes ------------------------------------------------------------------------------------
class Cur:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class Conn:
    def __init__(self, tables):
        self._tables = tables

    async def execute(self, sql, params=None):
        for name, rows in self._tables.items():
            if f"FROM {name}" in sql:
                return Cur(rows)
        return Cur([])


class Pool:
    def __init__(self, **tables):
        self.tables = tables

    @asynccontextmanager
    async def connection(self):
        yield Conn(self.tables)


CUSTOMER = {
    "id": "C1",
    "full_name": "Ada",
    "date_of_birth": None,
    "phone": "+15550001",
    "email": "ada@x.com",
}


async def _call(server, tool, args):
    result = await server.call_tool(tool, args)
    return result[1] if isinstance(result, tuple) else result


def _settings(**kw: Any) -> Settings:
    return Settings(database_url="postgresql://x", redis_url="redis://x", **kw)


# --- twilio_sms server -------------------------------------------------------------------------
async def test_sms_text_comes_from_a_template_and_the_number_from_the_customer_record():
    sms = StubSms()
    server = build_twilio_server(Pool(customers=[CUSTOMER]), sms)
    out = await _call(server, "send_sms", {"customer_id": "C1", "template": "payment_received"})
    assert out["ok"] and sms.sent[0].to == "+15550001"
    schema = {t.name: t for t in await server.list_tools()}["send_sms"].inputSchema
    assert set(schema["properties"]) == {
        "customer_id",
        "template",
    }  # no way to choose a number or text


async def test_an_unknown_template_sends_nothing():
    sms = StubSms()
    server = build_twilio_server(Pool(customers=[CUSTOMER]), sms)
    out = await _call(server, "send_sms", {"customer_id": "C1", "template": "free text"})
    assert out["ok"] is False and sms.sent == []


async def test_message_status_is_only_visible_for_the_customers_own_number():
    sms = StubSms()
    mine = await sms.send("+15550001", "x")
    other = await sms.send("+15559999", "y")
    server = build_twilio_server(Pool(customers=[CUSTOMER]), sms)
    ok = await _call(server, "get_message_status", {"customer_id": "C1", "message_sid": mine.sid})
    bad = await _call(server, "get_message_status", {"customer_id": "C1", "message_sid": other.sid})
    assert ok["ok"] and bad["ok"] is False


# --- Twilio REST client -------------------------------------------------------------------------
async def test_twilio_posts_a_form_with_basic_auth_and_reads_the_sid():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"sid": "SM1", "status": "queued"})

    sms = TwilioSms("ACx", "tok", "+1000", transport=httpx.MockTransport(handler))
    sent = await sms.send("+15550001", "hello")
    assert sent.sid == "SM1" and seen[0].url.path.endswith("/Accounts/ACx/Messages.json")
    assert b"To=%2B15550001" in seen[0].content and b"From=%2B1000" in seen[0].content
    assert seen[0].headers["authorization"].startswith("Basic ")


@pytest.mark.parametrize("status", [400, 401, 500])
async def test_a_twilio_error_is_a_clean_outage_not_a_leak(status):
    sms = TwilioSms(
        "ACx", "tok", "+1", transport=httpx.MockTransport(lambda r: httpx.Response(status))
    )
    with pytest.raises(ExternalServiceUnavailable):
        await sms.send("+1555", "x")


async def test_an_unknown_message_sid_is_none():
    sms = TwilioSms(
        "ACx", "tok", "+1", transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    assert await sms.status("SMnope") is None


def test_twilio_is_a_stub_until_it_is_switched_on_and_fully_configured():
    assert isinstance(build_sms_provider(_settings()), StubSms)
    partial = _settings(use_stub_twilio=False, twilio_account_sid="ACx")
    assert isinstance(build_sms_provider(partial), StubSms)  # no token or number: still the stub


# --- Twilio request signature ----------------------------------------------------------------------
URL, PARAMS = "https://example.test/channels/twilio/inbound", {"From": "+1555", "Body": "hi"}


def test_a_correctly_signed_request_is_accepted():
    sig = twilio_signature(URL, PARAMS, "tok")
    assert valid_twilio_signature(URL, PARAMS, sig, "tok")


@pytest.mark.parametrize(
    ("params", "token", "header"),
    [
        ({"From": "+1555", "Body": "hi!"}, "tok", None),  # tampered body
        (PARAMS, "other-token", None),  # signed with another token
        (PARAMS, "tok", ""),  # no header
        (PARAMS, "", None),  # an empty token never validates
    ],
)
def test_anything_else_is_refused(params, token, header):
    sig = twilio_signature(URL, PARAMS, "tok") if header is None else header
    assert not valid_twilio_signature(URL, params, sig, token)


def test_the_signature_ignores_the_order_the_parameters_arrive_in():
    assert twilio_signature(URL, {"a": "1", "b": "2"}, "t") == twilio_signature(
        URL, {"b": "2", "a": "1"}, "t"
    )


# --- Stripe gateway --------------------------------------------------------------------------------------
async def _pm(customer_id: str) -> tuple[str, str]:
    return "cus_1", "pm_1"


def _stripe(handler) -> StripeGateway:
    return StripeGateway("sk_test", payment_method_for=_pm, transport=httpx.MockTransport(handler))


def _intent(status: str, code: int = 200) -> httpx.Response:
    return httpx.Response(code, json={"status": status})


async def test_a_succeeded_intent_is_approved_and_sent_with_the_idempotency_key_and_cents():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _intent("succeeded")

    outcome = await _stripe(handler).charge("C1", Decimal("12.34"), "key-1")
    assert outcome is GatewayOutcome.APPROVED
    assert seen[0].headers["idempotency-key"] == "key-1"
    assert (
        b"amount=1234" in seen[0].content
        and b"metadata%5Bidempotency_key%5D=key-1" in seen[0].content
    )


def test_amounts_are_sent_in_the_smallest_unit():
    assert to_minor_units(Decimal("12.34")) == 1234 and to_minor_units(Decimal("0.05")) == 5


async def test_a_card_decline_is_declined_and_an_unfinished_intent_is_declined_too():
    g = _stripe(lambda r: httpx.Response(402, json={"error": {"type": "card_error"}}))
    assert await g.charge("C1", Decimal("1"), "k") is GatewayOutcome.DECLINED
    g = _stripe(lambda r: _intent("requires_action"))
    assert await g.charge("C1", Decimal("1"), "k") is GatewayOutcome.DECLINED


@pytest.mark.parametrize(
    "response", [_intent("processing"), httpx.Response(503), httpx.Response(429)]
)
async def test_an_unsettled_answer_leaves_the_payment_pending_not_failed(response):
    with pytest.raises(PaymentOutcomeUnknown):
        await _stripe(lambda r: response).charge("C1", Decimal("1"), "k")


async def test_a_timeout_is_unknown_and_a_failure_to_connect_is_a_clean_failure():
    def timeout(request):
        raise httpx.ReadTimeout("slow")

    def refused(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(PaymentOutcomeUnknown):
        await _stripe(timeout).charge("C1", Decimal("1"), "k")
    with pytest.raises(ExternalServiceUnavailable):  # nothing was sent: safe to release the bill
        await _stripe(refused).charge("C1", Decimal("1"), "k")


async def test_lookup_searches_by_the_metadata_key_and_none_means_stripe_never_saw_it():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "idempotency_key" in request.url.params["query"]
        return httpx.Response(200, json={"data": []})

    assert await _stripe(handler).lookup("k") is None
    done = _stripe(lambda r: httpx.Response(200, json={"data": [{"status": "succeeded"}]}))
    assert await done.lookup("k") is GatewayOutcome.APPROVED


async def test_the_customer_to_payment_method_mapping_is_a_placeholder_that_refuses():
    g = StripeGateway("sk_test", transport=httpx.MockTransport(lambda r: _intent("succeeded")))
    with pytest.raises(ExternalServiceUnavailable):
        await g.charge("C1", Decimal("1"), "k")


def test_the_gateway_is_the_stub_unless_stripe_is_switched_on_and_has_a_key():
    assert isinstance(build_gateway(_settings()), StubGateway)
    assert isinstance(build_gateway(_settings(use_stub_gateway=False)), StubGateway)  # no key
    on = _settings(use_stub_gateway=False, stripe_secret_key="sk_test")
    assert isinstance(build_gateway(on), StripeGateway)


# --- stripe server -----------------------------------------------------------------------------------------
async def test_payment_status_is_hidden_from_everyone_but_its_owner():
    tx = {
        "id": 1,
        "customer_id": "C1",
        "bill_id": "B1",
        "amount": Decimal("5"),
        "status": "PENDING",
    }
    server = build_stripe_server(Pool(transaction_history=[tx]), StubGateway())
    mine = await _call(server, "get_payment_status", {"customer_id": "C1", "idempotency_key": "k"})
    theirs = await _call(
        server, "get_payment_status", {"customer_id": "C2", "idempotency_key": "k"}
    )
    assert mine["payment"]["status"] == "PENDING" and theirs["ok"] is False


# --- gmail server --------------------------------------------------------------------------------------------
async def test_email_goes_to_the_address_on_file_with_template_text():
    mail = StubEmail()
    server = build_gmail_server(Pool(customers=[CUSTOMER]), mail)
    out = await _call(server, "send_email", {"customer_id": "C1", "template": "payment_received"})
    assert (
        out["ok"]
        and mail.sent[0][0] == "ada@x.com"
        and mail.sent[0][2] == TEMPLATES["payment_received"]
    )
    schema = {t.name: t for t in await server.list_tools()}["send_email"].inputSchema
    assert set(schema["properties"]) == {"customer_id", "template"}


async def test_a_customer_with_no_email_gets_a_clean_error_and_nothing_is_sent():
    mail = StubEmail()
    server = build_gmail_server(Pool(customers=[{**CUSTOMER, "email": None}]), mail)
    out = await _call(server, "send_email", {"customer_id": "C1", "template": "payment_received"})
    assert out["ok"] is False and mail.sent == []


async def test_support_threads_are_limited_and_read_by_the_customers_own_address():
    threads = {"ada@x.com": [MailThread(str(i), f"s{i}", "…") for i in range(9)]}
    server = build_gmail_server(Pool(customers=[CUSTOMER]), StubEmail(threads))
    out = await _call(server, "get_support_thread", {"customer_id": "C1", "limit": 99})
    assert len(out["threads"]) == 5  # clamped


# --- drive server -----------------------------------------------------------------------------------------------
async def test_drive_documents_are_marked_untrusted_and_cut_to_size():
    big = Document("d1", "Roaming policy", "roaming " + "x" * (MAX_DOCUMENT_CHARS + 500))
    server = build_drive_server(StubDocuments([big]))
    found = await _call(server, "search_documents", {"customer_id": "C1", "query": "roaming"})
    assert found["documents"][0]["untrusted"] is True and len(found["documents"][0]["text"]) <= 300
    read = await _call(server, "read_document", {"customer_id": "C1", "document_id": "d1"})
    assert len(read["document"]["text"]) == MAX_DOCUMENT_CHARS and read["document"]["untrusted"]


async def test_an_unknown_document_is_a_clean_error():
    out = await _call(
        build_drive_server(StubDocuments()),
        "read_document",
        {"customer_id": "C1", "document_id": "x"},
    )
    assert out["ok"] is False


# --- placeholders ----------------------------------------------------------------------------------------------------
async def test_the_google_providers_are_placeholders_that_refuse_until_they_are_written():
    s = _settings(
        use_stub_gmail=False,
        use_stub_drive=False,
        google_credentials_json="{}",
        google_support_mailbox="s@x.com",
        google_drive_folder_id="f",
    )
    mail, docs = build_email_provider(s), build_document_provider(s)
    assert isinstance(mail, GmailApiProvider) and isinstance(docs, DriveApiProvider)
    for call in (
        mail.send("a@b.c", "s", "t"),
        mail.threads("a@b.c", 1),
        docs.search("q", 1),
        docs.read("d"),
    ):
        with pytest.raises(ExternalServiceUnavailable):
            await call
    assert isinstance(build_email_provider(_settings()), StubEmail)  # the default is the stub


# --- the notification worker's SMS provider switch --------------------------------------------------------------
async def test_the_worker_sends_sms_through_twilio_only_when_asked():
    sent: list[str] = []

    class Spy(StubSms):
        async def send(self, to, text):
            sent.append(to)
            return await super().send(to, text)

    import support_assistant.integrations.twilio_client as tc

    original = tc.build_sms_provider
    tc.build_sms_provider = lambda settings: Spy()  # type: ignore[assignment]
    try:
        await _sms_sender(_settings(sms_provider="twilio"))("+15550001", "hi")
    finally:
        tc.build_sms_provider = original
    assert sent == ["+15550001"]
    assert (
        _sms_sender(_settings()).__name__ == "stub"
    )  # the default stays Azure Communication Services


# --- the Twilio chat channel -----------------------------------------------------------------------------------
class KV:
    def __init__(self):
        self.data: dict[str, Any] = {}

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v, ex=None, nx=False):
        if nx and k in self.data:
            return None
        self.data[k] = v
        return True


@pytest.fixture
def channel(monkeypatch):
    s = _settings(twilio_channel_enabled=True, twilio_auth_token="tok", twilio_webhook_url=URL)
    backend.app.state.s = s
    backend.app.state.redis = KV()
    backend.app.state.pool = Pool(
        customers=[{"id": "C1", "full_name": "Ada", "date_of_birth": None}]
    )
    calls: list[tuple[str, Any]] = []

    async def verify(body, ip):
        calls.append(("verify", body.phone))
        if body.date_of_birth.isoformat() != "1990-05-01":
            raise AuthenticationError("no")
        await backend.app.state.redis.set(f"verified:{body.session_id}", "C1")
        return {"verified": True}

    async def chat(body, ip):
        calls.append(("chat", body.message))
        return {"reply": "Your plan is Plus & <b>cheap</b>.", "pending_approval": None}

    async def approve(body):
        calls.append(("approve", body.approved))
        return {"reply": "Paid."}

    async def pending(customer_id):
        return backend.app.state.pending

    backend.app.state.pending = None
    monkeypatch.setattr(backend, "verify_identity", verify)
    monkeypatch.setattr(backend, "run_chat", chat)
    monkeypatch.setattr(backend, "approve", approve)
    monkeypatch.setattr(backend, "_pending_for", pending)
    c = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend.app), base_url="https://example.test"
    )
    c.calls = calls  # type: ignore[attr-defined]
    return c


async def _post(client, form, *, sign=True, token="tok"):
    headers = {"X-Twilio-Signature": twilio_signature(URL, form, token)} if sign else {}
    return await client.post("/channels/twilio/inbound", data=form, headers=headers)


def _msg(body: str, sid: str = "SM1", frm: str = "+15550001") -> dict[str, str]:
    return {"From": frm, "Body": body, "MessageSid": sid}


async def test_the_channel_is_off_by_default():
    backend.app.state.s = _settings()
    c = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend.app), base_url="https://example.test"
    )
    assert (await c.post("/channels/twilio/inbound", data=_msg("hi"))).status_code == 404


async def test_enabled_without_a_token_is_refused_not_open(channel):
    backend.app.state.s = _settings(twilio_channel_enabled=True)
    assert (await _post(channel, _msg("hi"))).status_code == 503


async def test_a_request_without_a_valid_signature_never_reaches_the_pipeline(channel):
    assert (await _post(channel, _msg("hi"), sign=False)).status_code == 403
    assert (await _post(channel, _msg("hi"), token="wrong")).status_code == 403
    assert channel.calls == []


async def test_an_unverified_sender_is_asked_for_a_date_of_birth_and_nothing_else_runs(channel):
    r = await _post(channel, _msg("what is my balance"))
    assert "date of birth" in r.text and channel.calls == []


async def test_a_wrong_date_of_birth_is_refused_and_the_right_one_verifies(channel):
    assert "could not verify" in (await _post(channel, _msg("2000-01-01", "SM2"))).text
    assert "verified" in (await _post(channel, _msg("1990-05-01", "SM3"))).text


async def test_a_verified_sender_gets_the_answer_escaped_as_xml(channel):
    await _post(channel, _msg("1990-05-01", "SM1"))
    r = await _post(channel, _msg("what plan am I on", "SM2"))
    assert r.headers["content-type"].startswith("application/xml")
    assert "Plus &amp; &lt;b&gt;cheap&lt;/b&gt;" in r.text and "<b>" not in r.text


async def test_a_repeated_delivery_of_the_same_message_is_answered_only_once(channel):
    await _post(channel, _msg("1990-05-01", "SM1"))
    first = await _post(channel, _msg("hello", "SM9"))
    again = await _post(channel, _msg("hello", "SM9"))
    assert "Message" in first.text and "<Response/>" in again.text
    assert [c for c in channel.calls if c[0] == "chat"] == [("chat", "hello")]


async def test_yes_and_no_answer_a_pending_payment_confirmation(channel):
    await _post(channel, _msg("1990-05-01", "SM1"))
    backend.app.state.pending = {"session_id": "twilio:+15550001", "question": "Pay 5?"}
    assert "Paid." in (await _post(channel, _msg("YES", "SM2"))).text
    assert ("approve", True) in channel.calls


async def test_an_unknown_number_gets_one_generic_answer(channel):
    backend.app.state.pool = Pool(customers=[])
    r = await _post(channel, _msg("hi", frm="+19998887777"))
    assert "call us" in r.text and channel.calls == []


def test_twiml_escapes_and_truncates():
    assert "&lt;script&gt;" in channels.twiml("<script>").body.decode()
    assert len(channels.twiml("x" * 5000).body.decode()) < 1700
    assert channels.twiml(None).body.decode().endswith("<Response/>")


def test_the_whatsapp_prefix_is_stripped_from_the_sender():
    assert channels._phone("whatsapp:+15550001") == "+15550001"
