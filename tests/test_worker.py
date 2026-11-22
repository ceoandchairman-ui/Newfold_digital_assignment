import json

from support_assistant.errors import ExternalServiceUnavailable, NotFoundError
from support_assistant.worker import Outcome, handle_message

OK = json.dumps({"customer_id": "C1", "channel": "sms", "template": "payment_received"})


async def _phone(_: str) -> str:
    return "+15550001111"


async def test_sends_and_completes():
    sent: list[tuple[str, str]] = []

    async def sms(phone: str, text: str) -> None:
        sent.append((phone, text))

    assert await handle_message(OK, _phone, sms) is Outcome.COMPLETE
    assert sent == [("+15550001111", "We received your payment. Thank you.")]


async def test_malformed_and_unknown_template_dead_letter():
    async def sms(*_):  # must never be called
        raise AssertionError

    assert await handle_message("not json", _phone, sms) is Outcome.DEAD_LETTER
    bad = json.dumps({"customer_id": "C1", "channel": "sms", "template": "nope"})
    assert await handle_message(bad, _phone, sms) is Outcome.DEAD_LETTER


async def test_email_is_dead_lettered_until_supported():
    email = json.dumps({"customer_id": "C1", "channel": "email", "template": "payment_received"})

    async def sms(*_):
        raise AssertionError

    assert await handle_message(email, _phone, sms) is Outcome.DEAD_LETTER


async def test_provider_outage_retries_and_unknown_customer_dead_letters():
    async def down(*_):
        raise ExternalServiceUnavailable("x")

    assert await handle_message(OK, _phone, down) is Outcome.RETRY

    async def no_customer(_: str) -> str:
        raise NotFoundError("customer")

    async def sms(*_):
        raise AssertionError

    assert await handle_message(OK, no_customer, sms) is Outcome.DEAD_LETTER
