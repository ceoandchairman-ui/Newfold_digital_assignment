"""Notification worker: reads the Service Bus queue and sends through Communication Services.

The message handler is a pure function over injected senders so it is testable offline.
Outcomes: sent -> complete the message; bad message -> dead-letter (no retry storm);
provider outage -> abandon so Service Bus redelivers (max_delivery_count then dead-letters).
Email goes to customers.email (migration 0004); a customer without one is dead-lettered for email.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum

import structlog

from support_assistant import repositories as repo
from support_assistant.config import Settings, get_settings
from support_assistant.db import make_pool
from support_assistant.errors import ExternalServiceUnavailable, SupportError

log = structlog.get_logger()

TEMPLATES = {
    "payment_received": "We received your payment. Thank you.",
    "suspension_lifted": "Your service has been restored.",
    "escalation_followup": "A support agent will contact you shortly.",
}

SmsSender = Callable[[str, str], Awaitable[None]]  # (phone, text)
EmailSender = Callable[[str, str, str], Awaitable[None]]  # (address, subject, text)
PhoneLookup = Callable[[str], Awaitable[str]]  # customer_id -> phone
EmailLookup = Callable[[str], Awaitable[str | None]]  # customer_id -> email, None if unknown

SUBJECTS = {
    "payment_received": "Payment received",
    "suspension_lifted": "Your service is restored",
    "escalation_followup": "Your support request",
}


class Outcome(StrEnum):
    COMPLETE = "complete"
    DEAD_LETTER = "dead_letter"
    RETRY = "retry"


async def handle_message(
    raw: str,
    lookup_phone: PhoneLookup,
    send_sms: SmsSender,
    *,
    lookup_email: EmailLookup | None = None,
    send_email: EmailSender | None = None,
) -> Outcome:
    try:
        msg = json.loads(raw)
        customer_id, channel, template = msg["customer_id"], msg["channel"], msg["template"]
        text = TEMPLATES[template]
    except (ValueError, KeyError, TypeError):
        log.warning("notification_malformed")
        return Outcome.DEAD_LETTER
    if channel == "email" and (lookup_email is None or send_email is None):
        log.warning("notification_email_not_configured")
        return Outcome.DEAD_LETTER
    if channel not in ("sms", "email"):
        log.warning("notification_channel_unsupported", channel=channel)
        return Outcome.DEAD_LETTER
    try:
        if channel == "sms":
            await send_sms(await lookup_phone(customer_id), text)
        else:
            assert lookup_email is not None and send_email is not None
            address = await lookup_email(customer_id)
            if not address:
                log.warning("notification_no_email_on_file", customer_id=customer_id)
                return Outcome.DEAD_LETTER  # nothing to send to; retrying will not help
            await send_email(address, SUBJECTS[template], text)
    except ExternalServiceUnavailable:
        return Outcome.RETRY
    except SupportError:
        return Outcome.DEAD_LETTER  # unknown customer etc.; retrying will not help
    log.info("notification_sent", customer_id=customer_id, template=template)
    return Outcome.COMPLETE


def _sms_sender(settings: Settings) -> SmsSender:
    """The configured SMS provider: Twilio, or Azure Communication Services (the default)."""
    if settings.sms_provider == "twilio":
        return _twilio_sender(settings)
    return _acs_sender(settings)


def _twilio_sender(settings: Settings) -> SmsSender:
    from support_assistant.integrations.twilio_client import build_sms_provider

    provider = build_sms_provider(settings)  # a stub until Twilio is configured

    async def send(phone: str, text: str) -> None:
        try:
            await provider.send(phone, text)
        except ExternalServiceUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceUnavailable("sms provider unavailable") from exc

    return send


def _acs_sender(settings: Settings) -> SmsSender:
    """Communication Services SMS, or a log-only sender while use_stub_sms is on."""
    if settings.use_stub_sms:

        async def stub(phone: str, text: str) -> None:
            log.info("sms_stub", to="***" + phone[-4:], text=text)

        return stub

    from azure.communication.sms.aio import SmsClient
    from azure.identity.aio import DefaultAzureCredential

    endpoint = settings.acs_endpoint
    from_number = settings.acs_sms_from
    if not endpoint or not from_number:
        raise RuntimeError("ACS_ENDPOINT and ACS_SMS_FROM are required when USE_STUB_SMS=false")

    async def send(phone: str, text: str) -> None:
        try:
            async with DefaultAzureCredential() as cred, SmsClient(endpoint, cred) as client:
                await client.send(from_=from_number, to=[phone], message=text)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceUnavailable("sms provider unavailable") from exc

    return send


def _email_sender(settings: Settings) -> EmailSender:
    """Communication Services Email, or a log-only sender while use_stub_email is on."""
    if settings.use_stub_email:

        async def stub(address: str, subject: str, text: str) -> None:
            log.info("email_stub", to="***@" + address.rsplit("@", 1)[-1], subject=subject)

        return stub

    from azure.communication.email.aio import EmailClient
    from azure.identity.aio import DefaultAzureCredential

    endpoint, sender = settings.acs_endpoint, settings.acs_email_from
    if not endpoint or not sender:
        raise RuntimeError("ACS_ENDPOINT and ACS_EMAIL_FROM are required when USE_STUB_EMAIL=false")

    async def send(address: str, subject: str, text: str) -> None:
        message = {
            "senderAddress": sender,
            "recipients": {"to": [{"address": address}]},
            "content": {"subject": subject, "plainText": text},
        }
        try:
            async with DefaultAzureCredential() as cred, EmailClient(endpoint, cred) as client:
                poller = await client.begin_send(message)
                await poller.result()
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceUnavailable("email provider unavailable") from exc

    return send


async def run() -> None:
    from azure.identity.aio import DefaultAzureCredential
    from azure.servicebus.aio import ServiceBusClient

    settings = get_settings()
    if not settings.servicebus_namespace_fqdn:
        raise RuntimeError("SERVICEBUS_NAMESPACE_FQDN is required")
    pool = make_pool(settings)
    await pool.open()
    sender = _sms_sender(settings)
    email_sender = _email_sender(settings)

    async def lookup(customer_id: str) -> str:
        async with pool.connection() as conn:
            return str((await repo.get_customer(conn, customer_id))["phone"])

    async def lookup_email(customer_id: str) -> str | None:
        async with pool.connection() as conn:
            email = (await repo.get_customer(conn, customer_id)).get("email")
        return str(email) if email else None

    async with (
        DefaultAzureCredential() as cred,
        ServiceBusClient(settings.servicebus_namespace_fqdn, cred) as sb,
    ):
        async with sb.get_queue_receiver("notifications", max_wait_time=30) as receiver:
            while True:
                for m in await receiver.receive_messages(max_message_count=10, max_wait_time=30):
                    outcome = await handle_message(
                        str(m), lookup, sender, lookup_email=lookup_email, send_email=email_sender
                    )
                    if outcome is Outcome.COMPLETE:
                        await receiver.complete_message(m)
                    elif outcome is Outcome.DEAD_LETTER:
                        await receiver.dead_letter_message(m, reason="unprocessable")
                    else:
                        await receiver.abandon_message(m)


if __name__ == "__main__":
    asyncio.run(run())
