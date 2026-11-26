"""Notification worker: reads the Kafka topic and sends through Communication Services.

The message handler is a pure function over injected senders so it is testable offline.
Outcomes: sent -> commit the offset; bad message -> copy to the dead-letter topic (no retry
storm); provider outage -> try again with a growing pause, and after MAX_ATTEMPTS dead-letter it.
Kafka has none of a queue's per-message features, so the worker does them itself:
  * the offset is committed only after a message is sent or dead-lettered (at-least-once);
  * retries happen in place, so a failing message holds back the ones behind it in its partition;
  * a repeated message_id is remembered in Redis for DEDUPE_TTL_S and not sent again. This is best
    effort: if Redis is unavailable, or the worker dies between sending and recording, the customer
    can get the message twice.
Email goes to customers.email (migration 0004); a customer without one is dead-lettered for email.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any

import structlog

from support_assistant import repositories as repo
from support_assistant.config import Settings, get_settings
from support_assistant.db import make_pool
from support_assistant.errors import ExternalServiceUnavailable, SupportError
from support_assistant.logs import error_fields
from support_assistant.redis_factory import make_redis
from support_assistant.resilience import backoff_delay

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
    except ExternalServiceUnavailable as exc:
        log.warning("notification_send_failed", will_retry=True, **error_fields(exc))
        return Outcome.RETRY
    except SupportError as exc:
        log.warning("notification_send_failed", will_retry=False, **error_fields(exc))
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


RETRY_BASE_S = 2.0
RETRY_CAP_S = 60.0
MAX_ATTEMPTS = 5
DEDUPE_TTL_S = 24 * 3600


Headers = Sequence[tuple[str, bytes]]
DeadLetter = Callable[[bytes, Headers], Awaitable[None]]


def message_id_of(headers: Headers) -> str | None:
    return next(
        (value.decode(errors="replace") for name, value in headers if name == "message_id"), None
    )


class SentLog:
    """Which message ids were already sent. Redis being down never blocks a notification."""

    def __init__(self, redis: Any, ttl_s: int = DEDUPE_TTL_S) -> None:
        self._redis, self._ttl = redis, ttl_s

    async def seen(self, message_id: str) -> bool:
        try:
            return bool(await self._redis.exists(f"notification:sent:{message_id}"))
        except Exception as exc:  # noqa: BLE001
            log.warning("notification_dedupe_unavailable", **error_fields(exc))
            return False

    async def record(self, message_id: str) -> None:
        try:
            await self._redis.set(f"notification:sent:{message_id}", "1", ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            log.warning("notification_dedupe_unavailable", **error_fields(exc))


async def process_message(
    value: bytes,
    headers: Headers,
    handle: Callable[[str], Awaitable[Outcome]],
    dead_letter: DeadLetter,
    sent: SentLog,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Outcome:
    """Finish one message: send it, skip it as a repeat, or dead-letter it. Returns only when it is
    safe to commit. If dead-lettering fails this raises, so the offset is not committed and the
    message is read again."""
    message_id = message_id_of(headers)
    if message_id and await sent.seen(message_id):
        log.info("notification_duplicate_skipped")
        return Outcome.COMPLETE
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            outcome = await handle(value.decode(errors="replace"))
        except Exception as exc:  # noqa: BLE001 - e.g. the database: try again, keep working
            log.error("notification_handler_failed", **error_fields(exc))
            outcome = Outcome.RETRY
        if outcome is not Outcome.RETRY or attempt == MAX_ATTEMPTS:
            break
        # Without this pause an outage of a few seconds would use up every attempt.
        delay = RETRY_BASE_S + backoff_delay(attempt, RETRY_BASE_S, RETRY_CAP_S)
        log.warning("notification_backoff", attempt=attempt, delay_s=round(delay, 1))
        await sleep(delay)
    if outcome is Outcome.COMPLETE:
        if message_id:
            await sent.record(message_id)
        return Outcome.COMPLETE
    log.warning("notification_dead_lettered", retries_exhausted=outcome is Outcome.RETRY)
    await dead_letter(value, headers)
    return Outcome.DEAD_LETTER


async def run() -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition

    from support_assistant.kafka_client import connection_options, new_credential

    settings = get_settings()
    if not settings.kafka_bootstrap_servers:
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS is required")
    pool = make_pool(settings)
    await pool.open()
    sender = _sms_sender(settings)
    email_sender = _email_sender(settings)
    redis = make_redis(settings.redis_url.get_secret_value(), settings)
    sent = SentLog(redis)

    async def lookup(customer_id: str) -> str:
        async with pool.connection() as conn:
            return str((await repo.get_customer(conn, customer_id))["phone"])

    async def lookup_email(customer_id: str) -> str | None:
        async with pool.connection() as conn:
            email = (await repo.get_customer(conn, customer_id)).get("email")
        return str(email) if email else None

    async def handle(raw: str) -> Outcome:
        return await handle_message(
            raw, lookup, sender, lookup_email=lookup_email, send_email=email_sender
        )

    async with new_credential() as credential:
        options = connection_options(settings.kafka_bootstrap_servers, credential)
        producer = AIOKafkaProducer(**options, acks="all")
        consumer = AIOKafkaConsumer(
            settings.kafka_topic,
            **options,
            group_id=settings.kafka_consumer_group,
            enable_auto_commit=False,
            auto_offset_reset="earliest",  # a new group must not skip waiting notifications
            max_poll_records=1,
        )

        async def dead_letter(value: bytes, headers: Headers) -> None:
            await producer.send_and_wait(
                settings.kafka_dead_letter_topic, value, headers=list(headers)
            )

        await producer.start()
        try:
            await consumer.start()
            try:
                async for m in consumer:
                    await process_message(m.value or b"", m.headers, handle, dead_letter, sent)
                    partition = TopicPartition(m.topic, m.partition)
                    try:
                        await consumer.commit({partition: m.offset + 1})
                    except Exception as exc:  # noqa: BLE001 - e.g. a rebalance: it is read again
                        log.error("notification_commit_failed", **error_fields(exc))
            finally:
                await consumer.stop()
        finally:
            await producer.stop()
            await redis.aclose()
            await pool.close()


if __name__ == "__main__":
    asyncio.run(run())
