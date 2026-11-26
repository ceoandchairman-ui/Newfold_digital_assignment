"""Notification outbox (E6/E9): publish to Kafka, a worker sends through Communication Services.

Stub mode (use_stub_sms / use_stub_email) logs instead of sending, so nothing real is contacted.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Protocol

import structlog

log = structlog.get_logger()


class Outbox(Protocol):
    async def enqueue(
        self, customer_id: str, channel: str, template: str, event_id: str | None = None
    ) -> None: ...


def queue_message_id(customer_id: str, channel: str, template: str, event_id: str | None) -> str:
    """The id the worker uses to drop duplicates. With an `event_id` (what the notification is
    about, e.g. a payment id) a repeat of that event is dropped and a new event is not. Without one
    every call is a new message: guessing from the template would silently drop a second, real
    notification of the same kind."""
    if not event_id:
        return uuid.uuid4().hex
    return hashlib.sha256(f"{customer_id}|{channel}|{template}|{event_id}".encode()).hexdigest()


class LogOutbox:
    async def enqueue(
        self, customer_id: str, channel: str, template: str, event_id: str | None = None
    ) -> None:
        log.info("notification_stub", customer_id=customer_id, channel=channel, template=template)


class KafkaOutbox:
    def __init__(self, bootstrap_servers: str, topic: str = "notifications") -> None:
        self._servers, self._topic = bootstrap_servers, topic

    async def enqueue(
        self, customer_id: str, channel: str, template: str, event_id: str | None = None
    ) -> None:
        from aiokafka import AIOKafkaProducer

        from support_assistant.kafka_client import connection_options, new_credential

        message_id = queue_message_id(customer_id, channel, template, event_id)
        body = json.dumps({"customer_id": customer_id, "channel": channel, "template": template})
        async with new_credential() as credential:
            producer = AIOKafkaProducer(**connection_options(self._servers, credential), acks="all")
            await producer.start()
            try:
                # Keyed by customer, so one customer's messages stay in order. Kafka has no
                # duplicate detection: the worker drops a repeat by its message_id header.
                await producer.send_and_wait(
                    self._topic,
                    body.encode(),
                    key=customer_id.encode(),
                    headers=[("message_id", message_id.encode())],
                )
            finally:
                await producer.stop()
