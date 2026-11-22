"""Notification outbox (E6/E9): enqueue to Service Bus, a worker sends through Communication Services.

Stub mode (use_stub_sms / use_stub_email) logs instead of sending, so nothing real is contacted.
"""

from __future__ import annotations

import json
from typing import Protocol

import structlog

log = structlog.get_logger()


class Outbox(Protocol):
    async def enqueue(self, customer_id: str, channel: str, template: str) -> None: ...


class LogOutbox:
    async def enqueue(self, customer_id: str, channel: str, template: str) -> None:
        log.info("notification_stub", customer_id=customer_id, channel=channel, template=template)


class ServiceBusOutbox:
    def __init__(self, namespace_fqdn: str, queue: str = "notifications") -> None:
        self._ns, self._queue = namespace_fqdn, queue

    async def enqueue(self, customer_id: str, channel: str, template: str) -> None:
        from azure.identity.aio import DefaultAzureCredential
        from azure.servicebus import ServiceBusMessage
        from azure.servicebus.aio import ServiceBusClient

        body = json.dumps({"customer_id": customer_id, "channel": channel, "template": template})
        async with DefaultAzureCredential() as cred, ServiceBusClient(self._ns, cred) as client:
            async with client.get_queue_sender(self._queue) as sender:
                # message_id enables the queue's duplicate detection
                await sender.send_messages(
                    ServiceBusMessage(body, message_id=f"{customer_id}:{channel}:{template}")
                )
