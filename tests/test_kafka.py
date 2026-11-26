import json
import re
from pathlib import Path

import pytest

from support_assistant.kafka_client import connection_options, event_hubs_host
from support_assistant.notifications import KafkaOutbox, queue_message_id

SERVERS = "evh-x.servicebus.windows.net:9093"


class Credential:
    def __init__(self) -> None:
        self.scopes: list[str] = []

    async def get_token(self, scope):
        self.scopes.append(scope)
        return type("Token", (), {"token": "entra-token"})()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


async def test_the_connection_signs_in_with_entra_over_tls():
    credential = Credential()
    options = connection_options(SERVERS, credential)
    assert options["security_protocol"] == "SASL_SSL"
    assert options["sasl_mechanism"] == "OAUTHBEARER"
    assert await options["sasl_oauth_token_provider"].token() == "entra-token"
    assert credential.scopes == ["https://evh-x.servicebus.windows.net/.default"]


@pytest.mark.parametrize(
    "servers",
    [
        "kafka.example.com:9093",
        "evh-x.servicebus.windows.net.evil.com:9093",
        "evh-x.servicebus.windows.net:9092",
        f"{SERVERS},kafka.example.com:9093",
    ],
)
def test_the_token_is_never_offered_to_another_server(servers):
    with pytest.raises(RuntimeError):
        event_hubs_host(servers)


async def test_the_outbox_publishes_keyed_by_customer_with_the_message_id(monkeypatch):
    seen: dict[str, object] = {}

    class Producer:
        def __init__(self, **options) -> None:
            seen["acks"] = options["acks"]

        async def start(self):
            pass

        async def send_and_wait(self, topic, value, key, headers):
            seen.update(topic=topic, value=json.loads(value), key=key, headers=headers)

        async def stop(self):
            seen["stopped"] = True

    monkeypatch.setattr("aiokafka.AIOKafkaProducer", Producer)
    monkeypatch.setattr("support_assistant.kafka_client.new_credential", Credential)
    await KafkaOutbox(SERVERS).enqueue("C1", "sms", "payment_received", "tx-1")
    expected = queue_message_id("C1", "sms", "payment_received", "tx-1")
    assert seen == {
        "acks": "all",
        "topic": "notifications",
        "value": {"customer_id": "C1", "channel": "sms", "template": "payment_received"},
        "key": b"C1",
        "headers": [("message_id", expected.encode())],
        "stopped": True,
    }


def test_the_worker_may_use_the_redis_cluster_it_deduplicates_in():
    tf = (Path(__file__).resolve().parents[1] / "infra" / "terraform" / "redis.tf").read_text(
        encoding="utf-8"
    )
    line = next(x for x in tf.splitlines() if re.match(r"\s*cache\s*=\s*\[", x))
    assert '"notification-worker"' in line
