"""Kafka connection settings. In Azure the broker is an Event Hubs namespace (its Kafka endpoint,
port 9093) and each pod signs in with its own Entra identity (SASL OAUTHBEARER), so there is no
connection string or password.

NOT VERIFIED against a real Event Hubs namespace.
"""

from __future__ import annotations

import ssl
from typing import Any

from aiokafka.abc import AbstractTokenProvider
from azure.identity.aio import DefaultAzureCredential

EVENT_HUBS_SUFFIX = ".servicebus.windows.net"  # Event Hubs shares this domain


def event_hubs_host(bootstrap_servers: str) -> str:
    """The namespace host from `<namespace>.servicebus.windows.net:9093`.

    Anything else is refused: this app's only sign-in is an Entra token for that host, and it must
    never be handed to another server.
    """
    host, _, port = bootstrap_servers.partition(":")
    if "," in bootstrap_servers or not host.endswith(EVENT_HUBS_SUFFIX) or port != "9093":
        raise RuntimeError(
            f"KAFKA_BOOTSTRAP_SERVERS must be <namespace>{EVENT_HUBS_SUFFIX}:9093 (Entra sign-in)"
        )
    return host


class EntraTokenProvider(AbstractTokenProvider):  # type: ignore[misc]
    def __init__(self, credential: Any, host: str) -> None:
        self._credential, self._scope = credential, f"https://{host}/.default"

    async def token(self) -> str:
        return str((await self._credential.get_token(self._scope)).token)


def connection_options(bootstrap_servers: str, credential: Any) -> dict[str, Any]:
    return {
        "bootstrap_servers": bootstrap_servers,
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "OAUTHBEARER",
        "sasl_oauth_token_provider": EntraTokenProvider(
            credential, event_hubs_host(bootstrap_servers)
        ),
        "ssl_context": ssl.create_default_context(),
        # Event Hubs closes idle connections after 4 minutes: give them up before that.
        "connections_max_idle_ms": 180_000,
        "metadata_max_age_ms": 180_000,
    }


def new_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()
