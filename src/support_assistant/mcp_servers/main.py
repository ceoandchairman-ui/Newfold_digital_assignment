"""Launcher: one container, three MCP servers on three ports (D12, E8-8).

    python -m support_assistant.mcp_servers.main

database (8001), comms (8002) and device_diagnostics (8003) share this process and the Postgres
pool. Each is built by its own factory in this package; this file only wires and serves them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
import uvicorn
from mcp.server.fastmcp import FastMCP

from support_assistant.config import Settings, get_settings
from support_assistant.db import make_pool
from support_assistant.mcp_servers.comms import build_comms_server
from support_assistant.mcp_servers.database import build_database_server
from support_assistant.mcp_servers.device import build_device_server
from support_assistant.notifications import LogOutbox, Outbox, ServiceBusOutbox
from support_assistant.redis_factory import make_redis
from support_assistant.services.device_sim import RedisDeviceStore
from support_assistant.services.payments import StubGateway
from support_assistant.telemetry import instrument_asgi, setup_telemetry

log = structlog.get_logger()


def build_outbox(settings: Settings) -> Outbox:
    """Service Bus when configured and not in stub mode; otherwise log-only."""
    if settings.servicebus_namespace_fqdn and not (
        settings.use_stub_sms and settings.use_stub_email
    ):
        return ServiceBusOutbox(settings.servicebus_namespace_fqdn)
    return LogOutbox()


async def supervise(
    name: str,
    run: Callable[[], Awaitable[None]],
    *,
    max_backoff_s: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_restarts: int | None = None,
) -> None:
    """Keep one server running: if it stops or crashes, restart it with exponential backoff.

    Each server is supervised on its own, so one crashing never takes down the other two (E8-8).
    Cancellation (pod shutdown) is passed through and ends supervision.
    """
    restarts = 0
    while max_restarts is None or restarts <= max_restarts:
        try:
            await run()
            log.warning("mcp_server_exited", server=name)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("mcp_server_crashed", server=name)
        restarts += 1
        await sleep(min(2 ** min(restarts, 10), max_backoff_s))


async def _serve(server: FastMCP, port: int) -> None:
    app = instrument_asgi(server.streamable_http_app())  # joins the calling agent's trace
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    await uvicorn.Server(config).serve()


async def main() -> None:
    settings = get_settings()
    setup_telemetry("mcp-servers")
    pool = make_pool(settings)
    gateway = StubGateway()  # the real payment gateway is an open decision (use_stub_gateway)
    servers = [
        build_database_server(pool, gateway),
        build_comms_server(build_outbox(settings)),
        build_device_server(
            RedisDeviceStore(make_redis(settings.redis_url.get_secret_value(), settings))
        ),
    ]
    await pool.open()
    try:
        await asyncio.gather(
            *(
                supervise(s.name, lambda s=s: _serve(s, s.settings.port))  # type: ignore[misc]
                for s in servers
            )
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
