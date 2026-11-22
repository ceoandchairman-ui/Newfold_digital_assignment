"""MCP server `device_diagnostics` (E8-5, E3-8): the device troubleshooting tools.

Backed by the deterministic device simulation (services/device_sim.py) with per-customer,
per-device state. All tools are idempotent, so a retried call is always safe.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant.mcp_catalog import DEVICE
from support_assistant.mcp_servers.common import safe
from support_assistant.services.device_sim import (
    DeviceSimulation,
    DeviceStateStore,
    InMemoryDeviceStore,
)


def build_device_server(
    store: DeviceStateStore | None = None, *, host: str = "0.0.0.0", port: int = DEVICE.port
) -> FastMCP:
    sim = DeviceSimulation(store or InMemoryDeviceStore())
    server = FastMCP(DEVICE.name, host=host, port=port)

    async def run(fn: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return {"ok": True, **(await fn(**kwargs))}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def run_diagnostics(customer_id: str, device_id: str) -> dict[str, Any]:
        """Full device status: service, data, SIM, APN and signal."""
        return await run(sim.check_status, customer_id=customer_id, device_id=device_id)

    @server.tool()
    async def toggle_airplane_mode(
        customer_id: str, device_id: str, enabled: bool
    ) -> dict[str, Any]:
        """Set airplane mode on or off."""
        return await run(
            sim.toggle_airplane_mode, customer_id=customer_id, device_id=device_id, enabled=enabled
        )

    @server.tool()
    async def toggle_mobile_data(customer_id: str, device_id: str, enabled: bool) -> dict[str, Any]:
        """Set mobile data on or off."""
        return await run(
            sim.toggle_mobile_data, customer_id=customer_id, device_id=device_id, enabled=enabled
        )

    @server.tool()
    async def toggle_roaming(customer_id: str, device_id: str, enabled: bool) -> dict[str, Any]:
        """Set data roaming on or off."""
        return await run(
            sim.toggle_roaming, customer_id=customer_id, device_id=device_id, enabled=enabled
        )

    @server.tool()
    async def reseat_sim_card(customer_id: str, device_id: str) -> dict[str, Any]:
        """Record that the customer re-inserted the SIM card."""
        return await run(sim.reseat_sim_card, customer_id=customer_id, device_id=device_id)

    @server.tool()
    async def reset_apn_settings(customer_id: str, device_id: str) -> dict[str, Any]:
        """Restore the carrier's default APN."""
        return await run(sim.reset_apn_settings, customer_id=customer_id, device_id=device_id)

    @server.tool()
    async def reboot_device(customer_id: str, device_id: str) -> dict[str, Any]:
        """Restart the device."""
        return await run(sim.reboot_device, customer_id=customer_id, device_id=device_id)

    @server.tool()
    async def can_send_mms(customer_id: str, device_id: str) -> dict[str, Any]:
        """Whether picture messages can be sent right now, and why not."""
        return await run(sim.can_send_mms, customer_id=customer_id, device_id=device_id)

    return server
