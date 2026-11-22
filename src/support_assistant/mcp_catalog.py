"""The MCP contract both sides share: which servers exist, on which ports, serving which tools.

The servers (mcp_servers/) build themselves from this and the agent-side registry
(agents/mcp_client/) routes by it. A contract test checks that each running server exposes exactly
the tools listed here, so the two sides cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from support_assistant.config import Settings


@dataclass(frozen=True)
class ServerSpec:
    name: str
    port: int
    tools: frozenset[str]


DATABASE = ServerSpec(
    "database",
    8001,
    frozenset(
        {
            "get_account_summary",
            "get_bills",
            "check_suspension_lift",
            "check_recent_payments",
            "process_payment",
        }
    ),
)
COMMS = ServerSpec("comms", 8002, frozenset({"send_notification"}))
DEVICE = ServerSpec(
    "device_diagnostics",
    8003,
    frozenset(
        {
            "run_diagnostics",
            "toggle_airplane_mode",
            "toggle_mobile_data",
            "toggle_roaming",
            "reseat_sim_card",
            "reset_apn_settings",
            "reboot_device",
            "can_send_mms",
        }
    ),
)

ALL_SERVERS: tuple[ServerSpec, ...] = (DATABASE, COMMS, DEVICE)

# Tools that change state. The client never retries these on its own: a retry after a lost reply
# could send a second SMS. (process_payment is idempotent by key, but is still not auto-retried.)
WRITE_TOOLS: frozenset[str] = frozenset({"process_payment", "send_notification"})


def server_urls(settings: Settings) -> Mapping[str, str]:
    """Where each server is reached from other pods (ClusterIP DNS names, from configuration)."""
    return {
        DATABASE.name: settings.mcp_database_url,
        COMMS.name: settings.mcp_comms_url,
        DEVICE.name: settings.mcp_device_url,
    }
