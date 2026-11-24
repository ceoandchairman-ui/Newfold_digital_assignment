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
            "get_payment_methods",
        }
    ),
)
COMMS = ServerSpec("comms", 8002, frozenset({"send_notification"}))
# Who the customer is and what they hold: profile, devices, contract. Read-only.
ACCOUNT = ServerSpec(
    "account_management", 8004, frozenset({"get_customer_profile", "get_devices", "get_contract"})
)
# What we already know about the customer from past support: cases and the last days of chat.
CRM = ServerSpec("crm", 8005, frozenset({"get_case_history", "get_recent_interactions"}))
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

# Third-party providers. Each runs a stub until its credentials exist (see config.py).
TWILIO = ServerSpec("twilio_sms", 8006, frozenset({"send_sms", "get_message_status"}))
STRIPE = ServerSpec("stripe", 8007, frozenset({"get_payment_status", "reconcile_payment_attempt"}))
GMAIL = ServerSpec("gmail", 8008, frozenset({"send_email", "get_support_thread"}))
DRIVE = ServerSpec("google_drive", 8009, frozenset({"search_documents", "read_document"}))

ALL_SERVERS: tuple[ServerSpec, ...] = (
    DATABASE, COMMS, DEVICE, ACCOUNT, CRM, TWILIO, STRIPE, GMAIL, DRIVE,
)  # fmt: skip

# Tools that change state. The client never retries these on its own: a retry after a lost reply
# could send a second SMS. (process_payment is idempotent by key, but is still not auto-retried.)
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "process_payment",
        "send_notification",
        "send_sms",
        "send_email",
        "reconcile_payment_attempt",
    }
)


def server_urls(settings: Settings) -> Mapping[str, str]:
    """Where each server is reached from other pods (ClusterIP DNS names, from configuration)."""
    return {
        DATABASE.name: settings.mcp_database_url,
        COMMS.name: settings.mcp_comms_url,
        DEVICE.name: settings.mcp_device_url,
        ACCOUNT.name: settings.mcp_account_url,
        CRM.name: settings.mcp_crm_url,
        TWILIO.name: settings.mcp_twilio_url,
        STRIPE.name: settings.mcp_stripe_url,
        GMAIL.name: settings.mcp_gmail_url,
        DRIVE.name: settings.mcp_drive_url,
    }
