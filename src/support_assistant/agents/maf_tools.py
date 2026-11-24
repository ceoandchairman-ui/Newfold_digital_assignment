"""Expose harness-guarded tools to a Microsoft Agent Framework agent (E5-7).

When a MAF agent calls tools itself (model-driven tool calling), every call must still pass the
harness: verified customer, allow-list, identity injection, audit. So each MAF `FunctionTool` here
is a thin wrapper whose function is `harness.call(...)`.

The model never sees or supplies `customer_id`: it is not in any tool's input schema, and the
harness injects it from the verified request. Only the agent's allow-listed tools are built.
Checked against agent-framework 1.22 (`FunctionTool(name, description, func, input_model)`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from support_assistant.agents.harness import Harness
from support_assistant.contracts import AgentRequest


class NoArgs(BaseModel):
    """Tools that act on the verified customer only."""


class RecentPaymentsArgs(BaseModel):
    bill_id: str = Field(min_length=1)
    minutes: int = Field(10, ge=1, le=1440)


class CaseHistoryArgs(BaseModel):
    limit: int = Field(5, ge=1, le=10)


class RecentInteractionsArgs(BaseModel):
    days: int = Field(5, ge=1, le=30)


class DiagnosticsArgs(BaseModel):
    device_id: str = Field(min_length=1)


class ToggleArgs(BaseModel):
    device_id: str = Field(min_length=1)
    enabled: bool


# What the model may pass to each tool. customer_id is deliberately absent from all of them.
TOOL_INPUTS: dict[str, tuple[type[BaseModel], str]] = {
    "get_account_summary": (NoArgs, "Lines, plans and data usage for the verified customer."),
    "get_bills": (NoArgs, "All bills of the verified customer, with status."),
    "check_suspension_lift": (NoArgs, "Whether the customer's suspension may be lifted."),
    "get_customer_profile": (NoArgs, "Name, phone and masked email of the verified customer."),
    "get_contract": (NoArgs, "Per line: plan, status, contract end date and whether it expired."),
    "get_devices": (NoArgs, "The customer's devices (model, line, last four of the IMEI)."),
    "get_payment_methods": (NoArgs, "The customer's payment methods (kind, last four digits)."),
    "get_case_history": (CaseHistoryArgs, "The customer's past resolved support issues."),
    "get_recent_interactions": (
        RecentInteractionsArgs,
        "The customer's conversations from the last N days (default 5), across sessions.",
    ),
    "check_recent_payments": (RecentPaymentsArgs, "Payments for a bill in the last N minutes."),
    "run_diagnostics": (DiagnosticsArgs, "Run diagnostics on one of the customer's devices."),
    "toggle_airplane_mode": (ToggleArgs, "Turn airplane mode on or off on a device."),
    "toggle_mobile_data": (ToggleArgs, "Turn mobile data on or off on a device."),
    "toggle_roaming": (ToggleArgs, "Turn data roaming on or off on a device."),
    "reseat_sim_card": (DiagnosticsArgs, "Record that the SIM card was re-inserted."),
    "reset_apn_settings": (DiagnosticsArgs, "Restore the carrier's default APN settings."),
    "reboot_device": (DiagnosticsArgs, "Restart a device."),
    "can_send_mms": (DiagnosticsArgs, "Check whether picture messages can be sent, and why not."),
}


def build_maf_tools(harness: Harness, req: AgentRequest, allowed: frozenset[str]) -> list[Any]:
    """FunctionTools for the allow-listed tools only. Write tools are never offered to a model."""
    from agent_framework import FunctionTool

    tools = []
    for name in sorted(allowed):
        if name not in TOOL_INPUTS:
            continue  # e.g. process_payment: never model-callable; it runs only after approval
        model, description = TOOL_INPUTS[name]

        def make(tool_name: str) -> Any:
            async def run(**kwargs: Any) -> dict[str, Any]:
                kwargs.pop("customer_id", None)  # belt and braces: the harness overrides it anyway
                return await harness.call(req, tool_name, **kwargs)

            return run

        tools.append(
            FunctionTool(name=name, description=description, func=make(name), input_model=model)
        )
    return tools
