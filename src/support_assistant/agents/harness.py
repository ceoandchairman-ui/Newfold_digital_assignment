"""Shared agent harness (E5-7): the rules every agent service enforces, independent of the LLM.

- auth guard: no tool runs for an unverified customer
- identity-from-state: customer_id always comes from the request, never from model output
- tool allow-list: each agent can only call its own tools (the evaluator has no write tools)
- audit: every tool call is recorded

The harness decides WHETHER a call may happen. The MCP client registry (agents/mcp_client) decides
WHERE it goes. Tools are only ever reached through the registry.
"""

from __future__ import annotations

from typing import Any

import structlog

from support_assistant.agents.mcp_client import ToolRegistry
from support_assistant.contracts import AgentRequest
from support_assistant.errors import AuthenticationError, AuthorizationError
from support_assistant.observability import TOOL_CALLS

log = structlog.get_logger()


class Harness:
    def __init__(self, agent: str, allowed_tools: frozenset[str], registry: ToolRegistry) -> None:
        self.agent = agent
        self._allowed = allowed_tools
        self._registry = registry
        self.audit: list[dict[str, Any]] = []

    def require_verified(self, req: AgentRequest) -> None:
        if not req.identity_verified:
            raise AuthenticationError("identity not verified")

    async def call(self, req: AgentRequest, tool: str, **args: Any) -> dict[str, Any]:
        self.require_verified(req)
        if tool not in self._allowed or not self._registry.has(tool):
            raise AuthorizationError(f"{self.agent} may not call {tool}")
        args["customer_id"] = req.customer_id  # identity from state, overriding the model's value
        result = await self._registry.call(tool, args)
        ok = bool(result.get("ok"))
        entry = {"agent": self.agent, "tool": tool, "ok": ok}
        self.audit.append(entry)
        TOOL_CALLS.add(1, {"agent": self.agent, "tool": tool, "ok": ok})
        log.info("tool_call", session_id=req.session_id, **entry)
        return result
