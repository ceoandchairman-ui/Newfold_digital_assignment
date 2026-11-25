"""AgentClient: the only way the graph talks to an agent (D17, E4-11).

Swapping HTTP-hosted MAF agents for in-process ones (or fakes in tests) changes nothing in the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import structlog

from support_assistant.agents.mcp_client import ToolRegistry
from support_assistant.agents.mcp_client.registry import CircuitBreaker
from support_assistant.config import Settings
from support_assistant.contracts import (
    AgentName,
    AgentRequest,
    AgentResponse,
    Evidence,
    EvidenceRef,
    FetchResponse,
)
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields
from support_assistant.observability import AGENT_CIRCUIT_OPENED, outgoing_headers
from support_assistant.resilience import never_sent, retrying
from support_assistant.retrieval_client import Retrieval

log = structlog.get_logger()


class AgentClient(Protocol):
    async def run(self, agent: AgentName, req: AgentRequest) -> AgentResponse: ...


class HttpAgentClient:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._urls: dict[str, str] = {
            "director": settings.agent_director_url,
            "troubleshooter": settings.agent_troubleshooter_url,
            "evaluator": settings.agent_evaluator_url,
        }
        self._http = httpx.AsyncClient(timeout=30.0, transport=transport)
        # One breaker per agent: a dead agent costs microseconds, not a timeout on every turn, and the
        # other agents keep working. Only outages count; a 4xx is the caller's problem, not the agent's.
        self._breakers = {
            name: CircuitBreaker(
                settings.agent_circuit_threshold, settings.agent_circuit_cooldown_seconds
            )
            for name in self._urls
        }

    # An agent run can change a device, so it is repeated ONLY if the request never left. After a
    # read timeout the agent may still be working, and running it again could repeat an action.
    @retrying("agent_run", attempts=3, base=0.2, cap=2.0, retry_on=never_sent)
    async def _post(self, url: str, body: dict[str, Any]) -> httpx.Response:
        return await self._http.post(f"{url}/run", json=body, headers=outgoing_headers())

    async def run(self, agent: AgentName, req: AgentRequest) -> AgentResponse:
        breaker = self._breakers[agent]
        if breaker.is_open:
            log.warning("circuit_rejected_call", dependency=f"agent-{agent}")
            raise ExternalServiceUnavailable(f"agent {agent} is unavailable (circuit open)")
        try:
            resp = await self._post(self._urls[agent], req.model_dump())
        except httpx.TransportError as exc:
            log.error("agent_call_failed", agent=agent, **error_fields(exc))
            self._record_failure(agent, breaker)
            raise ExternalServiceUnavailable(f"agent {agent} unreachable") from exc
        if resp.status_code >= 400:
            log.warning("agent_call_refused", agent=agent, status=resp.status_code)
        if resp.status_code >= 500:
            self._record_failure(agent, breaker)
            raise ExternalServiceUnavailable(f"agent {agent} returned {resp.status_code}")
        breaker.success()
        if resp.status_code >= 400:
            return AgentResponse(reply=str(resp.json().get("detail", "")))
        return AgentResponse.model_validate(resp.json())

    @staticmethod
    def _record_failure(agent: str, breaker: CircuitBreaker) -> None:
        was_open = breaker.is_open
        breaker.failure()
        if breaker.is_open and not was_open:
            AGENT_CIRCUIT_OPENED.add(1, {"agent": agent})
            log.error("circuit_opened", dependency=f"agent-{agent}")

    async def aclose(self) -> None:
        await self._http.aclose()


class EmptyRetrieval:
    """A knowledge base with nothing in it: every search is empty and no reference can be found."""

    async def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        return []

    async def fetch(self, refs: Sequence[EvidenceRef]) -> FetchResponse:
        return FetchResponse(missing=list(refs))


class InProcessAgentClient:
    """Runs the agent logic in this process (D17 allows moving an agent between in-process and remote).

    Used by the journey tests and as a fallback; production uses HttpAgentClient. Each agent gets
    its own tool registry and LLM, exactly as its own service would.
    """

    def __init__(
        self,
        registries: dict[str, ToolRegistry],
        llms: dict[str, Any],
        retrieval: Retrieval | None = None,
    ) -> None:
        self._registries, self._llms = registries, llms
        self._retrieval: Retrieval = retrieval or EmptyRetrieval()

    async def run(self, agent: AgentName, req: AgentRequest) -> AgentResponse:
        from support_assistant.agents import roles
        from support_assistant.agents.harness import Harness

        harness = Harness(agent, roles.ALLOWED_TOOLS[agent], self._registries[agent])
        if agent == "director":
            mode = req.context.get("mode", "billing")
            llm = self._llms.get(f"director:{mode}", self._llms[agent])
            if mode == "route":
                return await roles.support_route(req, harness, self._llms.get("route_classifier"))
            if mode == "normal":
                return await roles.normal_resp_gen(req, harness, llm)
            if mode == "instruct":
                return await roles.instruct_gen(req, harness, llm, self._retrieval)
            return await roles.billing_gen(req, harness, llm)
        if agent == "evaluator":
            return await roles.evaluator(req, harness, retrieval=self._retrieval)
        return await roles.troubleshooter(req, harness, self._llms[agent], self._retrieval)
