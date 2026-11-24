"""AgentClient: the only way the graph talks to an agent (D17, E4-11).

Swapping HTTP-hosted MAF agents for in-process ones (or fakes in tests) changes nothing in the graph.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from support_assistant.agents.mcp_client import ToolRegistry
from support_assistant.agents.mcp_client.registry import CircuitBreaker
from support_assistant.config import Settings
from support_assistant.contracts import AgentName, AgentRequest, AgentResponse
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.observability import AGENT_CIRCUIT_OPENED, outgoing_headers


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

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.2, max=2),
        reraise=True,
    )
    async def _post(self, url: str, body: dict[str, Any]) -> httpx.Response:
        return await self._http.post(f"{url}/run", json=body, headers=outgoing_headers())

    async def run(self, agent: AgentName, req: AgentRequest) -> AgentResponse:
        breaker = self._breakers[agent]
        if breaker.is_open:
            raise ExternalServiceUnavailable(f"agent {agent} is unavailable (circuit open)")
        try:
            resp = await self._post(self._urls[agent], req.model_dump())
        except httpx.TransportError as exc:
            self._record_failure(agent, breaker)
            raise ExternalServiceUnavailable(f"agent {agent} unreachable") from exc
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

    async def aclose(self) -> None:
        await self._http.aclose()


class InProcessAgentClient:
    """Runs the agent logic in this process (D17 allows moving an agent between in-process and remote).

    Used by the journey tests and as a fallback; production uses HttpAgentClient. Each agent gets
    its own tool registry and LLM, exactly as its own service would.
    """

    def __init__(
        self,
        registries: dict[str, ToolRegistry],
        llms: dict[str, Any],
        search: Any | None = None,
    ) -> None:
        self._registries, self._llms, self._search = registries, llms, search

    async def run(self, agent: AgentName, req: AgentRequest) -> AgentResponse:
        from support_assistant.agents import roles
        from support_assistant.agents.harness import Harness

        async def no_search(_: str) -> list[Any]:
            return []

        harness = Harness(agent, roles.ALLOWED_TOOLS[agent], self._registries[agent])
        if agent == "director":
            mode = req.context.get("mode", "billing")
            llm = self._llms.get(f"director:{mode}", self._llms[agent])
            if mode == "route":
                return await roles.support_route(req, harness, self._llms.get("route_classifier"))
            if mode == "normal":
                return await roles.normal_resp_gen(req, harness, llm)
            if mode == "instruct":
                return await roles.instruct_gen(req, harness, llm)
            return await roles.billing_gen(req, harness, llm)
        if agent == "evaluator":
            return await roles.evaluator(req, harness)
        return await roles.troubleshooter(
            req, harness, self._llms[agent], self._search or no_search
        )
