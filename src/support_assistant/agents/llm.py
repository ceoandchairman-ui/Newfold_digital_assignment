"""LLM seam for the agent services.

`MafLLM` runs a Microsoft Agent Framework agent against Azure OpenAI with Entra auth.
`TieredLLM` routes each call to a small or large model. `TemplateLLM` keeps the services runnable
and fully testable without any model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from support_assistant.config import Settings


class LLM(Protocol):
    async def respond(self, instructions: str, facts: str, user_message: str) -> str: ...


class TemplateLLM:
    """Deterministic wording from facts. No model call; used in tests and as a safe fallback."""

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        return facts or "How can I help you today?"


class MafLLM:
    """Wraps a Microsoft Agent Framework agent (agent-framework 1.22: `Agent` + `OpenAIChatClient`).

    The import paths and signatures were checked against the installed package. Behaviour against a
    real Azure OpenAI deployment (Entra token flow, api_version) is still unverified (E4-0).
    """

    def __init__(self, settings: Settings, deployment: str) -> None:
        from agent_framework.openai import OpenAIChatClient
        from azure.identity import DefaultAzureCredential

        self._client = OpenAIChatClient(
            model=deployment,  # the Azure deployment name
            azure_endpoint=settings.azure_openai_endpoint,
            credential=DefaultAzureCredential(),
            api_version=settings.azure_openai_api_version,
        )

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        from agent_framework import Agent

        agent = Agent(self._client, instructions)
        prompt = f"Customer said: {user_message}\n\nVerified facts (do not invent others):\n{facts}"
        result = await agent.run(prompt)
        return str(result.text)


class Tier(StrEnum):
    SMALL = "small"
    LARGE = "large"


# A turn is "hard" when the model must reason over several facts or a proposed action.
HARD_FACTS_CHARS = 600
HARD_MESSAGE_CHARS = 400
_HARD_MARKERS = ("proposed:", "overdue", "dispute", "refund", "cancel", "why ")


def pick_tier(facts: str, user_message: str) -> Tier:
    """Deterministic, cheap routing signal. Easy turns go to the small model."""
    text = f"{facts} {user_message}".lower()
    if len(facts) > HARD_FACTS_CHARS or len(user_message) > HARD_MESSAGE_CHARS:
        return Tier.LARGE
    if any(marker in text for marker in _HARD_MARKERS):
        return Tier.LARGE
    return Tier.SMALL


class TieredLLM:
    """Routes each call to a small or large model (multi-tier LLMs inside one agent).

    The tier belongs to the call, not to the agent: a MAF `Agent` is cheap to build over any chat
    client, so the router holds one client per tier. If the small model fails, the large one is
    tried, so a cheap-tier outage degrades cost, not availability.
    """

    def __init__(self, small: LLM, large: LLM) -> None:
        self._by_tier = {Tier.SMALL: small, Tier.LARGE: large}
        self.last_tier: Tier | None = None  # for tests and telemetry

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        tier = pick_tier(facts, user_message)
        self.last_tier = tier
        if tier is Tier.SMALL:
            try:
                return await self._by_tier[Tier.SMALL].respond(instructions, facts, user_message)
            except Exception:  # noqa: BLE001 - any small-tier failure falls back to the large model
                self.last_tier = Tier.LARGE
        return await self._by_tier[Tier.LARGE].respond(instructions, facts, user_message)

    async def respond_with_tools(
        self, instructions: str, user_message: str, tools: list[Any]
    ) -> str:
        """Tool calling always uses the large tier: choosing tools is the hard part."""
        large: Any = self._by_tier[Tier.LARGE]
        result: str = await large.respond_with_tools(instructions, user_message, tools)
        return result


def build_llm(settings: Settings, deployment: str) -> LLM:
    """`deployment` is the agent's large model; the small tier is shared (`chat-small`)."""
    if not settings.azure_openai_endpoint:
        return TemplateLLM()
    large = MafLLM(settings, deployment)
    if not settings.azure_openai_small_deployment:
        return large
    return TieredLLM(MafLLM(settings, settings.azure_openai_small_deployment), large)
