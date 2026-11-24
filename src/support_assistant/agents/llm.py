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
    REASONING = "reasoning"  # only when a reasoning deployment is configured


# A turn is "hard" when the model must reason over several facts or a proposed action.
HARD_FACTS_CHARS = 600
HARD_MESSAGE_CHARS = 400
_HARD_MARKERS = ("proposed:", "overdue", "dispute", "refund", "cancel", "why ")
# The subset that needs real multi-step reasoning (policy and amounts), worth the extra seconds.
_REASONING_MARKERS = ("dispute", "refund", "cancel")


def pick_tier(facts: str, user_message: str, *, reasoning_available: bool = False) -> Tier:
    """Deterministic, cheap routing signal. Easy turns go to the small model."""
    text = f"{facts} {user_message}".lower()
    if reasoning_available and any(marker in text for marker in _REASONING_MARKERS):
        return Tier.REASONING
    if len(facts) > HARD_FACTS_CHARS or len(user_message) > HARD_MESSAGE_CHARS:
        return Tier.LARGE
    if any(marker in text for marker in _HARD_MARKERS):
        return Tier.LARGE
    return Tier.SMALL


class TieredLLM:
    """Routes each call to a small, large or (optionally) reasoning model, inside one agent.

    The tier belongs to the call, not to the agent: a MAF `Agent` is cheap to build over any chat
    client, so the router holds one client per tier. If the small or reasoning model fails, the
    large one is tried, so a tier outage degrades cost or depth, not availability.
    """

    def __init__(self, small: LLM, large: LLM, reasoning: LLM | None = None) -> None:
        self._by_tier: dict[Tier, LLM] = {Tier.SMALL: small, Tier.LARGE: large}
        if reasoning is not None:
            self._by_tier[Tier.REASONING] = reasoning
        self.last_tier: Tier | None = None  # for tests and telemetry

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        tier = pick_tier(facts, user_message, reasoning_available=Tier.REASONING in self._by_tier)
        self.last_tier = tier
        if tier is not Tier.LARGE:
            try:
                return await self._by_tier[tier].respond(instructions, facts, user_message)
            except Exception:  # noqa: BLE001 - any non-large failure falls back to the large model
                self.last_tier = Tier.LARGE
        return await self._by_tier[Tier.LARGE].respond(instructions, facts, user_message)

    async def respond_with_tools(
        self, instructions: str, user_message: str, tools: list[Any]
    ) -> str:
        """Tool calling always uses the large tier: choosing tools is the hard part."""
        large: Any = self._by_tier[Tier.LARGE]
        result: str = await large.respond_with_tools(instructions, user_message, tools)
        return result


def build_llm(settings: Settings, deployment: str, *, use_reasoning: bool = True) -> LLM:
    """`deployment` is the agent's large model; the small tier is shared (`chat-small`)."""
    if not settings.azure_openai_endpoint:
        return TemplateLLM()
    large = MafLLM(settings, deployment)
    if not settings.azure_openai_small_deployment:
        return large
    reasoning = (
        MafLLM(settings, settings.azure_openai_reasoning_deployment)
        if use_reasoning and settings.azure_openai_reasoning_deployment
        else None
    )
    return TieredLLM(MafLLM(settings, settings.azure_openai_small_deployment), large, reasoning)


def build_support_llms(settings: Settings, large_deployment: str) -> dict[str, LLM]:
    """One model per sub-agent of the support_agent (E4-3, E4-4). The graph picks the sub-agent.

    route   -> small model: a short, validated classification.
    normal  -> small model: simple dialogue and plain questions.
    instruct-> the reasoning deployment when configured, otherwise the large model.
    billing -> the large model with the small tier for easy turns (tool-driven, proposes payments).
    """
    if not settings.azure_openai_endpoint:
        return {k: TemplateLLM() for k in ("route", "normal", "instruct", "billing")}
    small = MafLLM(settings, settings.azure_openai_small_deployment or large_deployment)
    instruct_deployment = settings.azure_openai_reasoning_deployment or large_deployment
    return {
        "route": small,
        "normal": small,
        "instruct": MafLLM(settings, instruct_deployment),
        "billing": build_llm(settings, large_deployment, use_reasoning=False),
    }
