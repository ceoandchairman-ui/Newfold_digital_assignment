"""LLM seam for the agent services.

`MafLLM` runs a Microsoft Agent Framework agent against Azure OpenAI with Entra auth.
`TemplateLLM` keeps the services runnable and fully testable without any model.
`ResilientLLM` gives each call a deadline and a fallback deployment.
`build_role_llm` and `build_support_llms` pick the model for each job from settings (models.py).
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from support_assistant.config import Settings
from support_assistant.models import Role, model_for
from support_assistant.observability import LLM_FALLBACKS


class LLM(Protocol):
    async def respond(self, instructions: str, facts: str, user_message: str) -> str: ...


class TemplateLLM:
    """Deterministic wording from facts. No model call; used in tests and as a safe fallback."""

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        return facts or "How can I help you today?"


def reasoning_options(effort: str | None) -> dict[str, Any] | None:
    """Agent Framework `default_options` for a reasoning model, or None for a model without effort.

    `reasoning` is a documented key of `OpenAIChatOptions` in the installed agent-framework. No
    temperature is ever sent: reasoning models reject it.
    """
    return {"reasoning": {"effort": effort}} if effort else None


class MafLLM:
    """Wraps a Microsoft Agent Framework agent (`Agent` + `OpenAIChatClient`).

    The import paths and signatures were checked against the installed package. Behaviour against a
    real Azure OpenAI deployment (Entra token flow, api_version, reasoning effort) is still
    unverified (E4-0).
    """

    def __init__(self, settings: Settings, deployment: str, effort: str | None = None) -> None:
        from agent_framework.openai import OpenAIChatClient
        from azure.identity import DefaultAzureCredential

        self._effort = effort
        self._client = OpenAIChatClient(
            model=deployment,  # the Azure deployment name
            azure_endpoint=settings.azure_openai_endpoint,
            credential=DefaultAzureCredential(),
            api_version=settings.azure_openai_api_version,
        )

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        from agent_framework import Agent

        options: Any = reasoning_options(self._effort)
        agent = Agent(self._client, instructions, default_options=options)
        prompt = f"Customer said: {user_message}\n\nVerified facts (do not invent others):\n{facts}"
        result = await agent.run(prompt)
        return str(result.text)


class ResilientLLM:
    """A model call with a deadline and a fallback, so one slow or throttled deployment degrades the
    answer instead of failing the turn.

    The primary gets `timeout_s`. If it raises or times out, the fallback (a different deployment, so
    usually a different quota) gets the same budget. If there is no fallback, or it fails too, the
    error reaches the caller, which already has its own safe behaviour (a template, an escalation).
    """

    def __init__(
        self, primary: LLM, fallback: LLM | None, timeout_s: float, name: str = ""
    ) -> None:
        self._primary, self._fallback, self._timeout, self._name = (
            primary,
            fallback,
            timeout_s,
            name,
        )

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        try:
            return await asyncio.wait_for(
                self._primary.respond(instructions, facts, user_message), self._timeout
            )
        except Exception as exc:  # noqa: BLE001 - includes TimeoutError and provider errors
            if self._fallback is None:
                raise
            LLM_FALLBACKS.add(1, {"role": self._name, "reason": type(exc).__name__})
        return await asyncio.wait_for(
            self._fallback.respond(instructions, facts, user_message), self._timeout
        )


def build_role_llm(settings: Settings, role: Role) -> LLM:
    """The model configured for one role, with its timeout and fallback. Without an OpenAI endpoint,
    a template (no model)."""
    if not settings.azure_openai_endpoint:
        return TemplateLLM()
    spec = model_for(settings, role)
    primary = MafLLM(settings, spec.deployment, spec.effort)
    fallback = None
    if spec.fallback and spec.fallback != spec.deployment:
        fallback = MafLLM(settings, spec.fallback, None)  # no effort: the fallback may be any model
    return ResilientLLM(primary, fallback, settings.llm_timeout_seconds, role.value)


SUPPORT_ROLES = ("route", "normal", "instruct", "billing")


def build_support_llms(settings: Settings) -> dict[str, LLM]:
    """The support_agent has two models (E4-3, E4-4), and the graph picks the sub-agent per turn.

    CHAT    route, normal    the conversation: a fast classification and generic answers.
    PLANNER instruct, billing  reasoning over policy and facts, and task questions with tools.
    """
    chat = build_role_llm(settings, Role.CHAT)
    planner = build_role_llm(settings, Role.PLANNER)
    return {"route": chat, "normal": chat, "instruct": planner, "billing": planner}
