"""Chat with the assistant's logic on your own machine.

    uv run --frozen --no-sync python scripts/local_chat.py                 # C1002, owes a bill
    uv run --frozen --no-sync python scripts/local_chat.py C1001
    uv run --frozen --no-sync python scripts/local_chat.py C1002 --ai    # real models

Real: the conversation graph, agent roles, tool permissions, local injection check, payment
approval and payment state machine. Fake: the customer data, the knowledge base (one made-up
article, returned for every search), the payment gateway and the wording (TemplateLLM prints the
facts as they are). It skips the HTTP backend, so there is no identity check or rate limit.
Without --ai nothing leaves your machine and it says nothing about model quality. With --ai the
routing and wording come from the Azure OpenAI deployments at AZURE_OPENAI_ENDPOINT, signed in with
your `az login`; the query rewrite, billing-intent model and reply judge are still not used.
Search quality is not tested in either mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from langgraph.types import Command

from support_assistant.agents.intent import KeywordRouteClassifier, LlmRouteClassifier
from support_assistant.agents.llm import build_role_llm, build_support_llms
from support_assistant.backend.app import ATTACK_REPLY
from support_assistant.config import Settings
from support_assistant.content_safety import local_shield
from support_assistant.models import Role
from test_journeys import Env

ARTICLE = (
    "4.2",
    "Guide > Data > APN",
    "If mobile data fails, reset the APN settings to the carrier default.",
)


def real_llms() -> dict[str, Any]:
    """The same model per job as the deployed agent services (agents/service.py)."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise SystemExit("--ai needs AZURE_OPENAI_ENDPOINT (and `az login`)")
    # The database and Redis addresses are required settings but are never used here.
    settings = Settings(
        database_url="postgresql://unused",
        redis_url="redis://unused",
        azure_openai_endpoint=endpoint,
    )
    support = build_support_llms(settings)
    return {
        "director": support["billing"],
        **{f"director:{mode}": support[mode] for mode in ("normal", "instruct", "billing")},
        "route_classifier": LlmRouteClassifier(support["route"]),
        "troubleshooter": build_role_llm(settings, Role.TROUBLESHOOTER),
    }


async def main(customer: str, ai: bool) -> None:
    env = Env(pytest.MonkeyPatch(), docs=[ARTICLE], llms=real_llms() if ai else None)
    if customer not in env.world.lines:
        raise SystemExit(f"unknown customer {customer}; use one of {', '.join(env.world.lines)}")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    history: list[dict[str, Any]] = []
    model = "real AI models" if ai else "no AI model"
    print(f"Local chat as {customer} (fake data, {model}). Type quit to leave.")

    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            return
        if text.lower() in {"quit", "exit"}:
            return
        if not text:
            continue
        if not sys.stdin.isatty():
            print(text)  # show piped input in the transcript
        if local_shield(text, []).user_attack:
            print(f"assistant> {ATTACK_REPLY}")
            continue

        history.append({"role": "user", "content": text})
        hint = (await KeywordRouteClassifier().classify(text)).route
        state = env.state(customer, text, route=hint)
        state["messages"] = list(history)
        state["search_query"] = text if hint in ("TROUBLESHOOT", "INSTRUCT") else None
        out = await env.graph.ainvoke(state, config)

        while out.get("__interrupt__"):
            pending = out["__interrupt__"][0].value
            question = pending.get("question", pending) if isinstance(pending, dict) else pending
            print(f"assistant> {question}")
            try:
                answer = input("approve? (yes/no)> ").strip().lower()
            except EOFError:
                return
            if not sys.stdin.isatty():
                print(answer)
            out = await env.graph.ainvoke(Command(resume={"approved": answer == "yes"}), config)

        reply = out.get("reply", "")
        print(f"assistant> {reply}")
        print(
            f"           [route: {out.get('intent')} | charges so far: "
            f"{env.gateway.charges_applied}]"
        )
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--ai"]
    ai = "--ai" in sys.argv[1:]
    # With real models, show warnings so a failed or fallback model call is not hidden.
    level = logging.WARNING if ai else logging.CRITICAL
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    asyncio.run(main(args[0] if args else "C1002", ai))
