"""Azure AI Content Safety Prompt Shields (E10-9, E5-6): detect jailbreak / injection attempts.

Checks the customer's message (and optionally retrieved documents) with the managed classifier.
Optional and off by default because it adds a network call per turn. When it is off or
unavailable, the local pattern check in guards.py still runs, so protection never drops to zero.

API: POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01
     body {"userPrompt": str, "documents": [str]} ->
     {"userPromptAnalysis": {"attackDetected": bool}, "documentsAnalysis": [{"attackDetected": bool}]}
Confirm the api-version against the deployed resource on the first integration run.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
import structlog

from support_assistant.guards import looks_like_injection

log = structlog.get_logger()
API_VERSION = "2024-09-01"


@dataclass(frozen=True)
class ShieldResult:
    user_attack: bool
    document_attacks: tuple[bool, ...]
    source: str  # "prompt_shields" or "local"

    @property
    def any_attack(self) -> bool:
        return self.user_attack or any(self.document_attacks)


def local_shield(user_prompt: str, documents: list[str]) -> ShieldResult:
    return ShieldResult(
        looks_like_injection(user_prompt),
        tuple(looks_like_injection(d) for d in documents),
        "local",
    )


class PromptShields:
    def __init__(
        self,
        endpoint: str,
        token_provider: Callable[[], Awaitable[str]],
        timeout_s: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token_provider
        self._http = httpx.AsyncClient(base_url=endpoint, timeout=timeout_s, transport=transport)

    async def check(self, user_prompt: str, documents: list[str] | None = None) -> ShieldResult:
        docs = documents or []
        try:
            resp = await self._http.post(
                "/contentsafety/text:shieldPrompt",
                params={"api-version": API_VERSION},
                headers={"Authorization": f"Bearer {await self._token()}"},
                json={"userPrompt": user_prompt[:10_000], "documents": [d[:10_000] for d in docs]},
            )
            resp.raise_for_status()
            data = resp.json()
            return ShieldResult(
                bool(data["userPromptAnalysis"]["attackDetected"]),
                tuple(bool(d["attackDetected"]) for d in data.get("documentsAnalysis", [])),
                "prompt_shields",
            )
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            # Fail safe, not open: fall back to the local check rather than letting text through.
            log.warning("prompt_shields_unavailable")
            return local_shield(user_prompt, docs)

    async def aclose(self) -> None:
        await self._http.aclose()
