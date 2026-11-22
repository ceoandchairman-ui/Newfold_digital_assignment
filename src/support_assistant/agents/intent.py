"""Intent classification for the Director (E4-3): a model answer validated by Pydantic.

The model is asked for JSON and its answer is validated against `Intent`. Anything malformed, out
of range or low-confidence falls back to the deterministic keyword classifier, so a bad model reply
can never crash a turn or invent an intent. A "payment" intent only ever leads to a PROPOSAL: the
evaluator and the human approval step still decide whether anything is paid.
"""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from support_assistant.agents.llm import LLM

IntentKind = Literal["account", "billing", "payment", "other"]

_ACCOUNT_WORDS = ("plan", "data", "usage", "used", "allowance", "contract", "line", "number")
_BILLING_WORDS = ("bill", "owe", "balance", "due", "invoice", "charge")

CLASSIFY_INSTRUCTIONS = (
    "Classify the customer's message for a telecom support assistant. "
    'Reply with JSON only: {"kind": "account"|"billing"|"payment"|"other", "confidence": 0.0-1.0}. '
    "account = plan, data usage, lines, contract. billing = questions about bills or balance. "
    "payment = the customer wants to pay now. other = anything else."
)


class Intent(BaseModel):
    kind: IntentKind
    confidence: float = Field(ge=0.0, le=1.0)


class IntentClassifier(Protocol):
    async def classify(self, text: str) -> Intent: ...


class KeywordIntentClassifier:
    """Deterministic fallback. Also the classifier used when no model is configured."""

    async def classify(self, text: str) -> Intent:
        low = text.lower()
        if "pay" in low:
            return Intent(kind="payment", confidence=0.6)
        if any(w in low for w in _ACCOUNT_WORDS):
            return Intent(kind="account", confidence=0.6)
        if any(w in low for w in _BILLING_WORDS):
            return Intent(kind="billing", confidence=0.6)
        return Intent(kind="other", confidence=0.5)


class LlmIntentClassifier:
    def __init__(
        self,
        llm: LLM,
        fallback: IntentClassifier | None = None,
        min_confidence: float = 0.6,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or KeywordIntentClassifier()
        self._min = min_confidence

    async def classify(self, text: str) -> Intent:
        try:
            raw = await self._llm.respond(CLASSIFY_INSTRUCTIONS, "", text)
            start, end = raw.find("{"), raw.rfind("}")
            intent = Intent.model_validate(json.loads(raw[start : end + 1]))
        except (ValidationError, ValueError, Exception):  # noqa: BLE001 - any failure: fall back
            return await self._fallback.classify(text)
        if intent.confidence < self._min:
            return await self._fallback.classify(text)
        return intent
