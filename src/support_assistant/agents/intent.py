"""Intent classification for the Director (E4-3): a model answer validated by Pydantic.

The model is asked for JSON and its answer is validated against `Intent`. Anything malformed, out
of range or low-confidence falls back to the deterministic keyword classifier, so a bad model reply
can never crash a turn or invent an intent. A "payment" intent only ever leads to a PROPOSAL: the
evaluator and the human approval step still decide whether anything is paid.
"""

from __future__ import annotations

import json
from typing import Literal, Protocol

import structlog
from pydantic import BaseModel, Field

from support_assistant.agents.llm import LLM
from support_assistant.contracts import Route
from support_assistant.logs import error_fields

log = structlog.get_logger()

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


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)


class RouteClassifier(Protocol):
    async def classify(self, text: str, history: str = "") -> RouteDecision: ...


ROUTE_INSTRUCTIONS = (
    "Route the customer's message for a telecom support assistant. "
    'Reply with JSON only: {"route": "NORMAL"|"INSTRUCT"|"BILLING"|"TROUBLESHOOT", '
    '"confidence": 0.0-1.0}. '
    "NORMAL = greetings, thanks, general chat, simple questions needing no account data. "
    "INSTRUCT = questions that need reasoning over policy or the customer's situation: why was I "
    "charged, disputes, refunds, cancellations, contract or roaming rules, what happens if. "
    "BILLING = plan, usage, bills, balance, paying a bill, suspension. "
    "TROUBLESHOOT = a device or service problem: no signal, slow data, cannot connect, SIM, APN, MMS. "
    "Route the LATEST message; use earlier conversation only to see what it refers to."
)

# Deterministic fallback. Order matters: a stated problem beats a keyword that is also a topic, so
# "my data is not working" is a fault, while "how much data have I used" is an account question.
_FAULT = (
    "not work", "n't work", "stopped working", "no signal", "no service", "no data", "no internet",
    "can't connect", "cannot connect", "won't connect", "can't call", "can't send", "can't access",
    "cannot access", "slow", "dropping", "dropped", "broken", "sim card", "sim is", "apn", "mms",
)  # fmt: skip
_REASON = (
    "why ", "dispute", "refund", "cancel", "policy", "explain", "what happens", "eligible",
    "am i allowed", "how come", "overcharged", "wrong charge",
)  # fmt: skip
_BILLING = (
    "bill", "pay", "owe", "balance", "due", "invoice", "charge", "plan", "usage", "used",
    "allowance", "contract", "line", "number", "suspend", "suspension", "roaming",
)  # fmt: skip


class KeywordRouteClassifier:
    """Deterministic fallback, also used when no model is configured. It reads only the latest
    message: an old keyword must not decide the route."""

    async def classify(self, text: str, history: str = "") -> RouteDecision:
        low = f" {text.lower()} "
        if any(w in low for w in _FAULT):
            return RouteDecision(route="TROUBLESHOOT", confidence=0.6)
        if any(w in low for w in _REASON):
            return RouteDecision(route="INSTRUCT", confidence=0.6)
        if any(w in low for w in _BILLING):
            return RouteDecision(route="BILLING", confidence=0.6)
        return RouteDecision(route="NORMAL", confidence=0.5)


class LlmRouteClassifier:
    """A small model picks the route as validated JSON. Anything malformed, out of range or
    low-confidence falls back to the keyword rules, so a bad model answer never crashes a turn."""

    def __init__(
        self, llm: LLM, fallback: RouteClassifier | None = None, min_confidence: float = 0.6
    ) -> None:
        self._llm = llm
        self._fallback = fallback or KeywordRouteClassifier()
        self._min = min_confidence

    async def classify(self, text: str, history: str = "") -> RouteDecision:
        try:
            raw = await self._llm.respond(ROUTE_INSTRUCTIONS, "", text, history)
            start, end = raw.find("{"), raw.rfind("}")
            decision = RouteDecision.model_validate(json.loads(raw[start : end + 1]))
        except Exception as exc:  # noqa: BLE001 - any failure: fall back
            log.warning(
                "classifier_fallback", classifier="route", reason="error", **error_fields(exc)
            )
            return await self._fallback.classify(text)
        if decision.confidence < self._min:
            log.info(
                "classifier_fallback",
                classifier="route",
                reason="low_confidence",
                confidence=decision.confidence,
            )
            return await self._fallback.classify(text)
        return decision


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
        except Exception as exc:  # noqa: BLE001 - any failure: fall back
            log.warning(
                "classifier_fallback", classifier="intent", reason="error", **error_fields(exc)
            )
            return await self._fallback.classify(text)
        if intent.confidence < self._min:
            log.info(
                "classifier_fallback",
                classifier="intent",
                reason="low_confidence",
                confidence=intent.confidence,
            )
            return await self._fallback.classify(text)
        return intent
