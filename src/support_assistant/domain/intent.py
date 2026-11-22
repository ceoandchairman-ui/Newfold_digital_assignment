"""Cheap, deterministic message checks used to skip work that a turn does not need (E4-5)."""

from __future__ import annotations

import re

# Turns that close or acknowledge rather than describe a problem: no retrieval needed.
_ACK = re.compile(
    r"^\s*(ok(ay)?|thanks?( you)?|thank you|ty|great|cool|got it|perfect|yes|no|yep|nope|bye|"
    r"goodbye|that worked|it works( now)?|fixed|done|sure)[\s.!]*$",
    re.IGNORECASE,
)

CLOSING_REPLY = "Glad that helped. Is there anything else I can do for you?"


def needs_retrieval(text: str) -> bool:
    """Lazy retrieval: only search the manual when the message could describe a problem."""
    stripped = text.strip()
    return bool(stripped) and not _ACK.match(stripped)
