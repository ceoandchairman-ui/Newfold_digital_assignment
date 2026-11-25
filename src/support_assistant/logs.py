"""Structured logging rules shared by every service, following OpenTelemetry naming.

Correlation fields, bound once per request and added to every line inside it:

    session.id               OTel semantic convention. A KEYED HASH of the session id, never the id.
    gen_ai.conversation.id   OTel GenAI convention. Here a conversation is one session.
    conversation.turn.id     one customer message (the client's message_id, else a new UUID)
    enduser.pseudo.id        a keyed hash of the customer id
    trace_id / span_id       added by telemetry.py from the active span

A session id works as a login token, so it must never be written to a log. The processors below
enforce that centrally: a call site that passes `session_id=` or `customer_id=` is rewritten, and a
field that could carry what a customer typed is redacted.

Set LOG_HASH_KEY (a secret, the same in every service) so hashes cannot be reversed by guessing.
Without it they are plain SHA-256: fine for long random session ids, but short values such as
customer ids or one-word messages could be recovered by trying them all.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import traceback
import uuid
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import trace

SESSION_ID = "session.id"
CONVERSATION_ID = "gen_ai.conversation.id"
TURN_ID = "conversation.turn.id"
USER_ID = "enduser.pseudo.id"

SESSION_HEADER = "x-session-ref"  # carries the HASH between services, never the session id
TURN_HEADER = "x-turn-id"
_SAFE_HEADER = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# Field names that could carry what a customer typed or who they are.
_NEVER_LOGGED = frozenset(
    {
        "message",
        "text",
        "query",
        "reply",
        "content",
        "prompt",
        "history",
        "phone",
        "email",
        "address",
        "full_name",
        "date_of_birth",
        "token",
        "password",
    }
)

Event = MutableMapping[str, Any]


def pseudonym(value: str) -> str:
    """A stable, one-way stand-in for an identifier: the same input gives the same output."""
    key = os.environ.get("LOG_HASH_KEY", "")
    data = value.encode()
    if key:
        return hmac.new(key.encode(), data, hashlib.sha256).hexdigest()[:24]
    return hashlib.sha256(data).hexdigest()[:24]


def fingerprint(text: str) -> str:
    """The `query.hash` of a message: lets repeats be counted without logging the words. Case and
    spacing are ignored."""
    return pseudonym(" ".join(text.lower().split()))


def new_turn_id() -> str:
    return uuid.uuid4().hex


def error_fields(exc: BaseException) -> dict[str, str]:
    """The OTel `error.type` of an exception. Never its message, which can quote customer data."""
    return {"error.type": type(exc).__name__}


@contextmanager
def conversation(
    session_id: str | None = None,
    customer_id: str | None = None,
    turn_id: str | None = None,
    *,
    session_ref: str | None = None,
) -> Iterator[None]:
    """Bind the correlation fields for everything logged inside, and put them on the current span."""
    fields: dict[str, str] = {}
    ref = session_ref or (pseudonym(session_id) if session_id else None)
    if ref:
        fields[SESSION_ID] = fields[CONVERSATION_ID] = ref
    if customer_id:
        fields[USER_ID] = pseudonym(customer_id)
    if turn_id:
        fields[TURN_ID] = turn_id
    span = trace.get_current_span()
    for name, value in fields.items():
        span.set_attribute(name, value)
    with structlog.contextvars.bound_contextvars(**fields):
        yield


def correlation_headers() -> dict[str, str]:
    """What to send to a service whose request body names no session."""
    bound = structlog.contextvars.get_contextvars()
    pairs = ((SESSION_HEADER, SESSION_ID), (TURN_HEADER, TURN_ID))
    return {header: str(bound[name]) for header, name in pairs if bound.get(name)}


def header_value(headers: Mapping[str, str], name: str) -> str | None:
    """A correlation header, only if it looks like one: a caller cannot write free text into logs."""
    value = headers.get(name, "")
    return value if _SAFE_HEADER.match(value) else None


# --- structlog processors ------------------------------------------------------------------------
def protect_identifiers(_: Any, __: str, event: Event) -> Event:
    if (session := event.pop("session_id", None)) is not None:
        event[SESSION_ID] = event[CONVERSATION_ID] = pseudonym(str(session))
    if (customer := event.pop("customer_id", None)) is not None:
        event[USER_ID] = pseudonym(str(customer))
    for name in _NEVER_LOGGED.intersection(event):
        event[name] = "[redacted]"
    return event


def describe_exception(_: Any, __: str, event: Event) -> Event:
    """`log.exception(...)` keeps the error type and the stack frames, and drops the message."""
    exc_info = event.pop("exc_info", None)
    if not exc_info:
        return event
    if isinstance(exc_info, BaseException):
        exc: BaseException | None = exc_info
    elif isinstance(exc_info, tuple):
        exc = exc_info[1]
    else:
        import sys

        exc = sys.exc_info()[1]
    if exc is not None:
        event.update(error_fields(exc))
        event["exception.stacktrace"] = "".join(traceback.format_tb(exc.__traceback__))[-4000:]
    return event


def export_to_otel(_: Any, __: str, event: Event) -> Event:
    """Send the line through the OpenTelemetry Logs API too, so it reaches the collector over OTLP
    with its fields as attributes. With no log provider configured this does nothing. The SDK takes
    the trace and span ids from the active context."""
    try:
        from opentelemetry._logs import SeverityNumber, get_logger

        level = str(event.get("level", "info"))
        severity = {
            "debug": SeverityNumber.DEBUG,
            "info": SeverityNumber.INFO,
            "warning": SeverityNumber.WARN,
            "error": SeverityNumber.ERROR,
            "critical": SeverityNumber.FATAL,
        }.get(level, SeverityNumber.INFO)
        skip = {"event", "level", "timestamp", "trace_id", "span_id"}
        attributes = {
            k: v if isinstance(v, (str, bool, int, float)) else json.dumps(v, default=str)
            for k, v in event.items()
            if k not in skip and v is not None
        }
        get_logger("support_assistant").emit(
            severity_number=severity,
            severity_text=level.upper(),
            body=str(event.get("event", "")),
            attributes=attributes,
            event_name=str(event.get("event", "")),
        )
    except Exception:  # noqa: BLE001 - logging must never break the work being logged
        return event
    return event
