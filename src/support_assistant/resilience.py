"""One retry policy for the whole system: capped exponential backoff with full jitter.

    delay before retry n = random between 0 and min(cap, base * 2**(n-1))

Full jitter spreads out callers that failed together, so they do not all come back at once.

RETRY ONLY WHAT IS SAFE TO REPEAT. A retry is allowed when the operation only reads, when it is
idempotent, or when it certainly never happened (the connection was never made, or Postgres rolled
the transaction back). A timeout after a request was sent is none of these on its own.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, TypeVar, cast

import httpx
import structlog
from psycopg import errors as pg_errors
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_random_exponential

from support_assistant.logs import error_fields

log = structlog.get_logger()

F = TypeVar("F", bound=Callable[..., Any])


def backoff_delay(
    attempt: int, base: float, cap: float, rand: Callable[[], float] = random.random
) -> float:
    """Full-jitter delay in seconds before retry number `attempt` (1 is the first retry)."""
    ceiling: float = min(cap, base * 2.0 ** (max(attempt, 1) - 1))
    return rand() * ceiling


def _retry_after(exc: BaseException | None) -> float | None:
    """The server's own `Retry-After` in seconds, when it sent a number."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return max(0.0, float(exc.response.headers.get("retry-after", "")))
        except ValueError:
            return None
    return None


def retrying(
    operation: str,
    *,
    attempts: int,
    base: float,
    cap: float,
    retry_on: Callable[[BaseException], bool],
) -> Callable[[F], F]:
    """A decorator applying the policy. The last error is re-raised unchanged."""
    jittered = wait_random_exponential(multiplier=base, max=cap)

    def wait(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        asked = _retry_after(exc)
        delay = float(jittered(state))
        return delay if asked is None else max(delay, min(asked, cap))

    def announce(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "retry_scheduled",
            operation=operation,
            attempt=state.attempt_number,
            max_attempts=attempts,
            delay_s=round(state.next_action.sleep, 3) if state.next_action else None,
            **(error_fields(exc) if exc else {}),
        )

    return cast(
        "Callable[[F], F]",
        retry(
            retry=retry_if_exception(retry_on),
            stop=stop_after_attempt(attempts),
            wait=wait,
            before_sleep=announce,
            reraise=True,
        ),
    )


def is_transport_error(exc: BaseException) -> bool:
    """Any connection problem. Safe to retry only for calls that read."""
    return isinstance(exc, httpx.TransportError)


def never_sent(exc: BaseException) -> bool:
    """The request certainly did not reach the server, so repeating it cannot repeat an effect."""
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))


def is_db_conflict(exc: BaseException) -> bool:
    """Postgres aborted and rolled back the whole transaction, so running it again is safe. A lost
    connection is NOT here: the commit may or may not have happened."""
    return isinstance(exc, (pg_errors.SerializationFailure, pg_errors.DeadlockDetected))


db_conflict_retry = retrying(
    "db_transaction", attempts=3, base=0.05, cap=0.5, retry_on=is_db_conflict
)
