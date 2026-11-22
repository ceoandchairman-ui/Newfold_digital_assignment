"""Typed domain errors with customer-safe messages (E3-9).

Stack traces, table names and internal service names must never reach the customer.
"""

from __future__ import annotations

GENERIC_MESSAGE = "Sorry, something went wrong on our side. Please try again in a moment."


class SupportError(Exception):
    """Base class. `customer_message` is the only text a customer may see."""

    code = "support_error"
    http_status = 500
    customer_message = GENERIC_MESSAGE


class ValidationError(SupportError):
    code = "validation_error"
    http_status = 422
    customer_message = "I couldn't use that information. Could you check it and try again?"


class AuthenticationError(SupportError):
    code = "authentication_error"
    http_status = 401
    customer_message = "I need to verify your identity before I can help with that."


class AuthorizationError(SupportError):
    code = "authorization_error"
    http_status = 403
    customer_message = "I'm not able to access that information."


class NotFoundError(SupportError):
    code = "not_found"
    http_status = 404
    customer_message = "I couldn't find that record."


class ExternalServiceUnavailable(SupportError):
    code = "external_service_unavailable"
    http_status = 503
    customer_message = "That service is temporarily unavailable. Nothing has been changed. Please try again shortly."


class PaymentDeclined(SupportError):
    code = "payment_declined"
    http_status = 402
    customer_message = "Your payment was declined and you have not been charged."


class PaymentOutcomeUnknown(SupportError):
    """The gateway did not answer in time. The charge may or may not have happened (E3-5)."""

    code = "payment_outcome_unknown"
    http_status = 202
    customer_message = (
        "I couldn't confirm your payment yet. Please don't pay again; "
        "I'll check it and let you know the result."
    )


class DuplicateOperation(SupportError):
    code = "duplicate_operation"
    http_status = 409
    customer_message = "That request was already processed."


class ConflictError(SupportError):
    code = "conflict"
    http_status = 409
    customer_message = "That action isn't possible in the current state."


class InvariantViolation(SupportError):
    """A model or service broke a rule the system guarantees (E4-8). Fail closed and escalate."""

    code = "invariant_violation"
    http_status = 500
    customer_message = "I can't safely continue with this request."


class UnknownBillStatus(SupportError):
    """A bill carries a status we do not recognise. Fail closed and escalate (E1-5, E3-7)."""

    code = "unknown_bill_status"
    http_status = 409
    customer_message = "I can't safely continue with this bill."


def to_customer_message(exc: BaseException) -> str:
    """Translate any exception into text that is safe to show a customer."""
    if isinstance(exc, SupportError):
        return exc.customer_message
    return GENERIC_MESSAGE
