"""Stripe as the real payment gateway behind `process_payment` (replaces the stub gateway).

It implements the same `PaymentGateway` contract as the stub and the simulator, so the payment flow
(row lock, idempotency, read-after-write PAID check, PENDING on timeout, reconcile) does not change.

How Stripe's answers map onto the contract:
- succeeded                          -> APPROVED
- card declined, or needs action     -> DECLINED (we never complete a 3-D Secure flow in chat)
- processing, or a timeout           -> PaymentOutcomeUnknown (left PENDING; reconcile asks Stripe)
- could not connect                  -> ExternalServiceUnavailable (nothing was sent, safe to release)

The idempotency key is sent as Stripe's `Idempotency-Key` header AND stored in the intent's
metadata, because Stripe cannot look an intent up by its idempotency key; `lookup` searches the
metadata instead.

PLACEHOLDERS, NOT VERIFIED against a real Stripe account:
- Which Stripe customer and saved payment method belong to our customer. `payment_method_for` is
  that mapping; the default raises until it is decided and stored.
- The currency (`stripe_currency`).
- The REST shapes below follow Stripe's documented API from memory.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

import httpx

from support_assistant.config import Settings
from support_assistant.errors import ExternalServiceUnavailable, PaymentOutcomeUnknown
from support_assistant.services.payments import GatewayOutcome, PaymentGateway, StubGateway

# (our customer id) -> (stripe customer id, stripe payment method id)
PaymentMethodLookup = Callable[[str], Awaitable[tuple[str, str]]]


async def payment_method_not_configured(customer_id: str) -> tuple[str, str]:
    """PLACEHOLDER: map our customer to their Stripe customer and saved payment method."""
    raise ExternalServiceUnavailable("Stripe payment methods are not mapped to customers yet")


def to_minor_units(amount: Decimal) -> int:
    """Stripe amounts are integers in the smallest unit (cents for USD)."""
    return int((amount * 100).quantize(Decimal("1")))


class StripeGateway:
    def __init__(
        self,
        secret_key: str,
        *,
        base_url: str = "https://api.stripe.com",
        currency: str = "usd",
        payment_method_for: PaymentMethodLookup = payment_method_not_configured,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._currency, self._pm_for = currency, payment_method_for
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {secret_key}"},
            timeout=15.0,
            transport=transport,
        )

    async def charge(
        self, customer_id: str, amount: Decimal, idempotency_key: str
    ) -> GatewayOutcome:
        stripe_customer, payment_method = await self._pm_for(customer_id)
        form = {
            "amount": str(to_minor_units(amount)),
            "currency": self._currency,
            "customer": stripe_customer,
            "payment_method": payment_method,
            "confirm": "true",
            "off_session": "true",  # a saved method, charged without the customer present
            "metadata[idempotency_key]": idempotency_key,
            "metadata[customer_id]": customer_id,
        }
        try:
            resp = await self._http.post(
                "/v1/payment_intents", data=form, headers={"Idempotency-Key": idempotency_key}
            )
        except httpx.TimeoutException as exc:  # sent, reply lost: the charge may have happened
            raise PaymentOutcomeUnknown("stripe timed out") from exc
        except httpx.ConnectError as exc:  # never connected: nothing was sent
            raise ExternalServiceUnavailable("stripe unreachable") from exc
        except httpx.HTTPError as exc:  # failed mid-request: we cannot tell
            raise PaymentOutcomeUnknown("stripe request failed mid-flight") from exc
        return self._outcome(resp)

    @staticmethod
    def _outcome(resp: httpx.Response) -> GatewayOutcome:
        body: dict[str, Any] = resp.json() if resp.content else {}
        if resp.status_code == 402 or body.get("error", {}).get("type") == "card_error":
            return GatewayOutcome.DECLINED
        if resp.status_code >= 500 or resp.status_code == 429:
            raise PaymentOutcomeUnknown(f"stripe returned {resp.status_code}")
        if resp.status_code >= 400:
            raise ExternalServiceUnavailable("stripe rejected the request")
        status = body.get("status")
        if status == "succeeded":
            return GatewayOutcome.APPROVED
        if status == "processing":
            raise PaymentOutcomeUnknown("payment is still processing")
        return GatewayOutcome.DECLINED  # requires_action, requires_payment_method, canceled

    async def lookup(self, idempotency_key: str) -> GatewayOutcome | None:
        try:
            resp = await self._http.get(
                "/v1/payment_intents/search",
                params={"query": f"metadata['idempotency_key']:'{idempotency_key}'"},
            )
        except httpx.HTTPError as exc:
            raise ExternalServiceUnavailable("stripe unreachable") from exc
        if resp.status_code >= 400:
            raise ExternalServiceUnavailable("stripe search failed")
        found = resp.json().get("data") or []
        if not found:
            return None  # Stripe never received it
        return (
            GatewayOutcome.APPROVED
            if found[0].get("status") == "succeeded"
            else GatewayOutcome.DECLINED
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def build_gateway(settings: Settings) -> PaymentGateway:
    """The stub unless Stripe is switched on AND a key exists."""
    key = settings.stripe_secret_key
    if settings.use_stub_gateway or key is None:
        return StubGateway()
    return StripeGateway(
        key.get_secret_value(),
        base_url=settings.stripe_api_base,
        currency=settings.stripe_currency,
    )
