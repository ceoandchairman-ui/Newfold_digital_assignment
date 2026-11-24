"""Twilio: outbound SMS, and the signature check for the inbound chat channel.

The REST calls follow Twilio's documented Messages API (form-encoded POST, HTTP basic auth with the
account SID and auth token). NOT VERIFIED against a real Twilio account: nothing is configured yet.
Until `use_stub_twilio` is turned off and credentials exist, the stub is used everywhere.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import structlog

from support_assistant.config import Settings
from support_assistant.errors import ExternalServiceUnavailable

log = structlog.get_logger()

API = "https://api.twilio.com/2010-04-01"


@dataclass(frozen=True)
class SentSms:
    sid: str
    status: str
    to: str


class SmsProvider(Protocol):
    async def send(self, to: str, text: str) -> SentSms: ...

    async def status(self, sid: str) -> SentSms | None:
        """What the provider recorded for a message, or None if it has no such message."""
        ...


class StubSms:
    """Records messages in memory. Used until Twilio credentials exist."""

    def __init__(self) -> None:
        self.sent: list[SentSms] = []

    async def send(self, to: str, text: str) -> SentSms:
        msg = SentSms(f"SMstub{len(self.sent) + 1:04d}", "queued", to)
        log.info("sms_stub", to="***" + to[-4:], chars=len(text))
        self.sent.append(msg)
        return msg

    async def status(self, sid: str) -> SentSms | None:
        return next((m for m in self.sent if m.sid == sid), None)


class TwilioSms:
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._sid, self._from = account_sid, from_number
        self._http = httpx.AsyncClient(
            auth=(account_sid, auth_token), timeout=10.0, transport=transport
        )

    async def send(self, to: str, text: str) -> SentSms:
        try:
            resp = await self._http.post(
                f"{API}/Accounts/{self._sid}/Messages.json",
                data={"To": to, "From": self._from, "Body": text},
            )
        except httpx.HTTPError as exc:
            raise ExternalServiceUnavailable("sms provider unavailable") from exc
        if resp.status_code >= 400:
            raise ExternalServiceUnavailable("sms provider rejected the message")
        body: dict[str, Any] = resp.json()
        return SentSms(str(body["sid"]), str(body.get("status", "queued")), to)

    async def status(self, sid: str) -> SentSms | None:
        try:
            resp = await self._http.get(f"{API}/Accounts/{self._sid}/Messages/{sid}.json")
        except httpx.HTTPError as exc:
            raise ExternalServiceUnavailable("sms provider unavailable") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise ExternalServiceUnavailable("sms provider error")
        body: dict[str, Any] = resp.json()
        return SentSms(str(body["sid"]), str(body.get("status", "")), str(body.get("to", "")))

    async def aclose(self) -> None:
        await self._http.aclose()


def build_sms_provider(settings: Settings) -> SmsProvider:
    """The stub unless Twilio is switched on AND fully configured."""
    token = settings.twilio_auth_token
    if (
        settings.use_stub_twilio
        or not settings.twilio_account_sid
        or token is None
        or not settings.twilio_sms_from
    ):
        return StubSms()
    return TwilioSms(
        settings.twilio_account_sid, token.get_secret_value(), settings.twilio_sms_from
    )


def twilio_signature(url: str, params: Mapping[str, str], auth_token: str) -> str:
    """Twilio's request signature: HMAC-SHA1 over the URL plus every POST parameter, name then
    value, sorted by name, base64-encoded. The URL is the exact public URL Twilio was given."""
    data = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def valid_twilio_signature(
    url: str, params: Mapping[str, str], header: str | None, auth_token: str
) -> bool:
    """Constant-time comparison. A missing header or an empty token never validates."""
    if not header or not auth_token:
        return False
    return hmac.compare_digest(twilio_signature(url, params, auth_token), header)
