"""Twilio chat channel: customers talk to the assistant by text message.

This is an INBOUND channel adapter in the fat backend, not an MCP server. Twilio calls the webhook
for each message the customer sends; we answer in the same HTTP response (TwiML), so no outbound
API call is needed for the reply. (Outbound notifications are a separate thing: the `twilio_sms`
MCP server and the notification worker.)

What it adds on top of /chat is only the channel plumbing. Everything else is the same pipeline:
- The request must carry a valid Twilio signature, or it is refused.
- A repeated delivery (Twilio retries) is answered once, using the message SID.
- The sender's phone number identifies the customer, but is NOT proof of identity: the customer
  must still verify with their date of birth, exactly as on the web path, before any data is shown.
- Payment confirmation is a text reply of YES or NO to the question we asked.

Off by default (`twilio_channel_enabled`). PLACEHOLDER: the webhook URL, auth token and sender
number come from configuration and Key Vault. Not verified against a real Twilio account.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from xml.sax.saxutils import escape

import structlog
from fastapi import APIRouter, HTTPException, Request, Response

from support_assistant import repositories as repo
from support_assistant.errors import AuthenticationError, SupportError
from support_assistant.integrations.twilio_client import valid_twilio_signature
from support_assistant.logs import error_fields
from support_assistant.security import channel_session_id, subject_key

log = structlog.get_logger()
router = APIRouter(prefix="/channels/twilio")

MAX_REPLY_CHARS = 1500
YES = {"yes", "y", "confirm", "ok", "okay"}
NO = {"no", "n", "cancel", "stop"}
ASK_DOB = "To continue, please reply with your date of birth as YYYY-MM-DD."
VERIFIED = "Thank you, you are verified. How can I help?"
NOT_VERIFIED = "I could not verify that. Please try again, or call us."
UNKNOWN = "I cannot help with this number here. Please call us."
TROUBLE = "Sorry, something went wrong. Please try again in a moment."


def _phone(raw: str) -> str:
    return raw.removeprefix("whatsapp:").strip()


def twiml(text: str | None) -> Response:
    """An empty response (nothing to say) or one message, with every XML special character escaped."""
    body = (
        "<Response/>"
        if not text
        else f"<Response><Message>{escape(text[:MAX_REPLY_CHARS])}</Message></Response>"
    )
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?>{body}', media_type="application/xml"
    )


def _question(pending: Any) -> str:
    q = pending.get("question") if isinstance(pending, dict) else pending
    return f"{q} Reply YES or NO."


async def converse(phone: str, text: str, message_sid: str) -> str | None:
    """One inbound message in, one reply out (None = say nothing). Never raises to Twilio."""
    from support_assistant.backend import app as backend  # lazy: app imports this module

    redis = backend.app.state.redis
    if message_sid and not await redis.set(f"twilio:sid:{message_sid}", "1", ex=86_400, nx=True):
        log.info("twilio_duplicate_ignored")
        return None  # Twilio delivered this message again: already answered
    phone = _phone(phone)
    async with backend.app.state.pool.connection() as conn:
        customer = await repo.get_customer_by_phone(conn, phone)
    if customer is None:
        log.info("twilio_unknown_number")
        return UNKNOWN  # the same answer for every unknown number
    s = backend.app.state.s
    token = s.twilio_auth_token.get_secret_value() if s.twilio_auth_token else ""
    if not token:
        return TROUBLE  # cannot happen through the webhook, which checks the token first
    # A phone number is not a secret, so the session id is derived with a keyed hash: knowing a number
    # does not let anyone compute the id and use the web API as that customer.
    session_id = channel_session_id(token, "twilio", phone)
    caller = "twilio:" + subject_key(phone)  # rate limits are per phone, not one shared bucket
    try:
        if not await redis.get(f"verified:{session_id}"):
            try:
                dob = date.fromisoformat(text.strip())
            except ValueError:
                log.info("twilio_date_of_birth_requested", session_id=session_id)
                return ASK_DOB
            try:
                await backend.verify_identity(
                    backend.VerifyBody(session_id=session_id, phone=phone, date_of_birth=dob),
                    caller,
                )
            except AuthenticationError:
                log.warning("twilio_verification_failed", session_id=session_id)
                return NOT_VERIFIED
            return VERIFIED
        answer = text.strip().lower()
        pending = await backend._pending_for(str(customer["id"]))
        if pending and answer in YES | NO:
            result = await backend.approve(
                backend.ApproveBody(session_id=session_id, approved=answer in YES)
            )
        else:
            result = await backend.run_chat(
                backend.ChatBody(session_id=session_id, message=text[:2000]), caller
            )
        if result.get("pending_approval"):
            return _question(result["pending_approval"])
        return str(result.get("reply") or "") or None
    except SupportError as exc:
        log.warning("twilio_turn_refused", session_id=session_id, **error_fields(exc))
        return exc.customer_message
    except Exception:  # noqa: BLE001 - the customer gets a polite line, the log gets the detail
        log.exception("twilio_channel_failed")
        return TROUBLE


@router.post("/inbound")
async def inbound(request: Request) -> Response:
    from support_assistant.backend import app as backend

    s = backend.app.state.s
    if not s.twilio_channel_enabled:
        raise HTTPException(404)
    token = s.twilio_auth_token
    if token is None:
        raise HTTPException(503, "the Twilio channel is enabled but not configured")
    form = {k: str(v) for k, v in (await request.form()).items()}
    url = s.twilio_webhook_url or str(request.url)
    signature = request.headers.get("X-Twilio-Signature")
    if not valid_twilio_signature(url, form, signature, token.get_secret_value()):
        raise HTTPException(403, "invalid signature")
    reply = await converse(form.get("From", ""), form.get("Body", ""), form.get("MessageSid", ""))
    return twiml(reply)
