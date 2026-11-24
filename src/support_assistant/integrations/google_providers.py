"""Google Workspace providers: the company support mailbox (Gmail) and company documents (Drive).

Both are about the COMPANY's own mailbox and folder. They never read a customer's own account.

PLACEHOLDERS: the real providers are not written. They need a service account (or OAuth client)
with domain-wide delegation, the Gmail and Drive scopes, and a token flow, none of which exists
yet. Until then every call on a real provider raises ExternalServiceUnavailable, and the stubs are
what run. To finish: implement `GmailApiProvider` and `DriveApiProvider` below using
`settings.google_credentials_json`, `google_support_mailbox` and `google_drive_folder_id`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from support_assistant.config import Settings
from support_assistant.errors import ExternalServiceUnavailable, NotFoundError

MAX_DOCUMENT_CHARS = 20_000


@dataclass(frozen=True)
class MailThread:
    thread_id: str
    subject: str
    snippet: str


@dataclass(frozen=True)
class Document:
    document_id: str
    name: str
    text: str


class EmailProvider(Protocol):
    async def send(self, to: str, subject: str, text: str) -> str:
        """Send from the support mailbox. Returns the message id."""
        ...

    async def threads(self, address: str, limit: int) -> list[MailThread]:
        """Support-mailbox conversations with this address, newest first."""
        ...


class DocumentProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[Document]: ...

    async def read(self, document_id: str) -> Document: ...


# --- stubs: in memory, used in tests and until credentials exist ---------------------------------
class StubEmail:
    def __init__(self, threads: dict[str, list[MailThread]] | None = None) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self._threads = threads or {}

    async def send(self, to: str, subject: str, text: str) -> str:
        self.sent.append((to, subject, text))
        return f"stubmail-{len(self.sent)}"

    async def threads(self, address: str, limit: int) -> list[MailThread]:
        return self._threads.get(address.lower(), [])[:limit]


class StubDocuments:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self._docs = {d.document_id: d for d in docs or []}

    async def search(self, query: str, limit: int) -> list[Document]:
        words = [w for w in query.lower().split() if w]
        hits = [
            d for d in self._docs.values() if any(w in f"{d.name} {d.text}".lower() for w in words)
        ]
        return hits[:limit]

    async def read(self, document_id: str) -> Document:
        if document_id not in self._docs:
            raise NotFoundError(document_id)
        return self._docs[document_id]


# --- real providers: PLACEHOLDERS -----------------------------------------------------------------
class GmailApiProvider:
    """PLACEHOLDER. Not implemented: see the module docstring."""

    def __init__(self, settings: Settings) -> None:
        self._mailbox = settings.google_support_mailbox

    async def send(self, to: str, subject: str, text: str) -> str:
        raise ExternalServiceUnavailable("the Gmail provider is not implemented yet")

    async def threads(self, address: str, limit: int) -> list[MailThread]:
        raise ExternalServiceUnavailable("the Gmail provider is not implemented yet")


class DriveApiProvider:
    """PLACEHOLDER. Not implemented: see the module docstring."""

    def __init__(self, settings: Settings) -> None:
        self._folder = settings.google_drive_folder_id

    async def search(self, query: str, limit: int) -> list[Document]:
        raise ExternalServiceUnavailable("the Drive provider is not implemented yet")

    async def read(self, document_id: str) -> Document:
        raise ExternalServiceUnavailable("the Drive provider is not implemented yet")


def build_email_provider(settings: Settings) -> EmailProvider:
    ready = settings.google_credentials_json and settings.google_support_mailbox
    return StubEmail() if settings.use_stub_gmail or not ready else GmailApiProvider(settings)


def build_document_provider(settings: Settings) -> DocumentProvider:
    ready = settings.google_credentials_json and settings.google_drive_folder_id
    return StubDocuments() if settings.use_stub_drive or not ready else DriveApiProvider(settings)
