"""Customer file uploads (E13): untrusted input, so every step assumes the file is hostile.

Pipeline:  validate -> store in quarantine -> record in Postgres -> wait for the malware scan.
  1. Validate BEFORE touching storage: size, allowed type, and the file's real bytes ("magic
     numbers") must match its extension. The client's Content-Type header is never trusted.
  2. The filename is reduced to a safe basename. The blob path is `{customer}/{upload id}/{name}`,
     so a customer's files are always under their own prefix and a name cannot escape it.
  3. The file is recorded as PENDING_SCAN. Defender for Storage scans it on upload and writes the
     result as a blob index tag. `status()` reads that tag: "No threats found" -> CLEAN, malicious ->
     REJECTED and the blob is deleted. Nothing may use a file that is not CLEAN.
  4. Limits: per-file size and uploads per customer per 24 h.

What is deliberately NOT done: the file is never given to a model or parsed. Text inside an uploaded
file could contain instructions (prompt injection), and image/PDF understanding is a separate
feature. A CLEAN file is available to a human agent on escalation.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import structlog

from support_assistant.blobstore import BlobStore
from support_assistant.db import Pool
from support_assistant.errors import NotFoundError, UploadRejected

log = structlog.get_logger()

SCAN_TAG = "Malware Scanning scan result"  # written by Defender for Storage
SCAN_CLEAN = "No threats found"
SCAN_MALICIOUS = "Malicious"


class UploadStatus(StrEnum):
    PENDING_SCAN = "PENDING_SCAN"
    CLEAN = "CLEAN"
    REJECTED = "REJECTED"


# extension -> (content type we store, file signatures it must start with; () = plain text check)
ALLOWED_TYPES: dict[str, tuple[str, tuple[bytes, ...]]] = {
    ".png": ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ".jpg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".jpeg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".pdf": ("application/pdf", (b"%PDF-",)),
    ".txt": ("text/plain", ()),
}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class ValidatedUpload:
    safe_name: str
    content_type: str
    size: int
    sha256: str


@dataclass(frozen=True)
class UploadRecord:
    id: str
    status: UploadStatus
    original_name: str
    content_type: str
    size_bytes: int
    created_at: datetime | None = None

    def public(self) -> dict[str, Any]:
        """What a customer may see. Never the blob path or storage details."""
        return {
            "id": self.id,
            "status": self.status.value,
            "name": self.original_name,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }


def sanitize_filename(name: str, max_len: int = 100) -> str:
    base = re.split(r"[\\/]", name)[-1]  # drop any directory part, either separator
    base = _UNSAFE.sub("_", base).lstrip(".")  # no odd characters, no hidden/relative names
    base = re.sub(r"_{2,}", "_", base)[:max_len]
    return base or "file"


def validate_upload(filename: str, data: bytes, max_bytes: int) -> ValidatedUpload:
    if not data:
        raise UploadRejected("the file is empty")
    if len(data) > max_bytes:
        raise UploadRejected(f"the file is larger than {max_bytes // 1_000_000} MB", 413)
    safe = sanitize_filename(filename)
    ext = "." + safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if ext not in ALLOWED_TYPES:
        raise UploadRejected("only PNG, JPEG, PDF and plain text files are accepted")
    content_type, signatures = ALLOWED_TYPES[ext]
    if signatures:
        if not any(data.startswith(sig) for sig in signatures):
            raise UploadRejected("the file's content does not match its type")
    else:  # plain text: must be valid UTF-8 without NUL bytes (those mark binary data)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadRejected("the text file is not valid UTF-8") from None
        if b"\x00" in data:
            raise UploadRejected("the text file contains binary data")
    return ValidatedUpload(safe, content_type, len(data), hashlib.sha256(data).hexdigest())


class UploadService:
    def __init__(self, pool: Pool, blobs: BlobStore, *, max_bytes: int, daily_limit: int) -> None:
        self._pool, self._blobs = pool, blobs
        self._max_bytes, self._daily_limit = max_bytes, daily_limit

    async def accept(
        self, customer_id: str, session_id: str, filename: str, data: bytes
    ) -> UploadRecord:
        v = validate_upload(filename, data, self._max_bytes)  # reject early, before storage
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM customer_uploads"
                    " WHERE customer_id = %s AND created_at > now() - interval '24 hours'",
                    (customer_id,),
                )
            ).fetchone()
        if row and int(row["n"]) >= self._daily_limit:
            raise UploadRejected("you have reached today's upload limit", 429)

        upload_id = str(uuid.uuid4())
        blob_name = f"{customer_id}/{upload_id}/{v.safe_name}"
        await self._blobs.put(
            blob_name, data, content_type=v.content_type, metadata={"upload_id": upload_id}
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO customer_uploads (id, customer_id, session_id, blob_name,"
                        " original_name, content_type, size_bytes, sha256, status)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            upload_id,
                            customer_id,
                            session_id,
                            blob_name,
                            v.safe_name,
                            v.content_type,
                            v.size,
                            v.sha256,
                            UploadStatus.PENDING_SCAN.value,
                        ),
                    )
        except Exception:
            await self._blobs.delete(blob_name)  # no orphaned file without a record
            raise
        return UploadRecord(
            upload_id, UploadStatus.PENDING_SCAN, v.safe_name, v.content_type, v.size
        )

    async def _row(self, customer_id: str, upload_id: str) -> dict[str, Any]:
        """Always filtered by customer: another customer's upload id is simply 'not found'."""
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT id, blob_name, original_name, content_type, size_bytes, status, created_at"
                    " FROM customer_uploads WHERE id = %s AND customer_id = %s",
                    (upload_id, customer_id),
                )
            ).fetchone()
        if row is None:
            raise NotFoundError(upload_id)
        return dict(row)

    @staticmethod
    def _record(row: dict[str, Any], status: UploadStatus) -> UploadRecord:
        return UploadRecord(
            str(row["id"]),
            status,
            row["original_name"],
            row["content_type"],
            int(row["size_bytes"]),
            row.get("created_at"),
        )

    async def status(self, customer_id: str, upload_id: str) -> UploadRecord:
        """Current status, refreshed from the malware scan result while still pending."""
        row = await self._row(customer_id, upload_id)
        current = UploadStatus(row["status"])
        if current is not UploadStatus.PENDING_SCAN:
            return self._record(row, current)

        result = (await self._blobs.tags(row["blob_name"])).get(SCAN_TAG)
        if result == SCAN_CLEAN:
            new = UploadStatus.CLEAN
        elif result == SCAN_MALICIOUS:
            new = UploadStatus.REJECTED
        else:
            return self._record(
                row, current
            )  # not scanned yet (or an unknown result): stay quarantined

        if new is UploadStatus.REJECTED:
            log.warning("upload_rejected_by_scan", upload_id=upload_id)
            await self._blobs.delete(row["blob_name"])  # a threat is not kept, only the audit row
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE customer_uploads SET status = %s, scanned_at = now()"
                    " WHERE id = %s AND customer_id = %s AND status = 'PENDING_SCAN'",
                    (new.value, upload_id, customer_id),
                )
        return self._record(row, new)

    async def list_for(self, customer_id: str) -> list[UploadRecord]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, original_name, content_type, size_bytes, status, created_at"
                    " FROM customer_uploads WHERE customer_id = %s ORDER BY created_at DESC LIMIT 50",
                    (customer_id,),
                )
            ).fetchall()
        return [self._record(dict(r), UploadStatus(r["status"])) for r in rows]

    async def erase_customer(self, customer_id: str) -> int:
        """Erasure on request (E5-8): delete the customer's files from storage, then their rows."""
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT blob_name FROM customer_uploads WHERE customer_id = %s", (customer_id,)
                )
            ).fetchall()
        for r in rows:
            try:
                await self._blobs.delete(r["blob_name"])
            except NotFoundError:
                log.info("upload_blob_already_gone")  # expired or removed by the scanner
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM customer_uploads WHERE customer_id = %s", (customer_id,)
                )
        return len(rows)
