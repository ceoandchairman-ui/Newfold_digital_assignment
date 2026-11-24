"""Blob storage behind one small interface (E2-2, E13).

Two containers use it: `knowledge-base` (team documents the indexer reads) and `customer-uploads`
(files customers send). Production uses Azure Blob Storage over a private endpoint with Entra
auth (no account keys). Tests use the in-memory implementation.

Failures are translated: a missing blob is NotFoundError, anything else from the service is
ExternalServiceUnavailable, so callers never see SDK exceptions or storage URLs.
"""

from __future__ import annotations

from typing import Any, Protocol

from support_assistant.errors import ExternalServiceUnavailable, NotFoundError


class BlobStore(Protocol):
    async def put(
        self, name: str, data: bytes, *, content_type: str, metadata: dict[str, str] | None = None
    ) -> None: ...

    async def read(self, name: str) -> bytes: ...

    async def list(self, prefix: str = "") -> list[str]: ...

    async def etags(self, prefix: str = "") -> dict[str, str]:
        """name -> version marker that changes whenever the blob's content changes."""
        ...

    async def delete(self, name: str) -> None: ...

    async def tags(self, name: str) -> dict[str, str]: ...


class InMemoryBlobStore:
    """For tests. `set_tags` plays the part of the malware scanner writing its result."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self._tags: dict[str, dict[str, str]] = {}

    async def put(
        self, name: str, data: bytes, *, content_type: str, metadata: dict[str, str] | None = None
    ) -> None:
        if name in self.blobs:
            raise ExternalServiceUnavailable("blob already exists")  # uploads never overwrite
        self.blobs[name], self.content_types[name] = data, content_type
        self.metadata[name] = dict(metadata or {})

    async def read(self, name: str) -> bytes:
        if name not in self.blobs:
            raise NotFoundError(name)
        return self.blobs[name]

    async def list(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.blobs if n.startswith(prefix))

    async def etags(self, prefix: str = "") -> dict[str, str]:
        import hashlib

        return {
            n: hashlib.sha1(d).hexdigest()  # noqa: S324 - a change marker, not security
            for n, d in sorted(self.blobs.items())
            if n.startswith(prefix)
        }

    async def delete(self, name: str) -> None:
        if name not in self.blobs:
            raise NotFoundError(name)
        del self.blobs[name]
        self._tags.pop(name, None)

    async def tags(self, name: str) -> dict[str, str]:
        if name not in self.blobs:
            raise NotFoundError(name)
        return dict(self._tags.get(name, {}))

    def set_tags(self, name: str, tags: dict[str, str]) -> None:
        self._tags[name] = dict(tags)


class AzureBlobStore:
    """One container in a private storage account, authenticated as the pod's workload identity."""

    def __init__(self, account: str, container: str, credential: Any | None = None) -> None:
        from azure.identity.aio import DefaultAzureCredential
        from azure.storage.blob.aio import BlobServiceClient

        self._credential = credential or DefaultAzureCredential()
        self._service = BlobServiceClient(
            f"https://{account}.blob.core.windows.net", credential=self._credential
        )
        self._container = self._service.get_container_client(container)

    @staticmethod
    def _translate(exc: Exception, name: str) -> Exception:
        from azure.core.exceptions import ResourceNotFoundError

        if isinstance(exc, ResourceNotFoundError):
            return NotFoundError(name)
        return ExternalServiceUnavailable("blob storage unavailable")

    async def put(
        self, name: str, data: bytes, *, content_type: str, metadata: dict[str, str] | None = None
    ) -> None:
        from azure.storage.blob import ContentSettings

        try:
            await self._container.upload_blob(
                name,
                data,
                overwrite=False,  # a customer file is never silently replaced
                content_settings=ContentSettings(content_type=content_type),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, name) from exc

    async def read(self, name: str) -> bytes:
        try:
            stream = await self._container.download_blob(name)
            return bytes(await stream.readall())
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, name) from exc

    async def list(self, prefix: str = "") -> list[str]:
        try:
            return sorted(
                [b.name async for b in self._container.list_blobs(name_starts_with=prefix)]
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, prefix) from exc

    async def etags(self, prefix: str = "") -> dict[str, str]:
        try:
            return {
                b.name: str(b.etag)
                async for b in self._container.list_blobs(name_starts_with=prefix)
            }
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, prefix) from exc

    async def delete(self, name: str) -> None:
        try:
            await self._container.delete_blob(name)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, name) from exc

    async def tags(self, name: str) -> dict[str, str]:
        try:
            result = await self._container.get_blob_client(name).get_blob_tags()
            return dict(result)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, name) from exc

    async def aclose(self) -> None:
        await self._service.close()
        await self._credential.close()
