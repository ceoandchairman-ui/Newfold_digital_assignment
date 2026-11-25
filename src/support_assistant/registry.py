"""The central metadata registry, in Azure Cosmos DB (NoSQL API).

It holds METADATA only. Chunk text and vectors stay in Postgres.

    container `documents`  partition /collection   one record per knowledge document
    container `catalog`    partition /kind         published releases of shared settings

Cosmos and Postgres cannot share a transaction, so a registry record is a CLAIM, never proof: the
indexer checks Postgres before it trusts one (indexer.py).

Access is Entra only (account keys are disabled) with a data-plane role per workload, scoped to one
container (infra/terraform/cosmos.tf). Databases and containers are created by Terraform: these
roles cannot create them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable
from typing import Any, Final, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from support_assistant.config import Settings
from support_assistant.errors import ExternalServiceUnavailable

DOCUMENTS = "documents"
CATALOG = "catalog"
COLLECTION: Final = "company_knowledge"  # the one knowledge collection today

T = TypeVar("T")


class DocumentRecord(BaseModel):
    schema_version: int = 1
    collection: Literal["company_knowledge"] = COLLECTION
    source: str = Field(min_length=1)  # the blob name
    document_hash: str = Field(min_length=64, max_length=64)  # SHA-256 of the file's bytes
    indexing_fingerprint: str  # the search release it was indexed with (releases.py)
    chunk_hashes: list[str]  # sorted and distinct: what Postgres must hold for this source
    size_bytes: int = Field(ge=0)
    indexed_at: str  # ISO 8601, UTC

    @property
    def id(self) -> str:
        return document_id(self.source)


def document_id(source: str) -> str:
    """Cosmos ids may not contain `/ \\ ? #`, which blob names can, so the name is hashed."""
    return hashlib.sha256(source.encode()).hexdigest()


class Registry(Protocol):
    """Every method raises ExternalServiceUnavailable when the store cannot answer."""

    async def get_document(self, source: str) -> DocumentRecord | None: ...

    async def put_document(self, record: DocumentRecord) -> None: ...

    async def delete_document(self, source: str) -> None:
        """Deleting a record that is not there is not an error."""
        ...

    async def document_sources(self) -> list[str]: ...

    async def get_release(self, kind: str, version: str) -> dict[str, Any] | None: ...

    async def create_release(self, kind: str, version: str, body: dict[str, Any]) -> bool:
        """Create only. False if it already exists: a published release is never changed."""
        ...

    async def aclose(self) -> None: ...


class InMemoryRegistry:
    """For tests."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.releases: dict[tuple[str, str], dict[str, Any]] = {}
        self.down = False
        self.writes = 0

    def _check(self) -> None:
        if self.down:
            raise ExternalServiceUnavailable("registry unavailable")

    async def get_document(self, source: str) -> DocumentRecord | None:
        self._check()
        found = self.documents.get(source)
        return DocumentRecord.model_validate(found) if found else None

    async def put_document(self, record: DocumentRecord) -> None:
        self._check()
        self.writes += 1
        self.documents[record.source] = record.model_dump()

    async def delete_document(self, source: str) -> None:
        self._check()
        self.documents.pop(source, None)

    async def document_sources(self) -> list[str]:
        self._check()
        return sorted(self.documents)

    async def get_release(self, kind: str, version: str) -> dict[str, Any] | None:
        self._check()
        return self.releases.get((kind, version))

    async def create_release(self, kind: str, version: str, body: dict[str, Any]) -> bool:
        self._check()
        if (kind, version) in self.releases:
            return False
        self.releases[(kind, version)] = body
        return True

    async def aclose(self) -> None:
        return None


class CosmosRegistry:
    """The SDK retries throttled (429) and transient requests itself, so no retry is added here."""

    def __init__(
        self,
        endpoint: str,
        database: str,
        credential: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from azure.cosmos.aio import CosmosClient
            from azure.identity.aio import DefaultAzureCredential

            credential = credential or DefaultAzureCredential()
            client = CosmosClient(endpoint, credential=credential)
        self._client, self._credential = client, credential
        db = client.get_database_client(database)
        self._documents = db.get_container_client(DOCUMENTS)
        self._catalog = db.get_container_client(CATALOG)

    @staticmethod
    async def _call(operation: Awaitable[T]) -> T:
        from azure.cosmos.exceptions import (
            CosmosResourceExistsError,
            CosmosResourceNotFoundError,
        )

        try:
            return await operation
        except (CosmosResourceExistsError, CosmosResourceNotFoundError):
            raise  # answers, not outages
        except Exception as exc:  # noqa: BLE001 - SDK, network and sign-in failures alike
            raise ExternalServiceUnavailable("registry unavailable") from exc

    async def _read(self, container: Any, item: str, partition: str) -> dict[str, Any] | None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            found = await self._call(container.read_item(item=item, partition_key=partition))
        except CosmosResourceNotFoundError:
            return None
        return dict(found)

    async def get_document(self, source: str) -> DocumentRecord | None:
        found = await self._read(self._documents, document_id(source), COLLECTION)
        if found is None:
            return None
        try:
            return DocumentRecord.model_validate(found)
        except ValidationError:
            return None  # an unreadable record is treated as no record: the document is re-indexed

    async def put_document(self, record: DocumentRecord) -> None:
        body = {"id": record.id, **record.model_dump()}
        await self._call(self._documents.upsert_item(body=body))

    async def delete_document(self, source: str) -> None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            await self._call(
                self._documents.delete_item(item=document_id(source), partition_key=COLLECTION)
            )
        except CosmosResourceNotFoundError:
            return

    async def document_sources(self) -> list[str]:
        async def collect() -> list[str]:
            items = self._documents.query_items(
                query="SELECT VALUE c.source FROM c", partition_key=COLLECTION
            )
            return [str(x) async for x in items]

        return sorted(await self._call(collect()))

    async def get_release(self, kind: str, version: str) -> dict[str, Any] | None:
        found = await self._read(self._catalog, f"{kind}-{version}", kind)
        return dict(found["body"]) if found and isinstance(found.get("body"), dict) else None

    async def create_release(self, kind: str, version: str, body: dict[str, Any]) -> bool:
        from azure.cosmos.exceptions import CosmosResourceExistsError

        item = {"id": f"{kind}-{version}", "kind": kind, "version": version, "body": body}
        try:
            await self._call(self._catalog.create_item(body=item))
        except CosmosResourceExistsError:
            return False
        return True

    async def aclose(self) -> None:
        await self._client.close()
        if self._credential is not None:
            await self._credential.close()


def build_registry(settings: Settings) -> Registry | None:
    """None when no endpoint is configured (tests, local use). With REGISTRY_REQUIRED, as in
    production, a missing endpoint stops the service instead of silently skipping the checks."""
    if settings.cosmos_endpoint:
        return CosmosRegistry(settings.cosmos_endpoint, settings.cosmos_database)
    if settings.registry_required:
        raise RuntimeError("REGISTRY_REQUIRED is set but COSMOS_ENDPOINT is not")
    return None
