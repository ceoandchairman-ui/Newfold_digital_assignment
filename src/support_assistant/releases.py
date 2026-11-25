"""Shared settings that several workloads must agree on, published as releases.

    tools    MCP servers, their tools, which tools change state, and each agent's allow-list
    search   embedding model, version and dimension, and how documents are chunked

GIT IS THE SOURCE OF TRUTH. These are built from the code and settings in the image. The deploy's
pre-upgrade hook publishes them (`python -m support_assistant.releases publish`), and nothing at
runtime can change them: only the publisher may create catalog items, and nobody may replace one.
The registry NEVER grants a tool: an agent's allow-list always comes from its own code.

A release is named by the hash of its content and is never modified. During a rolling upgrade old
pods find the old release and new pods find the new one. Each workload checks that the release
matching ITS OWN code and settings was published, so a pod with settings nobody deployed (for
example another embedding model) does not become ready.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import structlog

from support_assistant.config import Settings, get_settings
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields
from support_assistant.registry import COLLECTION, Registry, build_registry

log = structlog.get_logger()


@dataclass(frozen=True)
class Release:
    kind: str
    body: dict[str, Any]

    @property
    def version(self) -> str:
        return digest(self.body)


def digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def tools_release() -> Release:
    from support_assistant.agents.roles import ALLOWED_TOOLS
    from support_assistant.mcp_catalog import ALL_SERVERS, WRITE_TOOLS

    return Release(
        "tools",
        {
            "servers": {s.name: {"port": s.port, "tools": sorted(s.tools)} for s in ALL_SERVERS},
            "write_tools": sorted(WRITE_TOOLS),
            "agent_tools": {name: sorted(tools) for name, tools in ALLOWED_TOOLS.items()},
        },
    )


def search_release(settings: Settings) -> Release:
    from support_assistant.indexer import CHUNKER_VERSION, MAX_CHARS

    return Release(
        "search",
        {
            "collection": COLLECTION,
            "embedding_model": settings.embedding_model,
            "embedding_version": settings.embedding_version,
            "embedding_dim": settings.embedding_dim,
            "chunker_version": CHUNKER_VERSION,
            "chunk_max_chars": MAX_CHARS,
        },
    )


async def publish(registry: Registry, releases: list[Release]) -> None:
    for release in releases:
        created = await registry.create_release(release.kind, release.version, release.body)
        # Read it back: an existing item must hold exactly this content.
        stored = await registry.get_release(release.kind, release.version)
        if stored is None or digest(stored) != release.version:
            raise RuntimeError(f"the published {release.kind} release does not match its version")
        log.info("release_published", kind=release.kind, version=release.version, created=created)


class ReleaseGate:
    """Has the release this workload was built with been published?

    A release never changes, so one successful check is remembered for the life of the process: a
    later registry outage does not make a verified pod unready. A pod that has never verified it is
    not ready while the registry is unreachable. With no registry configured the gate is open.
    """

    def __init__(self, registry: Registry | None, release: Release) -> None:
        self._registry, self._release = registry, release
        self._verified = registry is None

    async def check(self) -> bool:
        if self._verified or self._registry is None:
            return True
        kind, version = self._release.kind, self._release.version
        try:
            stored = await self._registry.get_release(kind, version)
        except ExternalServiceUnavailable as exc:
            log.warning("release_check_failed", kind=kind, **error_fields(exc))
            return False
        if stored is None or digest(stored) != version:
            reason = "not_published" if stored is None else "content_mismatch"
            log.error("release_not_verified", kind=kind, version=version, reason=reason)
            return False
        self._verified = True
        log.info("release_verified", kind=kind, version=version)
        return True


async def _publish_main() -> int:
    settings = get_settings()
    registry = build_registry(settings)
    if registry is None:
        print("no COSMOS_ENDPOINT configured: nothing to publish")
        return 0
    try:
        await publish(registry, [tools_release(), search_release(settings)])
    finally:
        await registry.aclose()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["publish"]:
        print("usage: python -m support_assistant.releases publish", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(_publish_main()))
