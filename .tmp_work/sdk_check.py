"""The REAL azure-cosmos async SDK against a local stand-in: does CosmosRegistry call it correctly
without `async with`? This checks the client side only, not Cosmos DB's behaviour."""

import asyncio
import datetime
import json
import ssl
import sys
import time
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AccessToken
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from support_assistant.registry import CosmosRegistry, DocumentRecord

PORT = 8765
BASE = f"https://127.0.0.1:{PORT}/"
store: dict[str, dict] = {}
seen: list[str] = []


class Credential:
    async def get_token(self, *scopes, **kwargs):
        return AccessToken("fake-token", int(time.time()) + 3600)

    async def close(self):
        return None


def reply(body, status=200):
    return web.json_response(body, status=status)


async def handle(request: web.Request) -> web.Response:
    path = request.path.strip("/")
    seen.append(f"{request.method} /{path}")
    if not path:
        region = [{"name": "local", "databaseAccountEndpoint": BASE}]
        return reply(
            {
                "id": "local",
                "_rid": "local",
                "writableLocations": region,
                "readableLocations": region,
                "userConsistencyPolicy": {"defaultConsistencyLevel": "Strong"},
                "enableMultipleWriteLocations": False,
                "queryEngineConfiguration": "{}",
            }
        )
    parts = path.split("/")
    if len(parts) == 4:  # a container's metadata
        key = "/collection" if parts[3] == "documents" else "/kind"
        return reply(
            {
                "id": parts[3],
                "_rid": "AAAAAA==",
                "_self": path,
                "partitionKey": {"paths": [key], "kind": "Hash", "version": 2},
            }
        )
    if request.method == "POST":
        body = json.loads(await request.text())
        key = f"{path}/{body['id']}"
        upsert = request.headers.get("x-ms-documentdb-is-upsert", "").lower() == "true"
        if key in store and not upsert:
            return reply({"code": "Conflict"}, 409)
        store[key] = body
        return reply(body, 201)
    if path in store:
        if request.method == "DELETE":
            del store[path]
            return web.Response(status=204)
        return reply(store[path])
    return reply({"code": "NotFound"}, 404)


def tls() -> ssl.SSLContext:
    """A throwaway certificate for 127.0.0.1, written inside the temporary folder."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(1).not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1)).sign(key, hashes.SHA256())
    )
    pem = Path(".tmp_work/check.pem")
    pem.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption())
        + cert.public_bytes(serialization.Encoding.PEM)
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(pem)
    return context


async def main() -> int:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT, ssl_context=tls()).start()
    from azure.cosmos.aio import CosmosClient

    credential = Credential()
    client = CosmosClient(BASE, credential=credential, connection_verify=False)
    registry = CosmosRegistry(BASE, "registry", credential=credential, client=client)
    record = DocumentRecord(
        source="a/b.md",
        document_hash="a" * 64,
        indexing_fingerprint="f",
        chunk_hashes=["h"],
        size_bytes=1,
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    try:
        async with asyncio.timeout(60):
            results = {
                "missing document is None": await registry.get_document("a/b.md") is None,
            }
            await registry.put_document(record)
            results["written then read back"] = await registry.get_document("a/b.md") == record
            await registry.delete_document("a/b.md")
            await registry.delete_document("a/b.md")
            results["deleted twice without error"] = await registry.get_document("a/b.md") is None
            first = await registry.create_release("tools", "v1", {"a": 1})
            second = await registry.create_release("tools", "v1", {"a": 2})
            results["release is create-only"] = (first, second) == (True, False)
            results["release read back"] = await registry.get_release("tools", "v1") == {"a": 1}
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", type(exc).__name__, "<-", type(exc.__cause__).__name__, exc.__cause__)
        print("last requests:", seen[-6:])
        return 1
    finally:
        await registry.aclose()
        await runner.cleanup()
    for name, ok in results.items():
        print("ok  " if ok else "BAD ", name)
    print("requests sent:", len(seen))
    return 0 if all(results.values()) else 1


sys.exit(asyncio.run(main()))
