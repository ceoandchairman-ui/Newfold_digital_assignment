"""Document landing zones (E2-2, E13): team documents -> index; customer files -> quarantine.

The fake database below implements just enough of the customer_uploads SQL to exercise the real
UploadService logic. What it cannot prove is the SQL itself (tests/integration covers that in CI).
"""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from support_assistant import governance, indexer
from support_assistant.backend import app as backend
from support_assistant.blobstore import InMemoryBlobStore
from support_assistant.docsource import (
    MAX_DOC_BYTES,
    LocalDirSource,
    StoreDocumentSource,
)
from support_assistant.embeddings import FakeEmbedder
from support_assistant.embeddings_lc import LcEmbeddings
from support_assistant.errors import ExternalServiceUnavailable, NotFoundError, UploadRejected
from support_assistant.uploads import (
    SCAN_TAG,
    UploadService,
    UploadStatus,
    sanitize_filename,
    validate_upload,
)
from test_indexer_chain import DOC, FakePool

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
MB = 1_000_000


# --- validation: every rule, with a hostile example ---------------------------------------------------
def test_each_allowed_type_passes_with_the_content_type_we_choose_not_the_clients():
    assert validate_upload("shot.png", PNG, MB).content_type == "image/png"
    assert validate_upload("shot.JPG", JPG, MB).content_type == "image/jpeg"
    assert validate_upload("bill.pdf", PDF, MB).content_type == "application/pdf"
    assert validate_upload("notes.txt", b"hello", MB).content_type == "text/plain"


@pytest.mark.parametrize(
    ("name", "data", "why"),
    [
        ("a.png", b"", "empty"),
        ("a.exe", b"MZ" + b"\x00" * 10, "only PNG"),
        (
            "invoice.pdf.exe",
            b"MZ" + b"\x00" * 10,
            "only PNG",
        ),  # double extension: the LAST one counts
        ("noext", b"hello", "only PNG"),
        ("virus.png", b"MZ\x90\x00" + b"\x00" * 10, "does not match"),  # an .exe renamed to .png
        ("fake.pdf", b"just some text", "does not match"),
        ("bin.txt", b"abc\x00def", "binary data"),
        ("latin.txt", "caf\u00e9".encode("latin-1"), "UTF-8"),
    ],
)
def test_hostile_or_mismatched_files_are_rejected_with_a_safe_reason(name, data, why):
    with pytest.raises(UploadRejected) as e:
        validate_upload(name, data, MB)
    assert why in e.value.customer_message and e.value.http_status == 422


def test_an_oversized_file_is_413_and_names_the_limit_not_internals():
    with pytest.raises(UploadRejected) as e:
        validate_upload("big.png", PNG + b"x" * (3 * MB), 2 * MB)
    assert e.value.http_status == 413 and "2 MB" in e.value.customer_message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.png", "passwd.png"),  # path traversal
        ("C:\\Users\\x\\shot.png", "shot.png"),  # Windows path
        (".hidden.png", "hidden.png"),  # no hidden or relative names
        ("a b$c<d>.png", "a_b_c_d_.png"),  # odd characters
        ("", "file"),
        ("x" * 300 + ".png", ("x" * 100)),  # length is capped
    ],
)
def test_filenames_are_reduced_to_a_safe_basename(raw, expected):
    out = sanitize_filename(raw)
    assert out.startswith(expected[:20]) and "/" not in out and "\\" not in out and ".." not in out
    assert len(out) <= 100


# --- a fake customer_uploads table -------------------------------------------------------------------
class Cur:
    def __init__(self, rows=None, rowcount=0):
        self._rows, self.rowcount = rows or [], rowcount

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class UploadsDB:
    def __init__(self, fail_insert: bool = False):
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_insert = fail_insert

    def conn(self):
        db = self

        class Conn:
            async def execute(self, sql, params=None):
                sql = " ".join(sql.split())
                if sql.startswith("SELECT count(*)"):
                    n = sum(1 for r in db.rows.values() if r["customer_id"] == params[0])
                    return Cur([{"n": n}])
                if sql.startswith("INSERT INTO customer_uploads"):
                    if db.fail_insert:
                        raise RuntimeError("db down")
                    uid, cid, sid, blob, name, ctype, size, sha, status = params
                    db.rows[uid] = dict(
                        id=uid,
                        customer_id=cid,
                        blob_name=blob,
                        original_name=name,
                        content_type=ctype,
                        size_bytes=size,
                        status=status,
                        created_at=None,
                    )
                    return Cur()
                if "FROM customer_uploads WHERE id = %s AND customer_id = %s" in sql:
                    r = db.rows.get(params[0])
                    return Cur([r] if r and r["customer_id"] == params[1] else [])
                if sql.startswith("UPDATE customer_uploads SET status"):
                    status, uid, cid = params
                    if (
                        db.rows[uid]["customer_id"] == cid
                        and db.rows[uid]["status"] == "PENDING_SCAN"
                    ):
                        db.rows[uid]["status"] = status
                    return Cur()
                if "FROM customer_uploads WHERE customer_id = %s ORDER BY" in sql:
                    return Cur([r for r in db.rows.values() if r["customer_id"] == params[0]])
                if sql.startswith("SELECT blob_name FROM customer_uploads"):
                    return Cur(
                        [
                            {"blob_name": r["blob_name"]}
                            for r in db.rows.values()
                            if r["customer_id"] == params[0]
                        ]
                    )
                if sql.startswith("DELETE FROM customer_uploads WHERE customer_id"):
                    gone = [k for k, r in db.rows.items() if r["customer_id"] == params[0]]
                    for k in gone:
                        del db.rows[k]
                    return Cur(rowcount=len(gone))
                raise AssertionError(f"unexpected SQL: {sql}")

            @asynccontextmanager
            async def transaction(self):
                yield

        return Conn()

    @asynccontextmanager
    async def connection(self):
        yield self.conn()


def service(db=None, blobs=None, daily=10, max_bytes=MB):
    db, blobs = db or UploadsDB(), blobs or InMemoryBlobStore()
    return UploadService(db, blobs, max_bytes=max_bytes, daily_limit=daily), db, blobs


# --- quarantine and the malware-scan gate ---------------------------------------------------------------
async def test_an_accepted_file_is_stored_under_the_customers_own_prefix_and_starts_quarantined():
    svc, db, blobs = service()
    rec = await svc.accept("C1", "S1", "../../shot.png", PNG)
    assert rec.status is UploadStatus.PENDING_SCAN
    (name,) = blobs.blobs
    assert name == f"C1/{rec.id}/shot.png"  # own prefix, sanitised name, no traversal
    assert blobs.content_types[name] == "image/png"
    assert db.rows[rec.id]["status"] == "PENDING_SCAN"
    assert "C1/" not in json.dumps(rec.public())  # a customer never sees storage paths


async def test_a_file_stays_quarantined_until_the_scanner_has_cleared_it():
    svc, _, blobs = service()
    rec = await svc.accept("C1", "S1", "a.pdf", PDF)
    assert (await svc.status("C1", rec.id)).status is UploadStatus.PENDING_SCAN  # not scanned yet
    (name,) = blobs.blobs
    blobs.set_tags(name, {SCAN_TAG: "Scanning in progress"})  # an unknown result
    assert (await svc.status("C1", rec.id)).status is UploadStatus.PENDING_SCAN
    blobs.set_tags(name, {SCAN_TAG: "No threats found"})
    assert (await svc.status("C1", rec.id)).status is UploadStatus.CLEAN


async def test_a_malicious_file_is_deleted_and_marked_rejected_but_its_record_is_kept():
    svc, db, blobs = service()
    rec = await svc.accept("C1", "S1", "a.pdf", PDF)
    (name,) = blobs.blobs
    blobs.set_tags(name, {SCAN_TAG: "Malicious"})
    assert (await svc.status("C1", rec.id)).status is UploadStatus.REJECTED
    assert (
        blobs.blobs == {} and db.rows[rec.id]["status"] == "REJECTED"
    )  # file gone, audit row kept


async def test_a_clean_decision_is_final_and_not_re_scanned():
    svc, _, blobs = service()
    rec = await svc.accept("C1", "S1", "a.png", PNG)
    (name,) = blobs.blobs
    blobs.set_tags(name, {SCAN_TAG: "No threats found"})
    await svc.status("C1", rec.id)
    blobs.set_tags(name, {SCAN_TAG: "Malicious"})  # a later, different tag changes nothing
    assert (await svc.status("C1", rec.id)).status is UploadStatus.CLEAN


# --- tenant isolation, limits and failure handling --------------------------------------------------------
async def test_another_customers_upload_id_is_simply_not_found():
    svc, _, _ = service()
    rec = await svc.accept("C1", "S1", "a.png", PNG)
    with pytest.raises(NotFoundError):
        await svc.status("C2", rec.id)
    assert await svc.list_for("C2") == [] and len(await svc.list_for("C1")) == 1


async def test_the_daily_limit_is_429_and_per_customer():
    svc, _, _ = service(daily=2)
    await svc.accept("C1", "S", "a.png", PNG)
    await svc.accept("C1", "S", "b.png", PNG)
    with pytest.raises(UploadRejected) as e:
        await svc.accept("C1", "S", "c.png", PNG)
    assert e.value.http_status == 429
    await svc.accept("C2", "S", "a.png", PNG)  # another customer is unaffected


async def test_an_invalid_file_never_reaches_storage():
    svc, db, blobs = service()
    with pytest.raises(UploadRejected):
        await svc.accept("C1", "S", "virus.png", b"MZ" + b"\x00" * 20)
    assert blobs.blobs == {} and db.rows == {}


async def test_if_the_record_cannot_be_saved_the_stored_file_is_removed():
    svc, _, blobs = service(db=UploadsDB(fail_insert=True))
    with pytest.raises(RuntimeError):
        await svc.accept("C1", "S", "a.png", PNG)
    assert blobs.blobs == {}  # no orphaned file without a record


async def test_erasure_removes_files_and_rows_and_governance_reports_the_count():
    svc, db, blobs = service()
    await svc.accept("C1", "S", "a.png", PNG)
    await svc.accept("C1", "S", "b.pdf", PDF)
    await svc.accept("C2", "S", "c.png", PNG)
    out = await governance.erase_customer_memory(_NoopPool(), [], "C1", uploads=svc)
    assert out["uploads"] == 2
    assert [n.split("/")[0] for n in blobs.blobs] == ["C2"] and list(db.rows.values())[0][
        "customer_id"
    ] == "C2"


class _NoopPool:
    @asynccontextmanager
    async def connection(self):
        class C:
            async def execute(self, sql, params=None):
                return Cur(rowcount=0)

            @asynccontextmanager
            async def transaction(self):
                yield

        yield C()


# --- team documents: sources -------------------------------------------------------------------------------
async def _store(files: dict[str, bytes]) -> InMemoryBlobStore:
    s = InMemoryBlobStore()
    for name, data in files.items():
        await s.put(name, data, content_type="text/markdown")
    return s


async def test_a_bad_document_never_stops_the_batch():
    store = await _store(
        {
            "manual.md": b"# Guide\n\nok",
            "notes.txt": b"plain",
            "scan.pdf": PDF,
            "latin.md": "caf\u00e9".encode("latin-1"),
            "huge.md": b"x" * (MAX_DOC_BYTES + 1),
        }
    )
    loaded = await StoreDocumentSource(store).load()
    assert loaded.names == ["manual.md", "notes.txt"]
    assert set(loaded.skipped) == {"scan.pdf", "latin.md", "huge.md"}
    assert "unsupported" in loaded.skipped["scan.pdf"] and "UTF-8" in loaded.skipped["latin.md"]


async def test_a_local_folder_gives_relative_names(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# A\n\nx")
    (tmp_path / "sub" / "b.md").write_text("# B\n\ny")
    (tmp_path / "img.png").write_bytes(PNG)
    loaded = await LocalDirSource(tmp_path).load()
    assert loaded.names == ["a.md", "sub/b.md"] and "img.png" in loaded.skipped
    single = await LocalDirSource(tmp_path / "a.md").load()
    assert single.names == ["a.md"]


# --- team documents: the whole indexing run mirrors the source ----------------------------------------------------
async def _run(pool, files: dict[str, bytes], monkeypatch) -> indexer.IndexReport:
    async def no_register(_c):
        return None

    monkeypatch.setattr(indexer, "register_vector_async", no_register)
    return await indexer.index_source(
        pool, LcEmbeddings(FakeEmbedder(8)), StoreDocumentSource(await _store(files)), "fake"
    )


async def test_deleting_a_document_from_storage_removes_its_chunks_next_run(monkeypatch):
    pool = FakePool()
    await _run(pool, {"a.md": DOC.encode(), "b.md": b"# B\n\nkeep me"}, monkeypatch)
    sources = lambda: {s for s in pool.store["rows"].values()}  # noqa: E731
    assert sources() == {"a.md", "b.md"}
    report = await _run(
        pool, {"b.md": b"# B\n\nkeep me"}, monkeypatch
    )  # a.md was deleted from storage
    assert sources() == {"b.md"} and report.sources_removed > 0


async def test_an_empty_listing_never_wipes_the_knowledge_base(monkeypatch):
    """A storage outage or a wrong container must not look like 'delete everything'."""
    pool = FakePool()
    await _run(pool, {"a.md": DOC.encode()}, monkeypatch)
    before = dict(pool.store["rows"])
    report = await _run(pool, {}, monkeypatch)
    assert pool.store["rows"] == before and report.documents == 0 and report.sources_removed == 0


async def test_the_report_lists_what_was_skipped(monkeypatch):
    report = await _run(FakePool(), {"a.md": DOC.encode(), "x.pdf": PDF}, monkeypatch)
    assert report.documents == 1 and "x.pdf" in report.skipped and report.chunks_added >= 3


# --- in-memory blob store behaves like the real one where it matters ------------------------------------------------
async def test_blobs_are_never_silently_overwritten():
    s = InMemoryBlobStore()
    await s.put("a", b"1", content_type="text/plain")
    with pytest.raises(ExternalServiceUnavailable):
        await s.put("a", b"2", content_type="text/plain")
    assert await s.read("a") == b"1"
    with pytest.raises(NotFoundError):
        await s.read("missing")


# --- the HTTP endpoints --------------------------------------------------------------------------------------------
class KV:
    def __init__(self):
        self.data = {"verified:S1": "C1", "verified:S2": "C2"}

    async def get(self, k):
        return self.data.get(k)


def client(max_bytes: int = MB, configured: bool = True):
    svc, db, blobs = service(max_bytes=max_bytes)
    backend.app.state.redis = KV()
    backend.app.state.s = type("S", (), {"upload_max_bytes": max_bytes})()
    backend.app.state.uploads = svc if configured else None
    transport = httpx.ASGITransport(app=backend.app)
    return httpx.AsyncClient(transport=transport, base_url="http://t"), svc, blobs


def files(name: str, data: bytes):
    return {"file": (name, io.BytesIO(data), "application/octet-stream")}


async def test_a_verified_customer_can_upload_and_sees_a_pending_status():
    c, _, blobs = client()
    r = await c.post("/upload", data={"session_id": "S1"}, files=files("shot.png", PNG))
    assert r.status_code == 200 and r.json()["status"] == "PENDING_SCAN" and len(blobs.blobs) == 1
    assert "blob_name" not in r.json()


async def test_an_unverified_session_cannot_upload():
    c, _, blobs = client()
    r = await c.post("/upload", data={"session_id": "NOPE"}, files=files("a.png", PNG))
    assert r.status_code == 401 and blobs.blobs == {}


async def test_an_oversized_upload_is_cut_off_with_413_before_it_is_fully_read():
    c, _, blobs = client(max_bytes=1000)
    r = await c.post("/upload", data={"session_id": "S1"}, files=files("a.png", PNG + b"x" * 5000))
    assert r.status_code == 413 and blobs.blobs == {}


async def test_a_wrong_type_is_422_with_a_customer_safe_message():
    c, _, _ = client()
    r = await c.post(
        "/upload", data={"session_id": "S1"}, files=files("run.exe", b"MZ" + b"\x00" * 20)
    )
    assert r.status_code == 422 and "PNG, JPEG, PDF" in r.json()["message"]


async def test_uploads_report_unavailable_when_storage_is_not_configured():
    c, _, _ = client(configured=False)
    r = await c.post("/upload", data={"session_id": "S1"}, files=files("a.png", PNG))
    assert r.status_code == 503


async def test_status_and_list_are_private_to_the_customer():
    c, _, _ = client()
    up = (await c.post("/upload", data={"session_id": "S1"}, files=files("a.png", PNG))).json()
    own = await c.get(f"/uploads/{up['id']}", params={"session_id": "S1"})
    other = await c.get(f"/uploads/{up['id']}", params={"session_id": "S2"})
    assert own.status_code == 200 and other.status_code == 404
    assert len((await c.get("/uploads", params={"session_id": "S1"})).json()["uploads"]) == 1
    assert (await c.get("/uploads", params={"session_id": "S2"})).json()["uploads"] == []
