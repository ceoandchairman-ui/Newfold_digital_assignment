"""Break one safeguard at a time, confirm its tests fail, then restore the file byte for byte."""

import subprocess
import sys
from pathlib import Path

CASES = [
    (
        "trust a record without checking Postgres",
        "src/support_assistant/indexer.py",
        "            and await _chunk_hashes_in_db(pool, name, model, version) == set(record.chunk_hashes)\n",
        "",
        "disagrees",
    ),
    (
        "stop reusing vectors for identical text",
        "src/support_assistant/indexer.py",
        'reusable = {r["content_hash"]: r["embedding"] for r in await cur.fetchall()}',
        "reusable = {}",
        "renamed",
    ),
    (
        "let the publisher overwrite releases",
        "infra/terraform/cosmos.tf",
        '    publisher = concat(local.registry_read, [\n',
        '    publisher = concat(local.registry_read, [\n      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/upsert",\n',
        "published_release",
    ),
]

for label, path, old, new, select in CASES:
    p = Path(path)
    original = p.read_bytes()
    text = original.decode("utf-8").replace("\r\n", "\n")
    if text.count(old) != 1:
        print(f"{label}: could not apply ({text.count(old)} matches)")
        continue
    try:
        p.write_bytes(text.replace(old, new).encode("utf-8"))
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "tests/test_registry.py", "-k", select],
            capture_output=True, text=True, timeout=200,
        )
        verdict = "caught" if run.returncode == 1 else f"NOT caught (exit {run.returncode})"
        print(f"{label}: {verdict}")
    finally:
        p.write_bytes(original)
