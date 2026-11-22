"""Migration safety lint (E11-A7): expand-and-contract only.

    python scripts/check_migrations.py [migrations/versions]

A migration that destroys data or breaks the running release (DROP TABLE / DROP COLUMN /
RENAME / ALTER TYPE / SET NOT NULL without a default / TRUNCATE) fails CI unless it carries the
explicit marker comment `-- contract-ok: <reason>` in its upgrade(), which forces a deliberate,
reviewed decision. Downgrade functions are not checked (they are destructive by nature).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

DANGEROUS = [
    (re.compile(r"\bDROP\s+TABLE\b", re.I), "DROP TABLE"),
    (re.compile(r"\bDROP\s+COLUMN\b", re.I), "DROP COLUMN"),
    (re.compile(r"\bRENAME\s+(COLUMN|TO)\b", re.I), "RENAME"),
    (re.compile(r"\bALTER\s+COLUMN\s+\w+\s+(SET\s+DATA\s+)?TYPE\b", re.I), "ALTER COLUMN TYPE"),
    (re.compile(r"\bSET\s+NOT\s+NULL\b", re.I), "SET NOT NULL"),
    (re.compile(r"\bTRUNCATE\b", re.I), "TRUNCATE"),
]
MARKER = re.compile(r"--\s*contract-ok:\s*\S+")


def upgrade_source(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade":
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    return ""


def problems_in(path: Path) -> list[str]:
    src = upgrade_source(path)
    if MARKER.search(src):
        return []
    return [
        f"{path.name}: {label} in upgrade() without '-- contract-ok: <reason>'"
        for rx, label in DANGEROUS
        if rx.search(src)
    ]


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("migrations/versions")
    problems = [p for f in sorted(root.glob("*.py")) for p in problems_in(f)]
    for p in problems:
        print(f"UNSAFE MIGRATION  {p}")
    print(f"checked {len(list(root.glob('*.py')))} migrations, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
