"""Terminal client for demos (E6-4):  python -m support_assistant.cli http://localhost:8000

Talks to the fat backend over HTTP exactly like any other client. Type `quit` to leave.
"""

from __future__ import annotations

import sys
import uuid
from datetime import date

import httpx


def _show(result: dict[str, object]) -> bool:
    """Print a turn. Returns True if the backend is waiting for payment approval."""
    pending = result.get("pending_approval")
    if pending:
        print(f"\nassistant> {pending['question']}")  # type: ignore[index]
        return True
    print(f"\nassistant> {result.get('reply', '')}")
    return False


def run(base_url: str, ask: object = input) -> None:
    session = uuid.uuid4().hex
    with httpx.Client(base_url=base_url, timeout=60.0) as http:
        phone = ask("phone number: ")  # type: ignore[operator]
        dob = date.fromisoformat(ask("date of birth (YYYY-MM-DD): "))  # type: ignore[operator]
        r = http.post(
            "/verify",
            json={"session_id": session, "phone": phone, "date_of_birth": dob.isoformat()},
        )
        if r.status_code != 200:
            print(r.json().get("message", "Verification failed."))
            return
        while True:
            text = ask("\nyou> ").strip()  # type: ignore[operator]
            if text.lower() in {"quit", "exit"}:
                return
            if not text:
                continue
            r = http.post("/chat", json={"session_id": session, "message": text})
            result = r.json()
            if r.status_code >= 400:
                print(f"\nassistant> {result.get('message') or result.get('detail')}")
                continue
            if _show(result):
                answer = ask("confirm? (yes/no): ").strip().lower()  # type: ignore[operator]
                r = http.post("/approve", json={"session_id": session, "approved": answer == "yes"})
                _show(r.json())


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
