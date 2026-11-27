"""The tool servers run as several replicas: any replica must be able to answer any request."""

from starlette.testclient import TestClient

from support_assistant.mcp_servers.device import build_device_server
from support_assistant.mcp_servers.main import http_app

HEADERS = {"Accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def test_a_session_opened_on_one_replica_works_on_another() -> None:
    with (
        TestClient(http_app(build_device_server())) as first,
        TestClient(http_app(build_device_server())) as second,
    ):
        opened = first.post("/mcp", json=INITIALIZE, headers=HEADERS)
        assert opened.status_code == 200

        session = opened.headers.get("mcp-session-id")
        headers = HEADERS | ({"mcp-session-id": session} if session else {})
        answer = second.post("/mcp", json=LIST_TOOLS, headers=headers)

        assert answer.status_code == 200, answer.text
        assert '"tools"' in answer.text
