"""Shared test setup."""

import io
import json

import pytest
import structlog

from support_assistant import telemetry


@pytest.fixture(autouse=True)
def _isolate_backend_state():
    """The fat backend is one module-level app, and tests configure it by setting `app.state`
    attributes. Put the state back after every test so one test's settings, limiter or fakes can
    never leak into the next (a strict settings object left behind once made unrelated tests fail)."""
    from support_assistant.backend import app as backend

    saved = dict(backend.app.state._state)
    yield
    backend.app.state._state.clear()
    backend.app.state._state.update(saved)


@pytest.fixture
def log_lines():
    """The JSON lines printed by the real processor chain, written to a buffer of our own. Each call
    returns the lines logged since the previous one."""
    saved = structlog.get_config()
    telemetry._configure_logging()
    out = io.StringIO()
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(out))

    def read() -> list[dict]:
        text = out.getvalue()
        out.seek(0)
        out.truncate()
        return [json.loads(x) for x in text.splitlines() if x.startswith("{")]

    yield read
    structlog.contextvars.clear_contextvars()
    structlog.configure(**saved)
