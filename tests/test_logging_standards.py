"""What reaches a log: correlation fields, hashed identifiers, no customer text, and an OTLP copy."""

import json

import structlog
from opentelemetry import _logs

from support_assistant import logs

SESSION = "s3cretSessionToken_abcdefghijklmnop"


def test_a_session_id_or_customer_id_passed_by_any_call_site_is_never_printed(log_lines):
    structlog.get_logger().info("x", session_id=SESSION, customer_id="C1001")
    (line,) = log_lines()
    assert SESSION not in json.dumps(line) and "C1001" not in json.dumps(line)
    assert line["session.id"] == line["gen_ai.conversation.id"] == logs.pseudonym(SESSION)
    assert line["enduser.pseudo.id"] == logs.pseudonym("C1001")
    assert "session_id" not in line and "customer_id" not in line


def test_fields_that_could_carry_what_a_customer_typed_are_redacted(log_lines):
    structlog.get_logger().info("x", message="my pin is 1234", phone="+15550001111", query="q")
    (line,) = log_lines()
    assert line["message"] == line["phone"] == line["query"] == "[redacted]"


def test_an_exception_keeps_its_type_and_frames_but_not_its_message(log_lines):
    try:
        raise ValueError("card 4111111111111111 for jane@example.com")
    except ValueError:
        structlog.get_logger().exception("failed")
    (line,) = log_lines()
    assert (
        line["error.type"] == "ValueError"
        and "test_logging_standards" in line["exception.stacktrace"]
    )
    assert "exc_info" not in line
    # the source line of the raise is part of a stack frame; the formatted message line is not
    assert "ValueError: card" not in line["exception.stacktrace"]


def test_conversation_binds_the_fields_for_its_block_only(log_lines):
    log = structlog.get_logger()
    with logs.conversation(SESSION, "C1", "turn-0001"):
        log.info("inside")
    log.info("outside")
    inside, outside = log_lines()
    assert inside["session.id"] == logs.pseudonym(SESSION)
    assert inside["conversation.turn.id"] == "turn-0001"
    assert "session.id" not in outside and "conversation.turn.id" not in outside


def test_every_line_is_also_emitted_through_the_otel_logs_api(log_lines, monkeypatch):
    emitted: list[dict] = []

    class Sink:
        def emit(self, **record):
            emitted.append(record)

    monkeypatch.setattr(_logs, "get_logger", lambda *a, **k: Sink())
    structlog.get_logger().warning("retry_scheduled", session_id=SESSION, attempt=2, ids=[1, 2])
    (record,) = emitted
    assert record["body"] == record["event_name"] == "retry_scheduled"
    assert record["severity_text"] == "WARNING"
    attributes = record["attributes"]
    assert attributes["attempt"] == 2 and attributes["ids"] == "[1, 2]"
    assert attributes["session.id"] == logs.pseudonym(SESSION)
    assert SESSION not in json.dumps(attributes)


def test_a_broken_exporter_never_breaks_the_work_being_logged(log_lines, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("collector down")

    monkeypatch.setattr(_logs, "get_logger", boom)
    structlog.get_logger().info("still_printed")
    assert log_lines()[0]["event"] == "still_printed"


def test_hashes_are_stable_keyed_and_ignore_case_and_spacing(monkeypatch):
    monkeypatch.delenv("LOG_HASH_KEY", raising=False)
    plain = logs.pseudonym("C1001")
    assert logs.fingerprint("No   SIGNAL") == logs.fingerprint("no signal")
    monkeypatch.setenv("LOG_HASH_KEY", "k1")
    keyed = logs.pseudonym("C1001")
    assert keyed != plain and keyed == logs.pseudonym("C1001") and len(keyed) == 24


def test_a_caller_cannot_write_free_text_into_logs_through_a_header():
    assert logs.header_value({"x-turn-id": "abcdef12"}, "x-turn-id") == "abcdef12"
    for bad in ("short", "has space in it", 'inject"}\n{"event":"x', "a" * 65):
        assert logs.header_value({"x-turn-id": bad}, "x-turn-id") is None
