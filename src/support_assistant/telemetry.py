"""OpenTelemetry setup for every service (E12: B1-B6). Standards first, vendor last.

- Traces, metrics and logs go over OTLP (http/protobuf) to the in-cluster OpenTelemetry Collector
  (`OTEL_EXPORTER_OTLP_ENDPOINT`). The collector does tail sampling, PII redaction and export to
  Azure Monitor, so services stay vendor-neutral and keep 100% of spans locally (AlwaysOn); the
  sampling decision is made once, at the collector, with the whole trace in view.
- Context propagation is W3C Trace Context + Baggage (`traceparent`), injected and extracted by the
  httpx / FastAPI / ASGI instrumentations, so one customer turn is ONE trace across services.
- Resource attributes follow the semantic conventions: service.name, service.namespace,
  service.version, deployment.environment.name, plus k8s.* added by the collector.
- Logs: structlog JSON on stdout with trace_id / span_id on every line. Each line is also emitted
  through the OTel Logs API with its fields as attributes, so it reaches the collector over OTLP.
  logs.py hashes identifiers and redacts customer text before either output. The stdlib handler
  below only carries what third-party libraries log through `logging`; those lines are not scrubbed
  here (the collector's redaction applies).
- GenAI: Microsoft Agent Framework's instrumentation (OTel GenAI conventions) is enabled with
  prompt / completion capture OFF (PII). `OTEL_GENAI_CAPTURE_CONTENT=true` turns it on (dev only).

With no OTLP endpoint (tests, CI) everything is a no-op except JSON logging.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace

from support_assistant.logs import describe_exception, export_to_otel, protect_identifiers

SERVICE_NAMESPACE = "support-assistant"
_configured = False


def _add_trace_ids(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor: put the active trace and span ids on every log line."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event["trace_id"] = format(ctx.trace_id, "032x")
        event["span_id"] = format(ctx.span_id, "016x")
    return event


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_trace_ids,
            # In this order: hide identifiers and customer text, make exceptions safe, then export.
            protect_identifiers,
            describe_exception,
            export_to_otel,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )


_SESSION_IN_URL = re.compile(r"(session_id=)[^&\s\"]+")


class RedactSessionInUrls(logging.Filter):
    """`GET /uploads?session_id=...` carries the session id in the URL, and uvicorn's access log
    prints URLs. This rewrites the value before the line is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _SESSION_IN_URL.sub(r"\1[redacted]", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


def resource_attributes(service: str) -> dict[str, str]:
    return {
        "service.name": service,
        "service.namespace": SERVICE_NAMESPACE,
        "service.version": os.environ.get("SERVICE_VERSION", "dev"),
        "deployment.environment.name": os.environ.get("DEPLOYMENT_ENVIRONMENT", "dev"),
    }


def setup_telemetry(service: str) -> None:
    """Idempotent. Call once per process at startup."""
    global _configured
    _configure_logging()
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(x, RedactSessionInUrls) for x in access.filters):
        access.addFilter(RedactSessionInUrls())
    if _configured or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    _configured = True

    from opentelemetry import metrics
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

    resource = Resource.create(resource_attributes(service))

    # Traces: keep everything here; the collector tail-samples with the whole trace in view.
    tracer_provider = TracerProvider(resource=resource, sampler=ParentBased(ALWAYS_ON))
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    metrics.set_meter_provider(
        MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(), 15_000)],
        )
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    logging.getLogger().addHandler(
        LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    )

    # Auto-instrumentation of outgoing calls and stores (W3C traceparent is injected by httpx).
    HTTPXClientInstrumentor().instrument()
    PsycopgInstrumentor().instrument(enable_commenter=False)  # no SQL comments with trace ids
    RedisInstrumentor().instrument()

    # GenAI spans from Microsoft Agent Framework; message content stays out unless asked for.
    capture = os.environ.get("OTEL_GENAI_CAPTURE_CONTENT", "").lower() == "true"
    try:
        from agent_framework.observability import enable_instrumentation

        enable_instrumentation(enable_sensitive_data=capture)
    except Exception:  # noqa: BLE001 - telemetry must never stop a service from starting
        structlog.get_logger().warning("maf_instrumentation_unavailable")


def instrument_fastapi(app: Any) -> None:
    """Server spans for a FastAPI app. Safe to call at import time, before setup_telemetry: spans
    are created through the global provider when requests arrive."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")


def instrument_asgi(app: Any) -> Any:
    """Server spans for a plain ASGI app (the MCP servers), extracting the caller's traceparent."""
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

    return OpenTelemetryMiddleware(app)
