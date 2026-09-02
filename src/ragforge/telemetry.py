"""Lightweight, dependency-free tracing for RAG pipeline observability.

Debugging a slow or wrong RAG answer requires knowing *where* time went and
*what* happened at each stage: was BM25 slow, was vector search, was
reranking, or was the LLM call itself? ``RagAnswer.total_latency_ms`` (see
:mod:`ragforge.pipeline`) already splits retrieval from generation, but it
cannot see *inside* retrieval. This module adds structured, nested spans
around each pipeline stage with a pluggable exporter, so the same
instrumentation can:

- print a human-readable trace during local debugging (``console_exporter``),
- collect everything in memory for assertions in tests or notebooks
  (``InMemoryExporter``), or
- forward spans into a real observability backend via the OpenTelemetry
  bridge (``otel_exporter``), without ragforge itself depending on the
  ``opentelemetry`` package.

This is a deliberately minimal, opinionated subset of what OpenTelemetry
offers -- a span with a name, timing, attributes, and parent-child nesting
via a stack -- not a general tracing SDK. If a project already has full
OpenTelemetry instrumentation, use ``otel_exporter`` to fold ragforge's
spans into it rather than running two competing tracing systems side by
side.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A single timed unit of work, optionally nested under a parent span."""

    name: str
    span_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration in milliseconds, or 0.0 if not yet ended."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 4),
            "attributes": dict(self.attributes),
        }


SpanExporter = Callable[[Span], None]


def console_exporter(span: Span) -> None:
    """Print a single-line, indented trace as each span completes."""
    indent = "  " if span.parent_id else ""
    attrs = " ".join(f"{key}={value}" for key, value in span.attributes.items())
    print(f"{indent}[{span.duration_ms:8.3f}ms] {span.name} {attrs}".rstrip())


class InMemoryExporter:
    """Collects every completed span in order, for tests and notebooks."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def __call__(self, span: Span) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def total_duration_ms(self, name: str | None = None) -> float:
        """Sum of span durations, optionally filtered to one span name."""
        return sum(s.duration_ms for s in self.spans if name is None or s.name == name)


def otel_exporter(tracer: Any = None) -> SpanExporter:
    """Bridge completed ragforge spans into a real OpenTelemetry tracer.

    Requires ``pip install opentelemetry-api`` (plus an SDK and configured
    exporter for spans to actually go anywhere). Pass a pre-configured
    ``tracer`` (from ``opentelemetry.trace.get_tracer(...)``) to control
    sampling and resource attributes; otherwise the globally configured
    tracer provider is used.

    Limitation: ragforge spans are only known to be complete *after* they
    end (see ``Tracer.span``), so this replays each finished span into
    OpenTelemetry via ``start_span``/``end`` with explicit timestamps rather
    than the live ``start_as_current_span`` context manager. Parent-child
    relationships are not reconstructed in the OTel span tree -- every
    exported span is a root span carrying its ragforge ``parent_id`` and
    ``span_id`` as attributes instead. For full trace-tree fidelity in an
    OpenTelemetry backend, instrument spans directly with the OTel SDK.
    """
    if tracer is not None:
        resolved_tracer = tracer
    else:
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise ImportError(
                "'opentelemetry-api' is required for otel_exporter. Install it with:\n"
                "    pip install opentelemetry-api opentelemetry-sdk"
            ) from exc
        resolved_tracer = trace.get_tracer("ragforge")

    def _export(span: Span) -> None:
        otel_span = resolved_tracer.start_span(
            span.name, start_time=int(span.start_time * 1_000_000_000)
        )
        otel_span.set_attribute("ragforge.span_id", span.span_id)
        otel_span.set_attribute("ragforge.parent_id", span.parent_id or "")
        for key, value in span.attributes.items():
            otel_span.set_attribute(key, value)
        end_time = span.end_time if span.end_time is not None else span.start_time
        otel_span.end(end_time=int(end_time * 1_000_000_000))

    return _export


class Tracer:
    """Manages a stack of nested spans and dispatches completed spans to exporters.

    Not thread-safe by design: the span stack mirrors call-stack nesting for
    a single synchronous request, matching how ``RagPipeline`` uses it. Share
    one ``Tracer`` per pipeline instance, not across concurrent requests.
    """

    def __init__(self, exporters: list[SpanExporter] | None = None) -> None:
        self.exporters = exporters or []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span, nested under whichever span is currently open (if any)."""
        parent = self._stack[-1] if self._stack else None
        current = Span(
            name=name,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_time=time.time(),
            attributes=dict(attributes),
        )
        self._stack.append(current)
        try:
            yield current
        finally:
            self._stack.pop()
            current.end_time = time.time()
            for exporter in self.exporters:
                exporter(current)
