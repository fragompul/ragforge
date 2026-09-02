"""Tests for the dependency-free tracing module."""

from __future__ import annotations

import sys
import time
import types

import pytest

from ragforge.telemetry import InMemoryExporter, Tracer, console_exporter, otel_exporter


def test_span_duration_ms_is_zero_before_end():
    tracer = Tracer()
    with tracer.span("work") as span:
        assert span.end_time is None
        assert span.duration_ms == 0.0


def test_span_records_positive_duration_after_context_exit():
    exporter = InMemoryExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.span("work"):
        time.sleep(0.001)

    assert len(exporter.spans) == 1
    assert exporter.spans[0].duration_ms > 0.0
    assert exporter.spans[0].end_time is not None


def test_nested_spans_record_parent_child_relationship():
    exporter = InMemoryExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.span("outer") as outer, tracer.span("inner") as inner:
        pass

    assert inner.parent_id == outer.span_id
    assert outer.parent_id is None
    # Inner span completes (and is exported) before the outer span does.
    assert [s.name for s in exporter.spans] == ["inner", "outer"]


def test_span_attributes_are_captured():
    exporter = InMemoryExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.span("search", query="cats", k=5):
        pass

    assert exporter.spans[0].attributes == {"query": "cats", "k": 5}


def test_span_exception_still_ends_and_exports():
    exporter = InMemoryExporter()
    tracer = Tracer(exporters=[exporter])

    with pytest.raises(ValueError), tracer.span("boom"):
        raise ValueError("kaboom")

    assert len(exporter.spans) == 1
    assert exporter.spans[0].end_time is not None


def test_in_memory_exporter_filters_and_sums_by_name():
    exporter = InMemoryExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    with tracer.span("a"):
        pass

    assert len(exporter.by_name("a")) == 2
    assert exporter.total_duration_ms() >= exporter.total_duration_ms("a")

    exporter.clear()
    assert exporter.spans == []


def test_console_exporter_prints_duration_and_attributes(capsys):
    tracer = Tracer(exporters=[console_exporter])
    with tracer.span("query", term="paris"):
        pass

    captured = capsys.readouterr()
    assert "query" in captured.out
    assert "term=paris" in captured.out
    assert "ms]" in captured.out


def test_span_to_dict_round_trips_expected_keys():
    tracer = Tracer()
    with tracer.span("work", x=1) as span:
        pass

    d = span.to_dict()
    assert d["name"] == "work"
    assert d["attributes"] == {"x": 1}
    assert "duration_ms" in d
    assert d["parent_id"] is None


def test_otel_exporter_raises_helpful_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with pytest.raises(ImportError, match="pip install opentelemetry"):
        otel_exporter()


def test_otel_exporter_forwards_span_to_provided_tracer():
    calls: dict[str, object] = {}

    class FakeOtelSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.ended_at: int | None = None

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

        def end(self, end_time: int) -> None:
            self.ended_at = end_time

    class FakeOtelTracer:
        def start_span(self, name: str, start_time: int) -> FakeOtelSpan:
            calls["name"] = name
            calls["start_time"] = start_time
            span = FakeOtelSpan()
            calls["span"] = span
            return span

    export = otel_exporter(tracer=FakeOtelTracer())
    tracer = Tracer(exporters=[export])

    with tracer.span("retrieve", k=3):
        pass

    assert calls["name"] == "retrieve"
    fake_span = calls["span"]
    assert isinstance(fake_span, FakeOtelSpan)
    assert fake_span.attributes["k"] == 3
    assert "ragforge.span_id" in fake_span.attributes
    assert fake_span.ended_at is not None


def test_otel_exporter_uses_global_tracer_when_none_provided(monkeypatch):
    fake_trace_module = types.ModuleType("opentelemetry")

    class FakeGlobalTracer:
        def start_span(self, name, start_time):
            class _Span:
                def set_attribute(self, *a, **k):
                    pass

                def end(self, end_time):
                    pass

            return _Span()

    def get_tracer(name: str) -> FakeGlobalTracer:
        assert name == "ragforge"
        return FakeGlobalTracer()

    fake_trace_submodule = types.SimpleNamespace(get_tracer=get_tracer)
    fake_trace_module.trace = fake_trace_submodule
    monkeypatch.setitem(sys.modules, "opentelemetry", fake_trace_module)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace_submodule)

    export = otel_exporter()
    tracer = Tracer(exporters=[export])
    with tracer.span("work"):
        pass
