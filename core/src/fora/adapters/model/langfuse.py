from __future__ import annotations

from typing import Any

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from pydantic_ai import InstrumentationSettings
from pydantic_ai.capabilities import Instrumentation

from fora.adapters.model.observability import ObservabilityConfig, current_trace

SCOPE = "pydantic-ai"


class TurnAttributeProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        del parent_context
        trace = current_trace()
        scope = span.instrumentation_scope
        if trace is None or scope is None or scope.name != SCOPE:
            return
        for key, value in trace.attributes().items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        del span

    def shutdown(self) -> None:
        return None


class LangfuseObservability:
    def __init__(
        self,
        config: ObservabilityConfig,
        exporter: SpanExporter | None = None,
    ) -> None:
        self._provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "fora",
                    "deployment.environment.name": config.environment,
                }
            )
        )
        self._provider.add_span_processor(TurnAttributeProcessor())
        self._provider.add_span_processor(
            BatchSpanProcessor(
                exporter
                or OTLPSpanExporter(
                    endpoint=config.endpoint(), headers=config.headers()
                ),
                schedule_delay_millis=500,
            )
        )
        self._settings = InstrumentationSettings(
            tracer_provider=self._provider,
            include_content=config.capture_content,
            include_binary_content=False,
            include_model_request_parameters=False,
        )

    def instrumentation(self) -> Any:
        return Instrumentation(settings=self._settings)

    def shutdown(self) -> None:
        self._provider.shutdown()
