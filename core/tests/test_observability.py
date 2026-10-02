from __future__ import annotations

from base64 import b64decode
from dataclasses import replace
from typing import cast

import pytest
from opentelemetry.sdk.trace import Span
from test_runner import request

from fora.adapters.model.observability import (
    ObservabilityConfig,
    TurnTrace,
    active_trace,
    current_trace,
)
from fora.runtime.reminder import Reminder, ReminderItem


def turn():
    reminder = Reminder(
        13,
        "Main",
        (
            ReminderItem(3, "rewrite", 7, 1, "You", False),
            ReminderItem(3, "rewrite", 9, 1, "You", True),
            ReminderItem(5, "sandbox", 2, 36, "Technical Manager", False),
        ),
    )
    return replace(
        request(),
        agent_id=13,
        agent_name="Main",
        reminder=reminder,
        prompt=reminder.render(),
    )


def test_config_requires_enabling_and_all_credentials() -> None:
    assert ObservabilityConfig.restore(None) is None
    assert ObservabilityConfig.restore({"enabled": False}) is None
    assert (
        ObservabilityConfig.restore(
            {"enabled": True, "base_url": "u", "public_key": "p"}
        )
        is None
    )


def test_endpoint_follows_the_langfuse_otlp_path() -> None:
    config = ObservabilityConfig.restore(
        {
            "enabled": True,
            "base_url": "https://us.cloud.langfuse.com/",
            "public_key": "pk",
            "secret_key": "sk",
        }
    )
    assert config is not None
    assert (
        config.endpoint() == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
    )


def test_headers_carry_basic_auth_and_the_ingestion_version() -> None:
    config = ObservabilityConfig.restore(
        {"enabled": True, "base_url": "u", "public_key": "pk", "secret_key": "sk"}
    )
    assert config is not None
    headers = config.headers()
    assert headers["x-langfuse-ingestion-version"] == "4"

    encoded = headers["Authorization"].removeprefix("Basic ")
    assert headers["Authorization"].startswith("Basic ")
    assert "pk:sk" not in encoded
    assert b64decode(encoded).decode() == "pk:sk"


def test_redacted_config_never_exposes_the_keys() -> None:
    config = ObservabilityConfig.restore(
        {
            "enabled": True,
            "base_url": "u",
            "public_key": "public-value",
            "secret_key": "secret-value",
        }
    )
    assert config is not None
    rendered = str(config.redacted())
    assert "secret-value" not in rendered
    assert "public-value" not in rendered
    assert config.redacted()["keys_set"] is True


def test_trace_summarises_the_turn() -> None:
    trace = TurnTrace.of(turn())
    assert trace.agent_id == 13
    assert trace.agent_name == "Main"
    assert trace.discussion_ids == (3, 5)
    assert trace.message_count == 3


def test_preparation_trace_has_identity_without_reminders() -> None:
    trace = TurnTrace.of(replace(turn(), reminder=None))
    assert trace.agent_id == 13
    assert trace.agent_name == "Main"
    assert trace.discussion_ids == ()
    assert trace.message_count == 0


def test_trace_attributes_match_the_langfuse_schema() -> None:
    attributes = TurnTrace.of(turn()).attributes()
    assert attributes["langfuse.trace.name"] == "Agent turn"
    assert attributes["langfuse.trace.tags"] == ["fora", "agent"]
    assert attributes["langfuse.trace.metadata.agent_id"] == 13
    assert attributes["langfuse.trace.metadata.discussion_ids"] == [3, 5]
    assert attributes["langfuse.trace.metadata.message_count"] == 3
    assert attributes["langfuse.session.id"] == "fora-agent-13"


def test_trace_scope_is_restored_after_use() -> None:
    assert current_trace() is None
    trace = TurnTrace.of(turn())
    with active_trace(trace):
        assert current_trace() is trace
    assert current_trace() is None


def test_span_processor_only_decorates_model_spans() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from fora.adapters.model.langfuse import TurnAttributeProcessor

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSpan:
        def __init__(self, scope: str | None) -> None:
            self.instrumentation_scope = FakeScope(scope) if scope else None
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    processor = TurnAttributeProcessor()
    with active_trace(TurnTrace.of(turn())):
        model_span = FakeSpan("pydantic-ai")
        other_span = FakeSpan("something-else")
        processor.on_start(cast(Span, model_span))
        processor.on_start(cast(Span, other_span))

    assert model_span.attributes["langfuse.trace.name"] == "Agent turn"
    assert other_span.attributes == {}


def test_spans_are_not_decorated_outside_a_turn() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from fora.adapters.model.langfuse import TurnAttributeProcessor

    class FakeSpan:
        def __init__(self) -> None:
            self.instrumentation_scope = type("S", (), {"name": "pydantic-ai"})()
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    span = FakeSpan()
    TurnAttributeProcessor().on_start(cast(Span, span))
    assert span.attributes == {}
