from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from dataclasses import replace
from itertools import pairwise
from typing import Any

import pytest
from pydantic_ai import (
    BinaryContent,
    ModelMessagesTypeAdapter,
    ModelRetry,
    Tool,
    models,
)
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelHTTPError,
    SkipToolExecution,
    ToolFailed,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from huddol.adapters.model.config import ModelConfig, thinking_settings
from huddol.adapters.model.observability import ObservabilityConfig
from huddol.adapters.model.prompt import SYSTEM_PROMPT
from huddol.adapters.model.runner import (
    UNAVAILABLE,
    LiveModel,
    PydanticModelRunner,
    is_context_exceeded,
)
from huddol.core.attachment import ImageData, ViewedImage
from huddol.core.errors import DomainError
from huddol.runtime.reminder import (
    HistoryPersistenceError,
    Reminder,
    ReminderItem,
    TurnRequest,
)


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def get_settings(self, section: str) -> dict[str, object] | None:
        return self.values.get(section)

    def set_settings(self, section: str, values: dict[str, object]) -> None:
        self.values[section] = values


class FakeObservability:
    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self.stopped = False
        self.calls = 0

    def instrumentation(self) -> Any:
        self.calls += 1
        return Hooks()

    def shutdown(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def no_network(monkeypatch) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def request() -> TurnRequest:
    reminder = Reminder(2, "Main", (ReminderItem(1, "work", 1, 1, "You", False),))
    return TurnRequest(
        agent_id=2,
        sequence=1,
        agent_name="Main",
        prompt=reminder.render(),
        reminder=reminder,
        history_json="[]",
        resident="Your MEMORY.md is empty.\n\nenvironment A",
        environment=lambda: "environment A",
        ephemeral=lambda: "",
        persist=lambda messages_json: None,
    )


def model_values(model: str) -> dict[str, object]:
    return {
        "api_type": "openai-chat",
        "base_url": "https://example.invalid/v1",
        "api_key": "unused",
        "model": model,
    }


@pytest.fixture
def settings() -> FakeSettings:
    settings = FakeSettings()
    settings.set_settings("model", model_values("first"))
    return settings


class Builder:
    def __init__(self, delay: threading.Event | None = None) -> None:
        self.calls: list[ModelConfig] = []
        self.entered = threading.Event()
        self._delay = delay

    def __call__(self, config: ModelConfig) -> FunctionModel:
        self.calls.append(config)
        self.entered.set()
        if self._delay is not None:
            assert self._delay.wait(timeout=5)
        return FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart(config.model)])
        )


def assert_settlement(message, calls) -> None:
    assert isinstance(message, ModelRequest)
    assert all(isinstance(part, ToolReturnPart) for part in message.parts)
    assert [
        (part.tool_name, part.tool_call_id, part.content) for part in message.parts
    ] == [
        (
            call.tool_name,
            call.tool_call_id,
            "The Turn ended without a confirmed result for this call. It may have executed; check the outcome before retrying.",
        )
        for call in calls
    ]


def cache_key(messages) -> str:
    key = messages[0].metadata["huddol"]["cache_key"]
    assert re.fullmatch(r"[0-9a-f]{32}", key)
    return key


def without_cache_key(messages) -> list:
    first = messages[0]
    huddol = {
        name: value
        for name, value in first.metadata["huddol"].items()
        if name != "cache_key"
    }
    metadata = {
        name: value for name, value in first.metadata.items() if name != "huddol"
    }
    if huddol:
        metadata["huddol"] = huddol
    return [replace(first, metadata=metadata or None), *messages[1:]]


def flattened(messages) -> list[tuple]:
    parts: list[tuple] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                parts.append(("call", part.tool_call_id, part.args_as_json_str()))
            elif isinstance(part, ToolReturnPart):
                parts.append(("return", part.tool_call_id, part.model_response_str()))
            else:
                parts.append((part.part_kind, part.content))
    return parts


@pytest.mark.parametrize("with_text", [False, True])
def test_next_turn_settles_unanswered_history_calls(settings, with_text) -> None:
    calls = [
        ToolCallPart("library", {"action": "list"}, tool_call_id="library-list"),
        ToolCallPart("workspace", {"action": "list"}, tool_call_id="workspace-list"),
    ]
    history = [
        ModelRequest(parts=[UserPromptPart("Old prompt")]),
        ModelResponse(parts=[TextPart("Checking"), *calls] if with_text else calls),
    ]
    received = []
    persisted = []

    def respond(messages, info):
        received.append(messages)
        assert without_cache_key(messages)[:2] == history
        assert_settlement(
            ModelRequest(
                parts=[
                    part
                    for part in messages[2].parts
                    if isinstance(part, ToolReturnPart)
                ]
            ),
            calls,
        )
        return ModelResponse(parts=[TextPart("Done")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(
        replace(
            request(),
            history_json=ModelMessagesTypeAdapter.dump_json(history).decode(),
            persist=persisted.append,
        ),
        None,
    )
    assert outcome.error is None
    assert len(received) == 1
    assert persisted == [outcome.messages_json]
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert without_cache_key(saved)[:2] == history
    assert_settlement(saved[2], calls)
    assert '"block":"resident"' not in outcome.messages_json


def test_exhausted_tool_retries_settle_failure_history_and_allow_next_turn(
    settings,
) -> None:
    persisted = []
    calls = []

    def respond(messages, info):
        call = ToolCallPart(
            "history",
            {"action": "read", "sequence": 99},
            tool_call_id=f"read-{len(calls)}",
        )
        calls.append(call)
        return ModelResponse(parts=[call])

    class Tools:
        def read_history(self, sequence):
            raise DomainError("not_found", "Run does not exist")

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(replace(request(), persist=persisted.append), Tools())
    assert outcome.error is not None and "exceeded max retries" in outcome.error
    assert len(calls) == len(persisted) - 1 == 3
    for raw, call in zip(persisted[1:], calls, strict=True):
        assert_settlement(ModelMessagesTypeAdapter.validate_json(raw)[-1], [call])
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert saved[-2].parts == [calls[-1]]
    assert_settlement(saved[-1], [calls[-1]])
    received = []

    def resume(messages, info):
        received.append(messages)
        assert_settlement(
            ModelRequest(
                parts=[
                    part
                    for message in messages
                    for part in message.parts
                    if isinstance(part, ToolReturnPart)
                ]
            ),
            [calls[-1]],
        )
        return ModelResponse(parts=[TextPart("Recovered")])

    following = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(resume)
    ).run(replace(request(), history_json=outcome.messages_json), None)
    assert following.error is None
    assert len(received) == 1
    assert (
        ModelMessagesTypeAdapter.validate_json(following.messages_json)[: len(saved)]
        == saved
    )


def test_prompt_does_not_warn_against_bare_numbers() -> None:
    assert "bare number" not in SYSTEM_PROMPT


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("priced", [False, True])
def test_each_response_persists_the_complete_history_without_ephemeral(
    settings, prior, priced
) -> None:
    persisted = []
    responses = []
    environment = "environment A"

    def respond(messages, info):
        assert len(persisted) - (not prior) == len(responses)
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    "organization", {"action": "list_members"}, tool_call_id="list"
                )
                if not responses
                else TextPart("Done")
            ],
            provider_name="openai" if priced else None,
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )
        responses.append(response)
        return response

    class Tools:
        def list_members(self):
            nonlocal environment
            assert len(persisted) == 1 + (not prior)
            environment = "environment B"
            return [{"id": 1, "title": "Work"}]

    original = request()
    if prior:
        previous = PydanticModelRunner(settings, build_model=Builder()).run(
            original, None
        )
        original = replace(original, history_json=previous.messages_json)
    outcome = PydanticModelRunner(
        settings,
        build_model=lambda config: FunctionModel(
            respond, model_name="gpt-4o" if priced else "local"
        ),
    ).run(
        replace(
            original,
            persist=persisted.append,
            ephemeral=lambda: "EPHEMERAL",
            environment=lambda: environment,
        ),
        Tools(),
    )
    assert outcome.error is None
    assert len(persisted) - (not prior) == len(responses) == 2
    first, second = [
        ModelMessagesTypeAdapter.validate_json(raw) for raw in persisted[not prior :]
    ]
    assert (responses[0].usage.cost is not None) == priced
    assert_settlement(first[-1], responses[0].parts)
    assert first[-2] == responses[0]
    assert first[-3].parts[0].content == original.prompt
    assert second == ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert second[: len(first) - 1] == first[:-1]
    assert "without a confirmed result" not in persisted[-1]
    assert second[-1] == responses[1]
    assert second[-2].metadata == {
        "huddol": {"block": "durable", "environment": "environment B"}
    }
    assert second[-3].parts == [
        ToolReturnPart(
            "organization",
            [{"id": 1, "title": "Work"}],
            tool_call_id="list",
            timestamp=second[-3].parts[0].timestamp,
        )
    ]
    assert first[0].parts[0].content == original.resident
    assert first[0].metadata == {
        "huddol": {
            "block": "resident",
            "agents_instructions": None,
            "environment": "environment A",
            "cache_key": cache_key(first),
        }
    }
    assert all("EPHEMERAL" not in raw for raw in persisted)
    if prior:
        history = ModelMessagesTypeAdapter.validate_json(original.history_json)
        assert first[: len(history)] == history


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("fail_at", [1, 2, 3])
@pytest.mark.parametrize("tool_count", [1, 3])
def test_persistence_failure_stops_model_and_all_response_tools(
    settings, prior, fail_at, tool_count
) -> None:
    original = request()
    if prior:
        previous = PydanticModelRunner(settings, build_model=Builder()).run(
            original, None
        )
        original = replace(original, history_json=previous.messages_json)
    saved = []
    model_calls = []
    tool_calls = []
    failure = OSError("disk unavailable")

    def persist(raw):
        saved.append(raw)
        if len(saved) == fail_at:
            raise failure

    class Tools:
        def list_members(self):
            tool_calls.append(1)
            return []

    def respond(messages, info):
        model_calls.append(1)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "organization",
                    {"action": "list_members"},
                    tool_call_id=f"call-{len(model_calls)}-{index}",
                )
                for index in range(tool_count)
            ]
        )

    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    with pytest.raises(HistoryPersistenceError) as caught:
        runner.run(replace(original, persist=persist), Tools())
    assert caught.value.__cause__ is failure
    assert len(saved) == fail_at
    assert len(model_calls) == fail_at - (not prior)
    assert len(tool_calls) == max(0, len(model_calls) - 1) * tool_count


@pytest.mark.parametrize("raw", ["", "{", "null", "{}", '[{"kind":"unknown"}]'])
def test_invalid_history_fails_before_model_creation_or_saving(settings, raw):
    built = []
    saved = []
    runner = PydanticModelRunner(
        settings, build_model=lambda config: built.append(config)
    )
    with pytest.raises(ValueError):
        runner.run(replace(request(), history_json=raw, persist=saved.append), None)
    assert built == []
    assert saved == []


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_serialization_failure_propagates_at_snapshot_response_and_outcome(
    settings, monkeypatch, fail_at
):
    serialize = ModelMessagesTypeAdapter.dump_json
    calls = []
    saved = []
    model_calls = []
    failure = ValueError("history cannot be serialized")

    def encode(messages, **kwargs):
        calls.append(1)
        if len(calls) == fail_at:
            raise failure
        return serialize(messages, **kwargs)

    def respond(messages, info):
        model_calls.append(1)
        return ModelResponse(parts=[TextPart("done")])

    monkeypatch.setattr(ModelMessagesTypeAdapter, "dump_json", encode)
    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    with pytest.raises(HistoryPersistenceError) as caught:
        runner.run(replace(request(), persist=saved.append), None)
    assert caught.value.__cause__ is failure
    assert len(calls) == fail_at
    assert len(saved) == fail_at - 1
    assert len(model_calls) == (fail_at > 1)


def test_invalid_tool_responses_are_persisted_before_retry(settings) -> None:
    persisted = []

    def respond(messages, info):
        if len(persisted) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("discussion", {"action": "delete", "discussion_id": 1})
                ]
            )
        return ModelResponse(parts=[TextPart("Done")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(replace(request(), persist=persisted.append), None)
    assert outcome.error is None
    assert len(persisted) == 3
    assert '"tool-call"' in persisted[1]
    assert '"retry-prompt"' in persisted[2]
    assert persisted[-1] == outcome.messages_json


def test_runs_are_unavailable_until_a_model_is_configured() -> None:
    settings = FakeSettings()
    builder = Builder()
    runner = PydanticModelRunner(settings, build_model=builder)
    original = replace(request(), history_json='[{"kind":"response"}]')
    for values in ({}, {"compaction_threshold": 5}):
        settings.set_settings("model", values)
        outcome = runner.run(original, None)
        assert outcome.error == UNAVAILABLE
        assert outcome.messages_json == original.history_json
        assert builder.calls == []


def test_the_next_turn_picks_up_settings_with_its_own_model(settings) -> None:
    builder = Builder()
    runner = PydanticModelRunner(settings, build_model=builder)
    agent = runner._agent
    for name in ("first", "first", "second"):
        settings.set_settings("model", model_values(name))
        outcome = runner.run(request(), None)
        assert outcome.error is None
        assert (
            ModelMessagesTypeAdapter.validate_json(outcome.messages_json)[-1]
            .parts[0]
            .content
            == name
        )
    assert [config.model for config in builder.calls] == ["first", "first", "second"]
    assert runner._agent is agent
    settings.set_settings(
        "model", {**model_values("second"), "compaction_threshold": 9}
    )
    assert runner.run(request(), None).error is None
    assert len(builder.calls) == 4
    settings.set_settings("model", {})
    assert runner.run(request(), None).error == UNAVAILABLE


def test_model_and_request_limit_changes_apply_together_on_next_turn(settings) -> None:
    settings.set_settings("agent", {"request_limit": 1})
    settings.set_settings("model", model_values("first"))
    built = []
    calls = []

    def build(config):
        built.append(config.model)
        requests = 0

        def respond(messages, info):
            nonlocal requests
            requests += 1
            calls.append(config.model)
            if requests == 1:
                return ModelResponse(
                    parts=[ToolCallPart("organization", {"action": "list_members"})]
                )
            return ModelResponse(parts=[TextPart("Done")])

        return FunctionModel(respond)

    class Tools:
        def list_members(self):
            settings.set_settings("agent", {"request_limit": 2})
            settings.set_settings("model", model_values("second"))
            return []

    runner = PydanticModelRunner(settings, build_model=build)
    first = runner.run(request(), Tools())
    assert "request_limit of 1" in first.error
    assert calls == ["first"]
    second = runner.run(replace(request(), history_json=first.messages_json), Tools())
    assert second.error is None
    assert built == ["first", "second"]
    assert calls == ["first", "second", "second"]


@pytest.mark.parametrize("remove", [False, True])
def test_model_settings_are_fixed_for_the_turn(settings, remove) -> None:
    received = []
    built = []

    def build(config):
        built.append(config.model)

        def respond(messages, info):
            received.append(config.model)
            if len(received) == 1:
                return ModelResponse(
                    parts=[ToolCallPart("organization", {"action": "list_members"})]
                )
            return ModelResponse(parts=[TextPart("Done")])

        return FunctionModel(respond)

    class Tools:
        def list_members(self):
            settings.set_settings("model", {} if remove else model_values("second"))
            return []

    runner = PydanticModelRunner(settings, build_model=build)
    outcome = runner.run(request(), Tools())
    assert received == ["first", "first"]
    assert built == ["first"]
    assert outcome.error is None
    assert json.loads(outcome.usage_json)["requests"] == 2
    assert json.loads(outcome.usage_json)["tool_calls"] == 1
    following = runner.run(request(), Tools())
    if remove:
        assert following.error == UNAVAILABLE
        assert received == ["first", "first"]
    else:
        assert following.error is None
        assert received == ["first", "first", "second"]
        assert built == ["first", "second"]


def test_turn_budget_is_snapshotted_until_the_next_turn(settings) -> None:
    catalog = {
        "version": 2,
        "providers": [
            {
                "id": "p",
                "name": "Anthropic",
                "api_type": "anthropic",
                "base_url": "https://example.invalid",
                "api_key": "test-only-key",
                "enabled": True,
            }
        ],
        "models": [
            {
                "id": "m",
                "provider_id": "p",
                "name": "Model",
                "model": "custom-alias",
                "enabled": True,
                "thinking_budget_tokens": 6000,
            }
        ],
        "default_model_id": "m",
        "default_thinking": "budget",
        "agent_configs": {},
    }
    settings.set_settings("model", catalog)
    configs: list[ModelConfig] = []

    def build(config: ModelConfig):
        configs.append(config)
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("organization", {"action": "list_members"})]
                )
            return ModelResponse(parts=[TextPart("Done")])

        return FunctionModel(respond)

    class Tools:
        def list_members(self):
            current = settings.get_settings("model")
            settings.set_settings(
                "model",
                {
                    **current,
                    "models": [
                        {**current["models"][0], "thinking_budget_tokens": 8000}
                    ],
                },
            )
            return []

    runner = PydanticModelRunner(settings, build_model=build)
    first = runner.run(request(), Tools())
    assert first.error is None
    assert len(configs) == 1
    assert configs[0].thinking == "budget"
    assert configs[0].thinking_budget_tokens == 6000
    second = runner.run(request(), Tools())
    assert second.error is None
    assert [config.thinking_budget_tokens for config in configs] == [6000, 8000]


@pytest.mark.parametrize("fail", [False, True])
def test_ephemeral_is_loaded_each_call_and_never_persisted(settings, fail) -> None:
    received = []
    nudges = iter(["EPHEMERAL_ONE", "EPHEMERAL_TWO"])

    def respond(messages, info):
        received.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
        if len(received) == 1:
            return ModelResponse(
                parts=[ToolCallPart("organization", {"action": "list_members"})]
            )
        if fail:
            raise RuntimeError("local failure")
        return ModelResponse(parts=[TextPart("Done")])

    class Tools:
        def list_members(self):
            return []

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(replace(request(), ephemeral=lambda: next(nudges)), Tools())
    assert len(received) == 2
    assert "EPHEMERAL_ONE" in received[0]
    assert "EPHEMERAL_TWO" in received[1]
    assert "EPHEMERAL_ONE" not in received[1]
    assert "EPHEMERAL" not in outcome.messages_json
    assert "tool-return" in outcome.messages_json
    assert (outcome.error is not None) == fail


def test_resident_is_inserted_only_for_empty_history(settings) -> None:
    runner = PydanticModelRunner(settings, build_model=Builder())
    original = request()
    first = runner.run(original, None)
    messages = ModelMessagesTypeAdapter.validate_json(first.messages_json)
    assert len(messages) == 3
    assert messages[0].parts[0].content == original.resident
    assert messages[0].metadata == {
        "huddol": {
            "block": "resident",
            "agents_instructions": None,
            "environment": "environment A",
            "cache_key": cache_key(messages),
        }
    }
    assert messages[1].parts[0].content == original.reminder.render()
    second = runner.run(
        replace(original, history_json=first.messages_json, resident="changed"), None
    )
    saved = ModelMessagesTypeAdapter.validate_json(second.messages_json)
    assert len(saved) == 5
    assert saved[:3] == messages
    assert "changed" not in second.messages_json


def test_history_is_never_trimmed_by_its_byte_length(settings) -> None:
    original = request()
    history = [
        ModelRequest(
            parts=[UserPromptPart("旧记忆😀" * 20)],
            metadata={"huddol": {"block": "resident", "environment": "environment A"}},
        ),
        ModelResponse(parts=[TextPart("Old response")]),
    ]
    raw = ModelMessagesTypeAdapter.dump_json(history).decode()
    settings.set_settings("model", {**model_values("first"), "compaction_threshold": 1})
    outcome = PydanticModelRunner(settings, build_model=Builder()).run(
        replace(original, history_json=raw), None
    )
    assert outcome.error is None
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert without_cache_key(saved)[:2] == history
    assert saved[-2].parts[0].content == original.prompt


@pytest.mark.parametrize("initial", [None, "environment A"])
@pytest.mark.parametrize("fail", [False, True])
def test_durable_changes_are_persisted_once_and_survive_runner_restarts(
    settings, initial, fail
) -> None:
    environment = initial
    received = []

    def respond(messages, info):
        received.append(
            ModelMessagesTypeAdapter.validate_json(
                ModelMessagesTypeAdapter.dump_json(messages)
            )
        )
        if len(received) < 3:
            return ModelResponse(
                parts=[ToolCallPart("organization", {"action": "list_members"})]
            )
        if fail:
            raise RuntimeError("local failure after environment change")
        return ModelResponse(parts=[TextPart("Done")])

    class Tools:
        def list_members(self):
            nonlocal environment
            environment = "environment B"
            return []

    original = replace(request(), environment=lambda: environment)
    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    outcome = runner.run(original, Tools())
    assert (outcome.error is not None) == fail
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert saved[0].metadata == {
        "huddol": {
            "block": "resident",
            "agents_instructions": None,
            "environment": initial or "",
            "cache_key": cache_key(saved),
        }
    }
    durable = [
        message
        for message in saved
        if message.metadata
        and message.metadata.get("huddol", {}).get("block") == "durable"
    ]
    assert len(durable) == 1
    assert (
        durable[0].parts[0].content
        == "The execution environment changed.\nenvironment B"
    )
    assert durable[0].metadata == {
        "huddol": {"block": "durable", "environment": "environment B"}
    }
    assert [
        sum(
            isinstance(part, UserPromptPart)
            and part.content == durable[0].parts[0].content
            for message in messages
            for part in message.parts
        )
        for messages in received
    ] == [0, 1, 1]
    restarted = PydanticModelRunner(settings, build_model=Builder())
    continued = restarted.run(
        replace(original, history_json=outcome.messages_json), None
    )
    assert continued.error is None
    assert (
        ModelMessagesTypeAdapter.validate_json(continued.messages_json)[: len(saved)]
        == saved
    )
    environment = None
    unavailable = restarted.run(
        replace(original, history_json=continued.messages_json), None
    )
    assert unavailable.error is None
    assert unavailable.messages_json.count('"block":"durable"') == 1


def test_observability_is_reused_replaced_and_shutdown_per_turn(settings) -> None:
    builder = Builder()
    exporters = []

    def build(config):
        exporter = FakeObservability(config)
        exporters.append(exporter)
        return exporter

    runner = PydanticModelRunner(
        settings, build_model=builder, build_observability=build
    )
    assert runner.run(request(), None).error is None
    assert exporters == []
    tracing = {
        "enabled": True,
        "base_url": "https://langfuse.invalid",
        "public_key": "pk",
        "secret_key": "sk",
    }
    settings.set_settings("observability", tracing)
    for _ in range(2):
        assert runner.run(request(), None).error is None
    assert len(exporters) == 1 and exporters[0].calls == 2
    assert not exporters[0].stopped
    settings.set_settings("observability", {**tracing, "environment": "test"})
    assert runner.run(request(), None).error is None
    assert len(exporters) == 2 and exporters[0].stopped
    assert not exporters[1].stopped
    settings.set_settings("observability", {**tracing, "enabled": False})
    assert runner.run(request(), None).error is None
    assert exporters[1].stopped
    assert len(builder.calls) == 5


def test_observability_changes_during_a_turn_apply_to_the_next_turn(settings) -> None:
    exporters = []
    calls = 0

    def build(config):
        exporter = FakeObservability(config)
        exporters.append(exporter)
        return exporter

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("organization", {"action": "list_members"})]
            )
        assert len(exporters) == 1
        assert not exporters[0].stopped
        return ModelResponse(parts=[TextPart("Done")])

    class Tools:
        def list_members(self):
            settings.set_settings("observability", {"enabled": False})
            return []

    settings.set_settings(
        "observability",
        {
            "enabled": True,
            "base_url": "https://langfuse.invalid",
            "public_key": "pk",
            "secret_key": "sk",
        },
    )
    runner = PydanticModelRunner(
        settings,
        build_model=lambda config: FunctionModel(respond),
        build_observability=build,
    )
    assert runner.run(request(), Tools()).error is None
    assert exporters[0].calls == 1
    runner._build_model = Builder()
    settings.set_settings("model", model_values("second"))
    assert runner.run(request(), None).error is None
    assert exporters[0].stopped


def test_concurrent_turns_build_separate_models(settings) -> None:
    release = threading.Event()
    builder = Builder(delay=release)
    runner = PydanticModelRunner(settings, build_model=builder)
    seen = []
    threads = [
        threading.Thread(target=lambda: seen.append(runner.run(request(), None)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert builder.entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(builder.calls) == 4
    assert len(seen) == 4
    assert all(outcome.error is None for outcome in seen)


def test_a_failed_model_build_is_retried_on_the_next_turn(settings) -> None:
    attempts = []

    def flaky(config):
        attempts.append(config)
        if len(attempts) == 1:
            raise RuntimeError("provider rejected the configuration")
        return Builder()(config)

    runner = PydanticModelRunner(settings, build_model=flaky)
    assert "provider rejected" in runner.run(request(), None).error
    assert runner.run(request(), None).error is None
    assert len(attempts) == 2


@pytest.mark.parametrize("tokens", [0, 123])
def test_input_tokens_and_prompt_come_from_the_current_turn(settings, tokens) -> None:
    received = []

    def respond(messages, info):
        received.extend(messages)
        return ModelResponse(
            parts=[TextPart("Done")],
            usage=RequestUsage(input_tokens=tokens, output_tokens=1),
        )

    original = replace(
        request(),
        prompt="Save notes",
        reminder=None,
        resident="Resident\n\nReset notice",
    )
    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(original, None)
    assert outcome.error is None
    assert outcome.input_tokens == (tokens or None)
    assert json.loads(outcome.usage_json)["last_input_tokens"] == (tokens or None)
    assert received[-1].parts[-1].content == original.prompt
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    assert saved[0].parts[0].content.endswith("Reset notice")
    assert saved[1].parts[0].content == original.prompt


@pytest.mark.parametrize("last_tokens", [100, 0])
@pytest.mark.parametrize("fail", [False, True])
def test_last_input_tokens_excludes_old_history_and_is_not_a_turn_total(
    settings, last_tokens, fail
) -> None:
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("organization", {"action": "list_members"})],
                usage=RequestUsage(input_tokens=250, output_tokens=2),
            )
        if fail:
            raise ModelHTTPError(
                400, "fake", {"error": {"code": "context_length_exceeded"}}
            )
        return ModelResponse(
            parts=[TextPart("Done")],
            usage=RequestUsage(input_tokens=last_tokens, output_tokens=1),
        )

    class Tools:
        def list_members(self):
            return []

    history = ModelMessagesTypeAdapter.dump_json(
        [
            ModelRequest(parts=[UserPromptPart("Old prompt")]),
            ModelResponse(
                parts=[TextPart("Old response")], usage=RequestUsage(input_tokens=999)
            ),
        ]
    ).decode()
    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(replace(request(), history_json=history), Tools())
    assert calls == 2
    expected = 250 if fail or last_tokens == 0 else last_tokens
    assert outcome.input_tokens == expected
    usage = json.loads(outcome.usage_json)
    assert usage["last_input_tokens"] == expected
    assert usage["input_tokens"] == 250 + (0 if fail else last_tokens)
    assert usage["tool_calls"] == 1
    assert outcome.context_exceeded == fail
    assert (outcome.error is not None) == fail
    assert "tool-return" in outcome.messages_json
    assert "Old response" in outcome.messages_json


def test_turn_http_error_evidence_through_agent_run(
    settings, monkeypatch, tmp_path
) -> None:
    import httpx
    from openai import AsyncOpenAI

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    marker = "turn evidence refusal marker"
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            400, json={"error": {"message": marker, "type": "invalid_request_error"}}
        )

    def build(config: ModelConfig):
        client = AsyncOpenAI(
            api_key=config.api_key,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        return OpenAIChatModel(
            config.model,
            settings=thinking_settings(config.api_type, config.model, config.thinking),
            provider=OpenAIProvider(openai_client=client),
        )

    settings.set_settings(
        "model",
        {
            "version": 2,
            "providers": [
                {
                    "id": "p",
                    "name": "mock provider",
                    "api_type": "openai-chat",
                    "base_url": "https://mock.invalid/v1",
                    "api_key": "test-only-key",
                    "enabled": True,
                }
            ],
            "models": [
                {
                    "id": "m",
                    "provider_id": "p",
                    "name": "model",
                    "model": "gpt-5.2",
                    "enabled": True,
                }
            ],
            "default_model_id": "m",
            "default_thinking": "high",
            "agent_configs": {},
        },
    )
    outcome = PydanticModelRunner(settings, build_model=build).run(request(), None)
    assert marker in (outcome.error or "")
    assert "test-only-key" not in (outcome.error or "")
    assert len(requests) == 1
    assert requests[0]["model"] == "gpt-5.2"
    assert requests[0]["reasoning_effort"] == "high"
    evidence = {
        "request": requests[0],
        "error": outcome.error,
        "marker_preserved": marker in (outcome.error or ""),
    }
    output = tmp_path / "turn-error.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("prior", [False, True])
def test_overflow_before_any_response_keeps_partial_history_without_old_usage(
    settings, prior
) -> None:
    def respond(messages, info):
        raise ModelHTTPError(
            502, "fake", "Your input exceeds the context window of this model"
        )

    original = request()
    if prior:
        history = ModelMessagesTypeAdapter.dump_json(
            [
                ModelRequest(parts=[UserPromptPart("Old prompt")]),
                ModelResponse(
                    parts=[TextPart("Old response")],
                    usage=RequestUsage(input_tokens=999),
                ),
            ]
        ).decode()
        original = replace(original, history_json=history)
    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(original, None)
    assert outcome.context_exceeded
    assert "ModelHTTPError" in outcome.error
    assert outcome.input_tokens is None
    assert json.loads(outcome.usage_json)["last_input_tokens"] is None
    assert json.loads(outcome.usage_json)["input_tokens"] == 0
    assert original.prompt in [
        part.content
        for message in ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert ("Old response" in outcome.messages_json) == prior


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "context_length_exceeded", "message": "rejected"}},
        {"code": "invalid_request_error", "type": "context_length_exceeded"},
        {
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 210000 tokens",
            },
        },
        {
            "error": {
                "message": "The input token count exceeds the maximum number of tokens allowed",
                "status": "INVALID_ARGUMENT",
            }
        },
        "Your input exceeds the context window of this model",
        "Maximum context length reached",
        "The context window was exceeded",
        "prompt is too long",
        "Maximum context length exceeded",
    ],
)
@pytest.mark.parametrize("status", [400, 413, 502])
def test_context_exceeded_recognizes_provider_errors(body, status) -> None:
    assert is_context_exceeded(ModelHTTPError(status, "fake", body))


@pytest.mark.parametrize(
    "error",
    [
        ModelHTTPError(502, "fake", "upstream failure"),
        ModelHTTPError(413, "fake", "Request too large"),
        ModelHTTPError(413, "fake", "REQUEST TOO LARGE"),
        ModelHTTPError(400, "fake", "too many tokens"),
        ModelHTTPError(400, "fake", "exceeds the maximum number of tokens"),
        ModelHTTPError(400, "fake", "output token budget exceeded"),
        ModelHTTPError(
            502,
            "fake",
            "The context window limit is 128k; output token budget exceeded",
        ),
        ModelHTTPError(
            502,
            "fake",
            "The context window limit is 128k, output token budget exceeded",
        ),
        ModelHTTPError(
            502,
            "fake",
            "The context window limit is 128k and output token budget exceeded",
        ),
        ModelHTTPError(
            502,
            "fake",
            "The context window limit is 128k, but the output token budget was exceeded",
        ),
        ModelHTTPError(401, "fake", "invalid API key"),
        ModelHTTPError(429, "fake", "rate limit exceeded"),
        ModelHTTPError(400, "fake", {"metadata": "context_length_exceeded"}),
        ModelHTTPError(400, "fake", {"error": {"message": "context window"}}),
        ModelHTTPError(400, "fake", None),
        RuntimeError("context_length_exceeded"),
    ],
)
def test_context_exceeded_does_not_match_other_failures(error) -> None:
    assert not is_context_exceeded(error)


def test_streaming_appends_ephemeral_only_to_the_outgoing_copy() -> None:
    received = []

    async def stream(messages, info):
        received.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
        yield "Done"

    original = [ModelRequest(parts=[UserPromptPart("Reminder")])]
    model = LiveModel(
        lambda: FunctionModel(stream_function=stream), lambda: "EPHEMERAL", "key"
    )

    async def run():
        async with model.request_stream(
            original, None, ModelRequestParameters()
        ) as response:
            async for _ in response:
                pass
            assert response.get().parts[0].content == "Done"

    asyncio.run(run())
    assert "EPHEMERAL" in received[0]
    assert len(original[0].parts) == 1


def test_the_cache_key_is_created_with_the_window_and_kept_by_later_turns(
    settings,
) -> None:
    runner = PydanticModelRunner(settings, build_model=Builder())
    first = runner.run(request(), None)
    key = cache_key(ModelMessagesTypeAdapter.validate_json(first.messages_json))
    restarted = PydanticModelRunner(settings, build_model=Builder())
    second = restarted.run(replace(request(), history_json=first.messages_json), None)
    saved = ModelMessagesTypeAdapter.validate_json(second.messages_json)
    assert cache_key(saved) == key
    assert second.messages_json.count('"cache_key"') == 1
    fresh = restarted.run(request(), None)
    assert cache_key(ModelMessagesTypeAdapter.validate_json(fresh.messages_json)) != key


def test_a_window_without_a_cache_key_receives_one_on_its_next_turn(
    settings,
) -> None:
    history = [
        ModelRequest(parts=[UserPromptPart("Old prompt")]),
        ModelResponse(parts=[TextPart("Old response")]),
    ]
    raw = ModelMessagesTypeAdapter.dump_json(history).decode()
    runner = PydanticModelRunner(settings, build_model=Builder())
    outcome = runner.run(replace(request(), history_json=raw), None)
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    key = cache_key(saved)
    assert saved[0].metadata == {"huddol": {"cache_key": key}}
    assert without_cache_key(saved)[:2] == history
    again = runner.run(replace(request(), history_json=outcome.messages_json), None)
    assert cache_key(ModelMessagesTypeAdapter.validate_json(again.messages_json)) == key


@pytest.mark.parametrize("given", [None, {"temperature": 0.5}])
def test_the_cache_key_reaches_each_provider_in_its_own_setting(given) -> None:
    seen: dict[str, Any] = {}

    def recording(base: type) -> type:
        class Recording(base):
            async def request(self, messages, model_settings, model_request_parameters):
                seen[base.__name__] = model_settings
                return ModelResponse(parts=[TextPart("ok")])

        return Recording

    provider = OpenAIProvider(base_url="https://example.invalid/v1", api_key="unused")
    wrapped = [
        recording(OpenAIChatModel)("chat", provider=provider),
        recording(OpenAIResponsesModel)("responses", provider=provider),
        recording(AnthropicModel)(
            "claude", provider=AnthropicProvider(api_key="unused")
        ),
    ]
    for model in wrapped:
        live = LiveModel(lambda model=model: model, lambda: "", "window-key")
        asyncio.run(
            live.request(
                [ModelRequest(parts=[UserPromptPart("hi")])],
                given,
                ModelRequestParameters(),
            )
        )
    assert seen == {
        "OpenAIChatModel": {**(given or {}), "openai_prompt_cache_key": "window-key"},
        "OpenAIResponsesModel": {
            **(given or {}),
            "openai_prompt_cache_key": "window-key",
        },
        "AnthropicModel": {**(given or {}), "anthropic_cache": True},
    }
    untouched = []
    plain = FunctionModel(
        lambda messages, info: (
            untouched.append(info.model_settings),
            ModelResponse(parts=[TextPart("ok")]),
        )[1]
    )
    asyncio.run(
        LiveModel(lambda: plain, lambda: "", "window-key").request(
            [ModelRequest(parts=[UserPromptPart("hi")])],
            given,
            ModelRequestParameters(),
        )
    )
    assert untouched == [given]


@pytest.mark.parametrize("start", ["empty", "unanswered"])
def test_every_call_extends_what_the_previous_call_sent(settings, start) -> None:
    sent = []
    plan = iter(["tool", "text", "tool", "tool", "text", "text"])

    def respond(messages, info):
        sent.append(flattened(messages))
        if next(plan) == "tool":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "organization",
                        {"action": "list_members"},
                        tool_call_id=f"c{len(sent)}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(f"done {len(sent)}")])

    class Tools:
        def list_members(self):
            return [{"id": len(sent), "title": "Work"}]

    original = request()
    if start == "unanswered":
        history = [
            ModelRequest(parts=[UserPromptPart("Old prompt")]),
            ModelResponse(
                parts=[
                    TextPart("Checking"),
                    ToolCallPart("library", {"action": "list"}, tool_call_id="old"),
                ]
            ),
        ]
        original = replace(
            original,
            history_json=ModelMessagesTypeAdapter.dump_json(history).decode(),
        )
    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    for _ in range(3):
        outcome = runner.run(original, Tools())
        assert outcome.error is None
        original = replace(original, history_json=outcome.messages_json)
    assert len(sent) == 6
    for previous, current in pairwise(sent):
        assert current[: len(previous)] == previous
        assert len(current) > len(previous)


@pytest.mark.parametrize(
    "name,params,expected",
    [
        (
            "run",
            {"argv": ["rg", "needle"], "cwd": "/data/library", "timeout": 7},
            (["rg", "needle"], "/data/library", 7),
        ),
        (
            "edit",
            {
                "path": "workspace/MEMORY.md",
                "old_text": "before",
                "new_text": "",
                "replace_all": True,
            },
            ("workspace/MEMORY.md", "before", "", True, False),
        ),
        (
            "edit",
            {
                "path": "/data/script.py",
                "old_text": "",
                "new_text": "print('ready')\\n",
                "create": True,
            },
            ("/data/script.py", "", "print('ready')\\n", False, True),
        ),
    ],
)
def test_model_file_tools_forward_arguments_without_tree_tools(
    settings, name, params, expected
) -> None:
    calls = []

    class Tools:
        def run(self, *args):
            calls.append(args)
            return {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False}

        def edit(self, *args):
            calls.append(args)
            return {"path": args[0], "diff": "", "replacements": 1}

    def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        assert {"run", "edit"} <= names
        assert names.isdisjoint({"workspace", "library"})
        edit_schema = next(tool for tool in info.function_tools if tool.name == "edit")
        assert "create=true" in edit_schema.description
        assert (
            edit_schema.parameters_json_schema["properties"]["create"]["type"]
            == "boolean"
        )
        assert "create" not in edit_schema.parameters_json_schema["required"]
        if not calls:
            return ModelResponse(parts=[ToolCallPart(name, params)])
        return ModelResponse(parts=[TextPart("Done")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    ).run(request(), Tools())
    assert outcome.error is None
    assert calls == [expected]


def test_view_image_returns_native_binary_content_and_preserves_history(settings):
    from io import BytesIO

    from PIL import Image

    source = BytesIO()
    Image.new("RGB", (2, 3), (80, 90, 100)).save(source, format="PNG")
    data = source.getvalue()
    path = "/native/picture.png"
    persisted = []
    received = []

    class Tools:
        def view_image(self, value):
            assert value == path
            return ViewedImage(path, ImageData(data, "image/png", 2, 3))

    def respond(messages, info):
        received.append(messages)
        if len(received) == 1:
            view = next(
                tool for tool in info.function_tools if tool.name == "view_image"
            )
            assert view.parameters_json_schema["required"] == ["path"]
            assert "sent to the current model" in view.description
            assert "current Turn's model history" in view.description
            return ModelResponse(
                parts=[ToolCallPart("view_image", {"path": path}, "image-call")]
            )
        result = next(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "view_image"
        )
        description, image = result.content
        assert description == {
            "path": path,
            "size": len(data),
            "media_type": "image/png",
            "width": 2,
            "height": 3,
        }
        assert isinstance(image, BinaryContent)
        assert image.data == data
        return ModelResponse(parts=[TextPart("Done")])

    outcome = PydanticModelRunner(
        settings,
        build_model=lambda config: FunctionModel(respond),
    ).run(replace(request(), persist=persisted.append), Tools())
    assert outcome.error is None
    assert len(received) == 2
    saved = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)
    returned = next(
        part
        for message in saved
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "view_image"
    )
    assert returned.content[0]["path"] == path
    assert isinstance(returned.content[1], BinaryContent)
    assert returned.content[1].data == data
    assert persisted


@pytest.mark.parametrize("kind", ["sync", "async", "awaitable"])
def test_search_failures_continue_past_retry_budget(
    settings, monkeypatch, caplog, kind
):
    from huddol.adapters.model import runner as module

    executions = []
    received = []
    secret = "fake-secret-do-not-disclose"

    def fail(query: str) -> str:
        executions.append(query)
        raise RuntimeError(secret)

    async def async_fail(query: str) -> str:
        await asyncio.sleep(0)
        return fail(query)

    def awaitable_fail(query: str):
        return async_fail(query)

    function = {"sync": fail, "async": async_fail, "awaitable": awaitable_fail}[kind]
    monkeypatch.setattr(module, "duckduckgo_search_tool", lambda **kw: Tool(function))

    class Tools:
        def list_members(self):
            return ["alternate tool worked"]

    def respond(messages, info):
        received.append(messages)
        schema = next(t for t in info.function_tools if t.name == "web_search")
        assert schema.parameters_json_schema["required"] == ["query"]
        assert schema.parameters_json_schema["properties"]["query"]["type"] == "string"
        if len(received) <= 4:
            if len(received) > 1:
                part = messages[-1].parts[0]
                assert isinstance(part, ToolReturnPart)
                assert part.outcome == "failed"
                assert "tool_execution_failed: RuntimeError" in part.content
                assert secret not in part.content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "web_search",
                        {"query": "safe"},
                        tool_call_id=f"search-{len(received)}",
                    )
                ]
            )
        if len(received) == 5:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "organization",
                        {"action": "list_members"},
                        tool_call_id="alternate",
                    )
                ]
            )
        assert messages[-1].parts[0].content == ["alternate tool worked"]
        return ModelResponse(parts=[TextPart("Recovered")])

    runner = PydanticModelRunner(settings, build_model=lambda _: FunctionModel(respond))
    outcome = runner.run(request(), Tools())
    assert outcome.error is None
    assert len(received) == 6
    assert executions == ["safe"] * 4
    assert secret not in outcome.messages_json + caplog.text
    assert "test_runner.py:" in caplog.text
    assert "type=RuntimeError" in caplog.text
    assert "agent=2 turn=1 tool=web_search call=search-4" in caplog.text
    assert module._tool_call.get(None) is None


@pytest.mark.parametrize(
    "code",
    ["timeout", "execution_timeout", "execution_unavailable", "execution_protocol"],
)
def test_execution_domain_failures_do_not_replay_writes(settings, code):
    writes = []
    received = []

    class Tools:
        def send_message(self, discussion_id, body):
            writes.append(body)
            raise DomainError(code, "private execution diagnostic")

        def list_members(self):
            return []

    def respond(messages, info):
        received.append(messages)
        if len(received) <= 4:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discussion",
                        {
                            "action": "send",
                            "discussion_id": 1,
                            "body": str(len(received)),
                        },
                        tool_call_id=f"write-{len(received)}",
                    )
                ]
            )
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        assert len(returns) == 4
        assert all(p.outcome == "failed" and code in p.content for p in returns)
        assert all("Side effects may have occurred" in p.content for p in returns)
        return ModelResponse(parts=[TextPart("Stopped writing")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda _: FunctionModel(respond)
    ).run(request(), Tools())
    assert outcome.error is None
    assert writes == ["1", "2", "3", "4"]
    assert "private execution diagnostic" not in outcome.messages_json


@pytest.mark.parametrize("parallel", [False, True])
def test_failed_and_successful_batch_calls_pair_and_keep_context(
    settings, monkeypatch, caplog, parallel
):
    from huddol.adapters.model import runner as module

    entered = []
    received = []

    async def search(query: str):
        entered.append(query)
        await asyncio.sleep(0.01)
        assert module._tool_call.get()[2].tool_call_id == query
        if query == "bad":
            raise OSError("private")
        return query

    monkeypatch.setattr(module, "duckduckgo_search_tool", lambda **kw: Tool(search))

    class Tools:
        def read_history(self, sequence):
            call_id = "bad" if sequence == 1 else "good"
            entered.append(call_id)
            assert module._tool_call.get()[2].tool_call_id == call_id
            if sequence == 1:
                raise OSError("private")
            return call_id

    def respond(messages, info):
        received.append(messages)
        if len(received) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "web_search" if parallel else "history",
                        {"query": name}
                        if parallel
                        else {"action": "read", "sequence": i},
                        tool_call_id=name,
                    )
                    for i, name in [(1, "bad"), (2, "good")]
                ]
            )
        parts = messages[-1].parts
        assert len(parts) == 2
        paired = {p.tool_call_id: p for p in parts}
        assert paired["bad"].outcome == "failed"
        assert paired["good"].content == "good"
        assert paired["good"].outcome == "success"
        return ModelResponse(parts=[TextPart("Both accounted for")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda _: FunctionModel(respond)
    ).run(request(), Tools())
    assert outcome.error is None
    assert sorted(entered) == ["bad", "good"]
    assert "call=bad" in caplog.text
    assert "call=good" not in caplog.text
    assert module._tool_call.get(None) is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        ModelRetry("fix input"),
        ToolFailed("known failure"),
        SkipToolExecution("skipped"),
        CallDeferred(),
        ApprovalRequired(),
        asyncio.CancelledError(),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_tool_boundary_preserves_control_flow(error, asynchronous):
    from huddol.adapters.model.runner import _tool_boundary

    def sync_function(value: int) -> int:
        raise error

    async def async_function(value: int) -> int:
        raise error

    function = async_function if asynchronous else sync_function
    wrapped = _tool_boundary(function)
    assert inspect.signature(wrapped) == inspect.signature(function)
    assert inspect.iscoroutinefunction(wrapped) == asynchronous
    with pytest.raises(type(error)) as raised:
        if asynchronous:
            asyncio.run(wrapped(1))
        else:
            wrapped(1)
    assert raised.value is error


def test_tool_validation_still_returns_correctable_feedback(settings):
    received = []
    executions = []

    class Tools:
        def run(self, argv, cwd, timeout):
            executions.append(argv)
            return {"exit_code": 0}

    def respond(messages, info):
        received.append(messages)
        if len(received) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("run", {"argv": "not a list"}, tool_call_id="invalid")
                ]
            )
        if len(received) == 2:
            assert isinstance(messages[-1].parts[0], RetryPromptPart)
            return ModelResponse(
                parts=[ToolCallPart("run", {"argv": ["true"]}, tool_call_id="valid")]
            )
        assert messages[-1].parts[0].content == {"exit_code": 0}
        return ModelResponse(parts=[TextPart("Fixed")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda _: FunctionModel(respond)
    ).run(request(), Tools())
    assert outcome.error is None
    assert executions == [["true"]]


@pytest.mark.parametrize("stage", ["dispatch", "serialize", "model"])
def test_non_tool_faults_are_not_tool_failures(settings, monkeypatch, stage):
    from pydantic_ai.toolsets.function import FunctionToolset

    from huddol.adapters.model import runner as module

    executions = []

    class Unserializable:
        pass

    class Tools:
        def list_members(self):
            executions.append(1)
            return Unserializable()

    async def broken_dispatch(*args, **kwargs):
        raise RuntimeError("SDK dispatch fault")

    if stage == "dispatch":
        monkeypatch.setattr(FunctionToolset, "call_tool", broken_dispatch)

    def respond(messages, info):
        if stage == "model":
            raise RuntimeError("model fault")
        if executions:
            ModelMessagesTypeAdapter.dump_json(messages)
            return ModelResponse(parts=[TextPart("Done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "organization", {"action": "list_members"}, tool_call_id="call"
                )
            ]
        )

    runner = PydanticModelRunner(settings, build_model=lambda _: FunctionModel(respond))
    if stage == "serialize":
        from pydantic_core import PydanticSerializationError

        with pytest.raises(HistoryPersistenceError) as caught:
            runner.run(request(), Tools())
        assert isinstance(caught.value.__cause__, PydanticSerializationError)
    else:
        outcome = runner.run(request(), Tools())
        assert outcome.error is not None
        assert "tool_execution_failed" not in outcome.messages_json
    assert module._tool_call.get(None) is None


@pytest.mark.parametrize(
    "name,args,method",
    [
        ("organization", {"action": "list_members"}, "list_members"),
        ("discussion", {"action": "list"}, "list_discussions"),
        ("run", {"argv": ["true"]}, "run"),
        ("edit", {"path": "/fake", "old_text": "old", "new_text": "new"}, "edit"),
        ("history", {"action": "read", "sequence": 1}, "read_history"),
        (
            "history",
            {"action": "read", "sequence": 1, "offset": 2048},
            "read_history",
        ),
    ],
)
def test_all_builtin_registrations_share_execution_boundary(
    settings, name, args, method
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError("unavailable")

    tools = type("Tools", (), {method: fail})()

    def respond(messages, info):
        if not calls:
            return ModelResponse(
                parts=[ToolCallPart(name, args, tool_call_id="builtin")]
            )
        part = messages[-1].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.outcome == "failed"
        return ModelResponse(parts=[TextPart("Continue")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda _: FunctionModel(respond)
    ).run(request(), tools)
    assert outcome.error is None
    assert calls == [1]


def test_real_search_registration_handles_client_error(settings, monkeypatch):
    from pydantic_ai.common_tools.duckduckgo import DDGS

    calls = []

    def fail(self, *args, **kwargs):
        calls.append(1)
        raise OSError("fake search outage")

    monkeypatch.setattr(DDGS, "text", fail)

    def respond(messages, info):
        if not calls:
            return ModelResponse(
                parts=[
                    ToolCallPart("web_search", {"query": "test"}, tool_call_id="search")
                ]
            )
        assert messages[-1].parts[0].outcome == "failed"
        return ModelResponse(parts=[TextPart("No search available")])

    outcome = PydanticModelRunner(
        settings, build_model=lambda _: FunctionModel(respond)
    ).run(request(), None)
    assert outcome.error is None
    assert calls == [1]


def test_tool_context_does_not_leak_between_concurrent_turns(settings, caplog):
    from concurrent.futures import ThreadPoolExecutor

    from huddol.adapters.model import runner as module

    barrier = threading.Barrier(2)
    seen = []

    class Tools:
        def list_members(self):
            agent_id, sequence, call = module._tool_call.get()
            barrier.wait(timeout=5)
            assert module._tool_call.get() == (agent_id, sequence, call)
            seen.append((agent_id, sequence))
            raise RuntimeError("fake concurrent failure")

    def respond(messages, info):
        if any(
            isinstance(p, ToolReturnPart)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        ):
            return ModelResponse(parts=[TextPart("Done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "organization",
                    {"action": "list_members"},
                    tool_call_id="same-id",
                )
            ]
        )

    runner = PydanticModelRunner(settings, build_model=lambda _: FunctionModel(respond))

    def turn(i):
        outcome = runner.run(replace(request(), agent_id=i, sequence=i + 10), Tools())
        assert module._tool_call.get(None) is None
        return outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(turn, [2, 3]))
    assert all(o.error is None for o in outcomes)
    assert sorted(seen) == [(2, 12), (3, 13)]
    assert "agent=2 turn=12" in caplog.text
    assert "agent=3 turn=13" in caplog.text


def test_runner_does_not_convert_tool_cancellation_to_failure(settings):
    from huddol.adapters.model import runner as module

    class Tools:
        def list_members(self):
            raise asyncio.CancelledError()

    def respond(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("organization", {"action": "list_members"})]
        )

    runner = PydanticModelRunner(settings, build_model=lambda _: FunctionModel(respond))
    with pytest.raises(asyncio.CancelledError):
        runner.run(request(), Tools())
    assert module._tool_call.get(None) is None


def test_initial_snapshot_persistence_failure_prevents_model_creation(settings):
    built = []

    def fail(raw):
        snapshot = ModelMessagesTypeAdapter.validate_json(raw)[0]
        assert snapshot.metadata["huddol"]["agents_instructions"] == "raw\n "
        raise OSError("snapshot disk unavailable")

    runner = PydanticModelRunner(
        settings, build_model=lambda config: built.append(config)
    )
    with pytest.raises(HistoryPersistenceError) as caught:
        runner.run(replace(request(), agents_instructions="raw\n ", persist=fail), None)
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "snapshot disk unavailable"
    assert built == []


def test_failed_model_creation_keeps_snapshot_for_restart(settings):
    persisted = []

    def fail(config):
        raise RuntimeError("build failed")

    outcome = PydanticModelRunner(settings, build_model=fail).run(
        replace(request(), agents_instructions="original\n ", persist=persisted.append),
        None,
    )
    assert outcome.error == "RuntimeError: build failed"
    assert outcome.messages_json == persisted[0]
    seen = []

    def respond(messages, info):
        seen.append(info.model_request_parameters.instruction_parts[0].content)
        return ModelResponse(parts=[TextPart("done")])

    restarted = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    assert (
        restarted.run(
            replace(
                request(),
                history_json=outcome.messages_json,
                agents_instructions="changed",
            ),
            None,
        ).error
        is None
    )
    assert seen == [SYSTEM_PROMPT + "\n\noriginal\n "]


@pytest.mark.parametrize(
    "kind,switch",
    [
        ("openai-chat", False),
        ("openai-responses", False),
        ("anthropic", False),
        ("google", False),
        ("openai-chat", True),
        ("openai-responses", True),
    ],
)
def test_provider_wire_preserves_raw_snapshot_across_tools_turns_and_restart(
    settings, monkeypatch, kind, switch
):
    import httpx
    import httpx2
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    wire = []
    wire_kinds = []
    raw = "  全局😀\n\n\t member ${literal}\t \n"
    settings.set_settings("model", {**model_values("gpt-4o"), "api_type": kind})

    def respond(req):
        kind = {
            "/v1/chat/completions": "openai-chat",
            "/v1/responses": "openai-responses",
            "/v1/messages": "anthropic",
        }.get(req.url.path, "google")
        wire_kinds.append(kind)
        body = json.loads(req.content)
        wire.append(body)
        tool = len(wire) % 2 == 1
        if kind == "openai-chat":
            msg = {"role": "assistant", "content": "done"}
            if tool:
                msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {
                                "name": "organization",
                                "arguments": '{"action":"list_members"}',
                            },
                        }
                    ],
                }
            return httpx.Response(
                200,
                json={
                    "id": "chat",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": msg,
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                },
            )
        if kind == "openai-responses":
            output = (
                [
                    {
                        "id": "fc",
                        "type": "function_call",
                        "call_id": "call",
                        "name": "organization",
                        "arguments": '{"action":"list_members"}',
                        "status": "completed",
                    }
                ]
                if tool
                else [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "done", "annotations": []}
                        ],
                    }
                ]
            )
            return httpx.Response(
                200,
                json={
                    "id": f"resp{len(wire)}",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-4o",
                    "status": "completed",
                    "output": output,
                    "parallel_tool_calls": True,
                    "tools": [],
                    "tool_choice": "auto",
                },
            )
        if kind == "anthropic":
            content = (
                [
                    {
                        "type": "tool_use",
                        "id": "call",
                        "name": "organization",
                        "input": {"action": "list_members"},
                    }
                ]
                if tool
                else [{"type": "text", "text": "done"}]
            )
            return httpx2.Response(
                200,
                json={
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": content,
                    "stop_reason": "tool_use" if tool else "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        parts = (
            [
                {
                    "functionCall": {
                        "name": "organization",
                        "args": {"action": "list_members"},
                    }
                }
            ]
            if tool
            else [{"text": "done"}]
        )
        return httpx2.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        )

    def build(config):
        kind = config.api_type
        if kind == "anthropic":
            client = AsyncAnthropic(
                api_key="unused",
                max_retries=0,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
            )
            return AnthropicModel(
                "claude-sonnet-4-5", provider=AnthropicProvider(anthropic_client=client)
            )
        if kind == "google":
            return GoogleModel(
                "gemini-2.5-flash",
                provider=GoogleProvider(
                    api_key="unused",
                    http_client=httpx2.AsyncClient(
                        transport=httpx2.MockTransport(respond)
                    ),
                ),
            )
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        provider = OpenAIProvider(
            openai_client=AsyncOpenAI(api_key="unused", max_retries=0, http_client=http)
        )
        return (
            OpenAIResponsesModel if kind == "openai-responses" else OpenAIChatModel
        )("gpt-4o", provider=provider)

    class Tools:
        def list_members(self):
            if switch:
                previous = settings.get_settings("model")
                settings.set_settings(
                    "model",
                    {
                        **previous,
                        "api_type": "openai-responses"
                        if previous["api_type"] == "openai-chat"
                        else "openai-chat",
                    },
                )
            return []

    history = "[]"
    for index in range(3):
        if index == 2:
            history = "[]"
        runner = PydanticModelRunner(settings, build_model=build)
        outcome = runner.run(
            replace(
                request(),
                history_json=history,
                agents_instructions=raw if index == 0 else "new\n ",
                ephemeral=lambda: "EPHEMERAL",
            ),
            Tools(),
        )
        assert outcome.error is None
        history = outcome.messages_json
        assert not any(
            isinstance(part, SystemPromptPart)
            for msg in ModelMessagesTypeAdapter.validate_json(history)
            for part in msg.parts
        )
        assert "EPHEMERAL" not in history
    assert len(wire) == 6
    for index, body in enumerate(wire):
        kind = wire_kinds[index]
        expected = SYSTEM_PROMPT + "\n\n" + (raw if index < 4 else "new\n ")
        if kind == "openai-chat":
            system = [
                m["content"]
                for m in body["messages"]
                if m["role"] in ("system", "developer")
            ]
        elif kind == "openai-responses":
            assert "instructions" not in body
            system = [
                m["content"]
                for m in body["input"]
                if m.get("role") in ("system", "developer")
            ]
        elif kind == "anthropic":
            system = [part["text"] for part in body["system"]]
        else:
            system = [part["text"] for part in body["systemInstruction"]["parts"]]
        if kind.startswith("openai"):
            assert (
                body["prompt_cache_key"] == wire[index - index % 4]["prompt_cache_key"]
            )
        assert system == [expected]
        assert system[0].encode("utf-8") == expected.encode("utf-8")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("snapshot", [None, "raw\t \n"])
def test_responses_conversion_uses_copies_and_preserves_ephemeral_and_settings(
    streaming, snapshot
):
    from contextlib import asynccontextmanager
    from copy import deepcopy

    from pydantic_ai.messages import InstructionPart

    seen = []

    class Recording(OpenAIResponsesModel):
        async def request(self, messages, model_settings, model_request_parameters):
            seen.append((messages, model_settings, model_request_parameters))
            return ModelResponse(parts=[TextPart("ok")])

        @asynccontextmanager
        async def request_stream(
            self, messages, model_settings, model_request_parameters, run_context=None
        ):
            seen.append((messages, model_settings, model_request_parameters))
            yield None

    provider = OpenAIProvider(api_key="unused", base_url="https://example.invalid")
    wrapped = Recording("gpt-4o", provider=provider)
    messages = [
        ModelRequest(
            parts=[UserPromptPart("resident")], instructions="old history instructions"
        ),
        ModelRequest(
            parts=[UserPromptPart("reminder")], instructions="current instructions"
        ),
    ]
    parameters = ModelRequestParameters(
        instruction_parts=[InstructionPart(content="current instructions")]
    )
    given = {"temperature": 0.5}
    original = deepcopy((messages, parameters, given))
    live = LiveModel(lambda: wrapped, lambda: "EPHEMERAL", "key", snapshot)

    async def call():
        for _ in range(2):
            if streaming:
                async with live.request_stream(messages, given, parameters):
                    pass
            else:
                await live.request(messages, given, parameters)

    asyncio.run(call())
    assert (messages, parameters, given) == original
    for outgoing, sent_settings, sent_parameters in seen:
        assert outgoing[-1].parts[-1].content == "EPHEMERAL"
        assert sent_settings == {"temperature": 0.5, "openai_prompt_cache_key": "key"}
        if snapshot is None:
            assert sent_parameters is parameters
            assert len(outgoing) == len(messages)
            assert outgoing[0] is messages[0]
        else:
            assert len(outgoing) == len(messages) + 1
            assert len(outgoing[0].parts) == 1
            assert isinstance(outgoing[0].parts[0], SystemPromptPart)
            assert outgoing[0].parts[0].content == SYSTEM_PROMPT + "\n\n" + snapshot
            assert sent_parameters.instruction_parts == []
            assert all(msg.instructions is None for msg in outgoing)


@pytest.mark.parametrize(
    "metadata",
    [{"block": "resident"}, {"block": "resident", "agents_instructions": None}],
)
def test_old_or_null_window_snapshot_never_uses_new_turn_value(settings, metadata):
    seen = []
    history = [
        ModelRequest(parts=[UserPromptPart("resident")], metadata={"huddol": metadata})
    ]

    def respond(messages, info):
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    outcome = runner.run(
        replace(
            request(),
            history_json=ModelMessagesTypeAdapter.dump_json(history).decode(),
            agents_instructions="must not leak",
        ),
        None,
    )
    assert outcome.error is None
    assert seen == [SYSTEM_PROMPT]


def test_first_model_request_failure_preserves_snapshot(settings):
    persisted = []

    def fail(messages, info):
        assert persisted
        raise RuntimeError("first request failed")

    outcome = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(fail)
    ).run(
        replace(request(), agents_instructions="raw\n ", persist=persisted.append), None
    )
    assert outcome.error == "RuntimeError: first request failed"
    resident = ModelMessagesTypeAdapter.validate_json(outcome.messages_json)[0]
    assert resident.metadata["huddol"]["agents_instructions"] == "raw\n "
    assert resident == ModelMessagesTypeAdapter.validate_json(persisted[0])[0]
