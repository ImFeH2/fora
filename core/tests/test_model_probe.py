from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fora.adapters.model import probe
from fora.adapters.model.config import ModelConfig, thinking_settings
from fora.core.errors import DomainError

STORED = {
    "api_type": "openai-chat",
    "base_url": "https://stored.invalid/v1",
    "api_key": "STORED-KEY",
    "model": "stored-model",
}


async def names(config: ModelConfig) -> list[str]:
    del config
    return ["models/gemini-b", "gemini-a", "gemini-a", "", "models/gemini-a"]


def test_listing_sorts_deduplicates_and_strips_the_google_prefix() -> None:
    result = probe.list_models(
        {"api_type": "google", "base_url": "https://g.invalid", "api_key": "k"},
        None,
        fetch=names,
    )
    assert result == {"models": ["gemini-a", "gemini-b"]}


@pytest.mark.parametrize("supplied", [None, "", "   "])
def test_listing_falls_back_to_the_stored_key(supplied: str | None) -> None:
    seen: list[ModelConfig] = []

    async def record(config: ModelConfig) -> list[str]:
        seen.append(config)
        return ["m"]

    values: dict[str, Any] = {"api_type": "anthropic", "base_url": "https://a.invalid"}
    if supplied is not None:
        values["api_key"] = supplied
    assert probe.list_models(values, STORED, fetch=record) == {"models": ["m"]}
    assert seen[0].api_key == "STORED-KEY"
    assert seen[0].api_type == "anthropic"
    assert seen[0].base_url == "https://a.invalid"


def test_listing_without_any_key_names_the_problem() -> None:
    with pytest.raises(DomainError) as failed:
        probe.list_models(
            {"api_type": "openai-chat", "base_url": "https://a.invalid"},
            {},
            fetch=names,
        )
    assert failed.value.code == "model_key_missing"


@pytest.mark.parametrize("api_type", [None, "", "azure", "OpenAI"])
def test_listing_rejects_unknown_api_types(api_type: object) -> None:
    with pytest.raises(DomainError) as failed:
        probe.list_models(
            {"api_type": api_type, "base_url": "https://a.invalid", "api_key": "k"},
            None,
            fetch=names,
        )
    assert failed.value.code == "invalid_api_type"


def test_listing_requires_a_base_url() -> None:
    with pytest.raises(ValueError):
        probe.list_models(
            {"api_type": "openai-chat", "api_key": "k"}, None, fetch=names
        )


def test_listing_failures_never_echo_the_key() -> None:
    async def explode(config: ModelConfig) -> list[str]:
        raise RuntimeError(
            f"401 from https://a.invalid/models?key={config.api_key}"
            f" with header Bearer {config.api_key}"
        )

    with pytest.raises(DomainError) as failed:
        probe.list_models(
            {
                "api_type": "openai-chat",
                "base_url": "https://a.invalid",
                "api_key": "TOP",
            },
            None,
            fetch=explode,
        )
    assert failed.value.code == "model_list_failed"
    message = str(failed.value)
    assert "TOP" not in message
    assert message == (
        "RuntimeError: 401 from https://a.invalid/models?key=*** with header Bearer ***"
    )


def test_listing_is_bounded_by_a_timeout(monkeypatch) -> None:
    monkeypatch.setattr(probe, "LIST_TIMEOUT", 0.05)

    async def slow(config: ModelConfig) -> list[str]:
        await asyncio.sleep(5)
        return []

    with pytest.raises(DomainError) as failed:
        probe.list_models(
            {
                "api_type": "openai-chat",
                "base_url": "https://a.invalid",
                "api_key": "k",
            },
            None,
            fetch=slow,
        )
    assert failed.value.code == "model_list_failed"
    assert str(failed.value) == "No response within 0.05 s"


def test_the_default_lister_covers_every_api_type() -> None:
    source = probe.fetch_model_names.__code__.co_consts
    del source
    assert probe.api_type_of("openai-responses") == "openai-responses"
    for api_type in ("openai-chat", "openai-responses", "anthropic", "google"):
        assert probe.api_type_of(api_type) == api_type


def function_model(reply: str | Exception):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        if isinstance(reply, Exception):
            raise reply
        assert messages[-1].parts[-1].content == probe.TEST_PROMPT
        return ModelResponse(parts=[TextPart(reply)])

    return FunctionModel(respond)


def test_a_model_test_reports_success_latency_and_reply(monkeypatch) -> None:
    from pydantic_ai import models

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    built: list[ModelConfig] = []

    def build(config: ModelConfig):
        built.append(config)
        return function_model(" OK \n")

    result = probe.try_model(
        {
            "api_type": "openai-responses",
            "base_url": "https://a.invalid/v1",
            "api_key": "k",
            "model": "candidate",
        },
        STORED,
        build=build,
    )
    assert result["ok"] is True
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0
    assert result["reply"] == "OK"
    assert built[0] == ModelConfig(
        "openai-responses", "https://a.invalid/v1", "k", "candidate"
    )


def test_a_model_test_uses_the_stored_key_when_none_is_typed(monkeypatch) -> None:
    from pydantic_ai import models

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    built: list[ModelConfig] = []

    def build(config: ModelConfig):
        built.append(config)
        return function_model("OK")

    probe.try_model(
        {"api_type": "openai-chat", "base_url": "https://a.invalid/v1", "model": "m"},
        STORED,
        build=build,
    )
    assert built[0].api_key == "STORED-KEY"


def budget_catalog(
    *,
    models: list[dict[str, object]],
    default_model_id: str | None = None,
    api_type: str = "anthropic",
):
    return {
        "version": 2,
        "providers": [
            {
                "id": "p",
                "name": api_type,
                "api_type": api_type,
                "base_url": "https://mock.invalid",
                "api_key": "test-only-key",
                "enabled": True,
            }
        ],
        "models": models,
        "default_model_id": default_model_id,
        "default_thinking": "default",
        "agent_configs": {},
    }


def test_probe_resolves_saved_model_budget_and_unsaved_model_budget() -> None:
    saved = budget_catalog(
        models=[
            {
                "id": "m",
                "provider_id": "p",
                "name": "Model",
                "model": "custom-alias",
                "enabled": True,
                "thinking_budget_tokens": 12000,
            }
        ],
        default_model_id="m",
    )
    configured = probe.resolve(
        {"model_id": "m", "thinking": "budget"}, saved, model_required=True
    )
    assert configured.model == "custom-alias"
    assert configured.thinking_budget_tokens == 12000

    unsaved = budget_catalog(models=[])
    configured = probe.resolve(
        {
            "provider_id": "p",
            "model": "new-alias",
            "thinking": "budget",
            "thinking_budget_tokens": 9000,
        },
        unsaved,
        model_required=True,
    )
    assert configured.model == "new-alias"
    assert configured.thinking_budget_tokens == 9000


@pytest.mark.parametrize("thinking", [None, [], {}, 1, "unknown"])
def test_probe_rejects_invalid_thinking_values_and_types(thinking) -> None:
    with pytest.raises(DomainError):
        probe.resolve(
            {
                "api_type": "openai-chat",
                "base_url": "https://mock.invalid",
                "api_key": "test-only-key",
                "model": "custom-alias",
                "thinking": thinking,
            },
            None,
            model_required=True,
        )


def test_probe_requires_budget_and_rejects_invalid_budget_numbers() -> None:
    base = {
        "api_type": "anthropic",
        "base_url": "https://mock.invalid",
        "api_key": "test-only-key",
        "model": "custom-alias",
        "thinking": "budget",
    }
    with pytest.raises(DomainError, match="positive integer"):
        probe.resolve(base, None, model_required=True)
    for value in (True, 0, -1, 1.5, "1024"):
        with pytest.raises(DomainError):
            probe.resolve(
                {**base, "thinking_budget_tokens": value}, None, model_required=True
            )


def test_a_model_test_requires_a_model_name() -> None:
    with pytest.raises(ValueError):
        probe.try_model(
            {
                "api_type": "openai-chat",
                "base_url": "https://a.invalid",
                "api_key": "k",
            },
            None,
            build=lambda config: function_model("OK"),
        )


def test_a_failing_model_test_never_echoes_the_key(monkeypatch) -> None:
    from pydantic_ai import models

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    with pytest.raises(DomainError) as failed:
        probe.try_model(
            {
                "api_type": "anthropic",
                "base_url": "https://a.invalid",
                "api_key": "TOP",
                "model": "m",
            },
            None,
            build=lambda config: function_model(
                RuntimeError(f"rejected key TOP for {config.model}")
            ),
        )
    assert failed.value.code == "model_test_failed"
    assert "TOP" not in str(failed.value)
    assert "rejected key *** for m" in str(failed.value)


def test_test_http_error_evidence_through_probe_entry(tmp_path) -> None:
    import json

    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    marker = "test evidence refusal marker"
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
            settings=thinking_settings(
                config.api_type,
                config.model,
                config.thinking,
                config.thinking_budget_tokens,
            ),
            provider=OpenAIProvider(openai_client=client),
        )

    with pytest.raises(DomainError) as failed:
        probe.try_model(
            {
                "api_type": "openai-chat",
                "base_url": "https://mock.invalid/v1",
                "api_key": "test-only-key",
                "model": "gpt-5.2",
                "thinking": "high",
            },
            None,
            build=build,
        )
    assert failed.value.code == "model_test_failed"
    assert marker in str(failed.value)
    assert "test-only-key" not in str(failed.value)
    assert len(requests) == 1
    assert requests[0]["reasoning_effort"] == "high"
    token_limit = requests[0].get(
        "max_tokens", requests[0].get("max_completion_tokens")
    )
    assert token_limit == probe.TEST_MAX_TOKENS
    evidence = {
        "request": requests[0],
        "error": str(failed.value),
        "marker_preserved": marker in str(failed.value),
        "api_key_exposed": "test-only-key" in str(failed.value),
    }
    output = tmp_path / "test-error.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


def test_anthropic_budget_test_wire_evidence(tmp_path, monkeypatch) -> None:
    import json

    import httpx2
    from anthropic import AsyncAnthropic
    from pydantic_ai import models
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    marker = "anthropic budget test refusal marker"
    requests: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(
            400, json={"error": {"message": marker, "type": "invalid_request_error"}}
        )

    def build(config: ModelConfig):
        client = AsyncAnthropic(
            api_key=config.api_key,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )
        return AnthropicModel(
            config.model,
            settings=thinking_settings(
                config.api_type,
                config.model,
                config.thinking,
                config.thinking_budget_tokens,
            ),
            provider=AnthropicProvider(anthropic_client=client),
        )

    with pytest.raises(DomainError) as failed:
        probe.try_model(
            {"model_id": "m", "thinking": "budget"},
            budget_catalog(
                models=[
                    {
                        "id": "m",
                        "provider_id": "p",
                        "name": "Model",
                        "model": "claude-sonnet-4-5",
                        "thinking_budget_tokens": 1024,
                    }
                ],
                default_model_id="m",
            ),
            build=build,
        )
    assert failed.value.code == "model_test_failed"
    assert marker in str(failed.value)
    assert "test-only-key" not in str(failed.value)
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert requests[0]["max_tokens"] == 9216
    evidence = {
        "request": requests[0],
        "error": str(failed.value),
        "marker_preserved": marker in str(failed.value),
        "api_key_exposed": "test-only-key" in str(failed.value),
    }
    output = tmp_path / "anthropic-budget-test.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


def test_google_budget_test_uses_explicit_number_for_unsaved_model(
    tmp_path, monkeypatch
) -> None:
    import json

    import httpx2
    from pydantic_ai import models
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    marker = "google budget test refusal marker"
    requests: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            400,
            json={"error": {"message": marker, "status": "INVALID_ARGUMENT"}},
        )

    def build(config: ModelConfig):
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return GoogleModel(
            config.model,
            settings=thinking_settings(
                config.api_type,
                config.model,
                config.thinking,
                config.thinking_budget_tokens,
            ),
            provider=GoogleProvider(
                api_key=config.api_key, base_url=config.base_url, http_client=http
            ),
        )

    with pytest.raises(DomainError) as failed:
        probe.try_model(
            {
                "provider_id": "p",
                "model": "gemini-3-alias",
                "thinking": "budget",
                "thinking_budget_tokens": 12000,
            },
            budget_catalog(models=[], api_type="google"),
            build=build,
        )
    assert failed.value.code == "model_test_failed"
    assert marker in str(failed.value)
    assert "test-only-key" not in str(failed.value)
    assert len(requests) == 1
    generation = requests[0]["generationConfig"]
    assert generation["thinkingConfig"] == {"thinking_budget": 12000}
    assert generation["maxOutputTokens"] == 20192
    evidence = {
        "request": requests[0],
        "error": str(failed.value),
        "marker_preserved": marker in str(failed.value),
        "api_key_exposed": "test-only-key" in str(failed.value),
    }
    output = tmp_path / "google-budget-test.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


def test_a_broken_provider_is_reported_as_a_failed_test() -> None:
    def build(config: ModelConfig):
        raise ValueError("bad base_url")

    with pytest.raises(DomainError) as failed:
        probe.try_model(
            {
                "api_type": "openai-chat",
                "base_url": "https://a.invalid",
                "api_key": "k",
                "model": "m",
            },
            None,
            build=build,
        )
    assert failed.value.code == "model_test_failed"
    assert str(failed.value) == "ValueError: bad base_url"
