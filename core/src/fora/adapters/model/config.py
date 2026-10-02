from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fora.core.errors import DomainError

ApiType = Literal["openai-chat", "openai-responses", "anthropic", "google"]

API_TYPES: tuple[ApiType, ...] = get_args(ApiType)
Thinking = Literal[
    "default", "none", "minimal", "low", "medium", "high", "xhigh", "max", "budget"
]
THINKING_OPTIONS: tuple[Thinking, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
BUDGET_API_TYPES = ("anthropic", "google")


def thinking_options(
    api_type: ApiType, model: str, thinking_budget_tokens: int | None = None
) -> list[Thinking]:
    options = list(THINKING_OPTIONS)
    if api_type in BUDGET_API_TYPES and thinking_budget_tokens is not None:
        options.append("budget")
    return options


def validate_thinking(value: Any, model: str) -> Thinking:
    if type(value) is not str or value not in (*THINKING_OPTIONS, "budget"):
        raise DomainError(
            "unsupported_thinking", f"Invalid thinking option for model {model}"
        )
    return cast(Thinking, value)


def validate_thinking_budget(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise DomainError(
            "invalid_thinking_budget",
            "Thinking token budget must be a positive integer",
        )
    return value


def thinking_settings(
    api_type: ApiType,
    model: str,
    thinking: Thinking,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    thinking = validate_thinking(thinking, model)
    if thinking == "default":
        return {}
    if thinking == "budget":
        if api_type not in BUDGET_API_TYPES:
            raise DomainError(
                "unsupported_thinking", f"Model {model} does not support token budgets"
            )
        tokens = validate_thinking_budget(thinking_budget_tokens)
        if api_type == "anthropic":
            return {
                "anthropic_thinking": {"type": "enabled", "budget_tokens": tokens},
                "max_tokens": tokens + 8192,
            }
        return {
            "google_thinking_config": {"thinking_budget": tokens},
            "max_tokens": tokens + 8192,
        }
    if api_type in ("openai-chat", "openai-responses"):
        return {"openai_reasoning_effort": thinking}
    if api_type == "anthropic":
        if thinking == "none":
            return {"anthropic_thinking": {"type": "disabled"}}
        return {
            "anthropic_thinking": {"type": "adaptive"},
            "anthropic_effort": thinking,
        }
    if thinking == "none":
        return {
            "google_thinking_config": {"thinking_budget": 0},
            "max_tokens": 8192,
        }
    return {"google_thinking_config": {"thinking_level": thinking.upper()}}


class ConfigurationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class ProviderConfig(ConfigurationRecord):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    name: str = Field(min_length=1)
    api_type: ApiType
    base_url: str = ""
    api_key: str = Field(default="", repr=False)
    enabled: bool = True


class RegisteredModel(ConfigurationRecord):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    enabled: bool = True
    thinking_budget_tokens: int | None = Field(default=None, gt=0, strict=True)


LegacyThinking = Literal[
    "default", "none", "minimal", "low", "medium", "high", "xhigh", "max"
]


class LegacyRegisteredModel(ConfigurationRecord):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    enabled: bool = True


class LegacyAgentModelConfig(ConfigurationRecord):
    model_id: str | None = None
    thinking: LegacyThinking | None = None


class LegacyModelCatalog(ConfigurationRecord):
    version: Literal[1] = 1
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[LegacyRegisteredModel] = Field(default_factory=list)
    default_model_id: str | None = None
    default_thinking: LegacyThinking = "default"
    agent_configs: dict[str, LegacyAgentModelConfig] = Field(default_factory=dict)


class AgentModelConfig(ConfigurationRecord):
    model_id: str | None = None
    thinking: Thinking | None = None


class ModelCatalog(ConfigurationRecord):
    version: Literal[2] = 2
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[RegisteredModel] = Field(default_factory=list)
    default_model_id: str | None = None
    default_thinking: Thinking = "default"
    agent_configs: dict[str, AgentModelConfig] = Field(default_factory=dict)

    @classmethod
    def restore(cls, values: dict[str, Any] | None) -> ModelCatalog:
        if not values:
            return cls()
        if "version" in values:
            try:
                if values.get("version") == 1:
                    legacy = LegacyModelCatalog.model_validate(values)
                    catalog = cls(
                        providers=legacy.providers,
                        models=[
                            RegisteredModel.model_validate(model.model_dump())
                            for model in legacy.models
                        ],
                        default_model_id=legacy.default_model_id,
                        default_thinking=legacy.default_thinking,
                        agent_configs={
                            agent_id: AgentModelConfig.model_validate(
                                config.model_dump()
                            )
                            for agent_id, config in legacy.agent_configs.items()
                        },
                    )
                else:
                    catalog = cls.model_validate(values)
            except ValidationError:
                raise DomainError(
                    "invalid_model_config", "Invalid model configuration"
                ) from None
            catalog.validate_references()
            return catalog
        provider = ProviderConfig(
            name="Default provider",
            api_type=values.get("api_type", "openai-chat"),
            base_url=values.get("base_url", ""),
            api_key=values.get("api_key", ""),
        )
        model_name = values.get("model")
        if not model_name:
            return cls(providers=[provider])
        model = RegisteredModel(
            provider_id=provider.id, name=model_name, model=model_name
        )
        return cls(providers=[provider], models=[model], default_model_id=model.id)

    def provider(self, provider_id: str) -> ProviderConfig:
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        raise DomainError("provider_not_found", "Provider no longer exists")

    def registered_model(self, model_id: str) -> RegisteredModel:
        for model in self.models:
            if model.id == model_id:
                return model
        raise DomainError("model_not_found", "Model no longer exists")

    def selection(self, agent_id: int | None = None) -> tuple[str | None, Thinking]:
        selection = self.agent_configs.get(str(agent_id), AgentModelConfig())
        return (
            selection.model_id
            if selection.model_id is not None
            else self.default_model_id,
            selection.thinking
            if selection.thinking is not None
            else self.default_thinking,
        )

    def validate_selection(self, selection: AgentModelConfig, owner: str) -> None:
        model_id = (
            selection.model_id
            if selection.model_id is not None
            else self.default_model_id
        )
        thinking = (
            selection.thinking
            if selection.thinking is not None
            else self.default_thinking
        )
        if model_id is None:
            if thinking == "budget":
                raise DomainError(
                    "unsupported_thinking",
                    f"{owner}: select a model with a token budget",
                )
            return
        model = self.registered_model(model_id)
        provider = self.provider(model.provider_id)
        if not model.enabled or not provider.enabled:
            raise DomainError(
                "model_in_use", f"{owner} references a disabled model or provider"
            )
        options = thinking_options(
            provider.api_type, model.model, model.thinking_budget_tokens
        )
        if thinking not in options:
            raise DomainError(
                "unsupported_thinking",
                f"{owner}: model {model.name} needs a configured token budget for budget selection",
            )

    def validate_references(self) -> None:
        if len({item.id for item in self.providers}) != len(self.providers):
            raise DomainError("invalid_model_config", "Provider IDs must be unique")
        if len({item.id for item in self.models}) != len(self.models):
            raise DomainError("invalid_model_config", "Model IDs must be unique")
        for model in self.models:
            provider = self.provider(model.provider_id)
            if (
                model.thinking_budget_tokens is not None
                and provider.api_type not in BUDGET_API_TYPES
            ):
                raise DomainError(
                    "invalid_model_config",
                    f"Model {model.name} budgets require Anthropic or Google",
                )
        self.validate_selection(AgentModelConfig(), "Global default")
        for agent_id, selection in self.agent_configs.items():
            if not agent_id.isdecimal() or int(agent_id) <= 0:
                raise DomainError("invalid_model_config", "Invalid Agent ID")
            self.validate_selection(selection, f"Agent {agent_id}")

    def apply(self, change: dict[str, Any], agent_ids: set[int]) -> ModelCatalog:
        updated = self.model_copy(deep=True)
        action = change.get("action")
        values = dict(change.get("values", {}))
        record_id = change.get("id")
        try:
            if action == "save_provider":
                if record_id is not None:
                    provider = updated.provider(record_id)
                    merged = provider.model_dump()
                    if values.get("api_key") == "":
                        values.pop("api_key")
                    if values.pop("clear_key", False):
                        values["api_key"] = ""
                    merged.update(values)
                    merged["id"] = record_id
                    updated.providers[updated.providers.index(provider)] = (
                        ProviderConfig.model_validate(merged)
                    )
                else:
                    updated.providers.append(ProviderConfig.model_validate(values))
            elif action == "delete_provider":
                if not isinstance(record_id, str):
                    raise DomainError("invalid_model_config", "Provider ID is required")
                provider = updated.provider(record_id)
                if any(model.provider_id == record_id for model in updated.models):
                    raise DomainError(
                        "provider_in_use",
                        "Remove this provider's models before deleting the provider",
                    )
                updated.providers.remove(provider)
            elif action == "save_model":
                if record_id is not None:
                    model = updated.registered_model(record_id)
                    merged = {**model.model_dump(), **values, "id": record_id}
                    updated.models[updated.models.index(model)] = (
                        RegisteredModel.model_validate(merged)
                    )
                else:
                    updated.models.append(RegisteredModel.model_validate(values))
            elif action == "delete_model":
                if not isinstance(record_id, str):
                    raise DomainError("invalid_model_config", "Model ID is required")
                model = updated.registered_model(record_id)
                owners = []
                if updated.default_model_id == record_id:
                    owners.append("Global default")
                owners.extend(
                    f"Agent {agent_id}"
                    for agent_id, config in updated.agent_configs.items()
                    if config.model_id == record_id and int(agent_id) in agent_ids
                )
                if owners:
                    raise DomainError(
                        "model_in_use", "Model is used by " + ", ".join(owners)
                    )
                updated.models.remove(model)
            elif action == "set_defaults":
                defaults = AgentModelConfig.model_validate(values)
                if defaults.thinking is None:
                    raise DomainError(
                        "invalid_model_config",
                        "Global thinking must have an explicit value",
                    )
                updated.default_model_id = defaults.model_id
                updated.default_thinking = defaults.thinking
            elif action == "set_agent":
                if type(record_id) is not int or record_id not in agent_ids:
                    raise DomainError("not_found", "Agent does not exist")
                updated.agent_configs[str(record_id)] = AgentModelConfig.model_validate(
                    values
                )
            else:
                raise DomainError(
                    "invalid_model_action", "Unknown model configuration action"
                )
        except ValidationError:
            raise DomainError(
                "invalid_model_config", "Invalid model configuration fields"
            ) from None
        updated.agent_configs = {
            key: value
            for key, value in updated.agent_configs.items()
            if int(key) in agent_ids
        }
        updated.validate_references()
        updated.validate_selection(AgentModelConfig(), "Global default")
        for agent_id in sorted(agent_ids):
            updated.validate_selection(
                updated.agent_configs.get(str(agent_id), AgentModelConfig()),
                f"Agent {agent_id}",
            )
        return updated

    def redacted(self) -> dict[str, Any]:
        values = self.model_dump(exclude={"providers": {"__all__": {"api_key"}}})
        for model, public in zip(self.models, values["models"], strict=True):
            public["thinking_options"] = thinking_options(
                self.provider(model.provider_id).api_type,
                model.model,
                model.thinking_budget_tokens,
            )
        for provider, public in zip(self.providers, values["providers"], strict=True):
            public["api_key_set"] = bool(provider.api_key)
        return values

    def resolve(self, agent_id: int | None = None) -> ModelConfig:
        model_id, thinking = self.selection(agent_id)
        if model_id is None:
            raise DomainError(
                "model_unavailable",
                "Configure a model in Settings before running Agents",
            )
        model = self.registered_model(model_id)
        provider = self.provider(model.provider_id)
        if not model.enabled or not provider.enabled:
            raise DomainError(
                "model_unavailable", "The selected model or provider is disabled"
            )
        if not provider.base_url.strip():
            raise DomainError(
                "model_unavailable", f"Provider {provider.name} needs a base URL"
            )
        if not provider.api_key:
            raise DomainError(
                "model_unavailable", f"Provider {provider.name} needs an API key"
            )
        return ModelConfig(
            api_type=provider.api_type,
            base_url=provider.base_url,
            api_key=provider.api_key,
            model=model.model,
            thinking=thinking,
            thinking_budget_tokens=model.thinking_budget_tokens,
        )


@dataclass(frozen=True)
class ModelConfig:
    api_type: ApiType
    base_url: str
    api_key: str = field(repr=False)
    model: str
    thinking: Thinking = "default"
    thinking_budget_tokens: int | None = None
