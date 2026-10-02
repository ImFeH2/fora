from collections.abc import Mapping
from dataclasses import dataclass, fields

from fora.core.errors import DomainError


@dataclass(frozen=True)
class AgentParameters:
    context_window_tokens: int = 200_000
    exchange_nudge_after: int = 6
    max_concurrent_turns: int = 4
    idle_streak_after: int = 3
    no_tool_turns_before_pause: int = 3
    memory_index_bytes: int = 16_384
    token_limit: int = 0
    request_limit: int = 0


def agent_parameters(values: Mapping[str, object] | None) -> AgentParameters:
    return AgentParameters(**validate_parameters({} if values is None else values))


def validate_parameters(values: Mapping[str, object]) -> dict[str, int]:
    names = {field.name for field in fields(AgentParameters)}
    validated: dict[str, int] = {}
    for key, value in values.items():
        if key not in names:
            raise DomainError("invalid_setting", f"Unknown Agent parameter: {key}")
        minimum = 0 if key in ("token_limit", "request_limit") else 1
        if type(value) is not int or value < minimum:
            kind = "non-negative" if minimum == 0 else "positive"
            raise DomainError(
                "invalid_parameter", f"Agent parameter {key} must be a {kind} integer"
            )
        validated[key] = value
    return validated
