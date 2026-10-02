from dataclasses import asdict, fields

import pytest

from fora.core.errors import DomainError
from fora.core.parameters import (
    AgentParameters,
    agent_parameters,
    validate_parameters,
)


@pytest.mark.parametrize("values", [None, {}])
def test_parameters_fill_defaults(values) -> None:
    assert asdict(agent_parameters(values)) == {
        "context_window_tokens": 200_000,
        "exchange_nudge_after": 6,
        "max_concurrent_turns": 4,
        "idle_streak_after": 3,
        "no_tool_turns_before_pause": 3,
        "memory_index_bytes": 16_384,
        "token_limit": 0,
        "request_limit": 0,
    }


def test_parameters_accept_positive_integers() -> None:
    values = {field.name: 100 for field in fields(AgentParameters)}
    assert validate_parameters(values) == values
    assert asdict(agent_parameters(values)) == values
    assert validate_parameters({}) == {}


@pytest.mark.parametrize("key", [field.name for field in fields(AgentParameters)])
@pytest.mark.parametrize("value", [-1, True, False, "100", 1.5, None, [], {}])
@pytest.mark.parametrize("validate", [agent_parameters, validate_parameters])
def test_invalid_parameters_are_rejected_on_read_and_write(
    key, value, validate
) -> None:
    with pytest.raises(DomainError) as error:
        validate({key: value})
    assert error.value.code == "invalid_parameter"
    kind = "non-negative" if key in ("token_limit", "request_limit") else "positive"
    assert str(error.value) == f"Agent parameter {key} must be a {kind} integer"


@pytest.mark.parametrize("key", [field.name for field in fields(AgentParameters)])
def test_limits_accept_zero(key) -> None:
    values = {key: 0}
    if key in ("token_limit", "request_limit"):
        assert agent_parameters(values) == AgentParameters()
        assert validate_parameters(values) == values
    else:
        for validate in (agent_parameters, validate_parameters):
            with pytest.raises(DomainError) as error:
                validate(values)
            assert error.value.code == "invalid_parameter"


@pytest.mark.parametrize("key", [field.name for field in fields(AgentParameters)])
def test_missing_fields_use_defaults(key) -> None:
    values = {field.name: 100 for field in fields(AgentParameters) if field.name != key}
    expected = {**values, key: getattr(AgentParameters(), key)}
    assert asdict(agent_parameters(values)) == expected


@pytest.mark.parametrize("validate", [agent_parameters, validate_parameters])
def test_unknown_parameters_are_rejected_on_read_and_write(validate) -> None:
    with pytest.raises(DomainError) as error:
        validate({"unknown": 100})
    assert error.value.code == "invalid_setting"
    assert str(error.value) == "Unknown Agent parameter: unknown"
