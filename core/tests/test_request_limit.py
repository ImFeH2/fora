from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits
from test_sidecar_process import drive, response

from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import SqliteStore
from fora.core.parameters import AgentParameters, agent_parameters


@pytest.mark.parametrize("request_limit", [0, 1, 50, 75])
def test_request_limit_survives_settings_api_and_restart(
    tmp_path: Path, request_limit: int
) -> None:
    data = tmp_path / "data"
    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        SqliteAgentStore(store._db).set_settings("agent", {"token_limit": 1000})
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "settings.get", "params": {"section": "agent"}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "agent",
                    "values": {"request_limit": request_limit},
                },
            },
            {"id": 3, "method": "settings.get", "params": {"section": "agent"}},
        ],
    )
    assert code == 0, stderr
    original = {**asdict(AgentParameters()), "token_limit": 1000}
    expected = {**original, "request_limit": request_limit}
    assert response(frames, 1)["result"] == original
    assert response(frames, 2)["result"] == expected
    assert response(frames, 3)["result"] == expected
    frames, code, stderr = drive(
        data,
        [{"id": 1, "method": "settings.get", "params": {"section": "agent"}}],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"] == expected


@pytest.mark.parametrize("value", [-1, 1.5, True, False, "50", None])
def test_invalid_request_limit_preserves_settings(
    tmp_path: Path, value: object
) -> None:
    data = tmp_path / "data"
    original = {"request_limit": 75, "token_limit": 1000}
    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        SqliteAgentStore(store._db).set_settings("agent", original)
    frames, code, stderr = drive(
        data,
        [
            {
                "id": 1,
                "method": "settings.update",
                "params": {"section": "agent", "values": {"request_limit": value}},
            },
            {"id": 2, "method": "settings.get", "params": {"section": "agent"}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["error"]["code"] == "invalid_parameter"
    assert response(frames, 2)["result"] == {**asdict(AgentParameters()), **original}
    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        assert SqliteAgentStore(store._db).get_settings("agent") == original


@pytest.mark.parametrize(
    "values", [None, {}, {"token_limit": 1000}, {"request_limit": 0}]
)
def test_unlimited_requests_pass_sdk_check(values) -> None:
    limits = UsageLimits(request_limit=agent_parameters(values).request_limit or None)
    for requests in (0, 49, 50, 51, 1000):
        limits.check_before_request(RunUsage(requests=requests))
    assert limits.request_limit is None


@pytest.mark.parametrize("request_limit", [1, 50, 75])
def test_sdk_blocks_next_request_at_configured_limit(request_limit: int) -> None:
    limits = UsageLimits(request_limit=request_limit)
    for requests in (0, request_limit - 1):
        limits.check_before_request(RunUsage(requests=requests))
    for requests in (request_limit, request_limit + 1):
        with pytest.raises(
            UsageLimitExceeded, match=f"request_limit of {request_limit}"
        ):
            limits.check_before_request(RunUsage(requests=requests))
