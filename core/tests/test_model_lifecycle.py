from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic_ai.models import override_allow_model_requests

from fora.adapters.model.config import API_TYPES
from fora.adapters.model.runner import PydanticModelRunner, build_model
from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import SqliteStore
from fora.runtime.reminder import TurnRequest


@pytest.mark.parametrize("api_type", API_TYPES)
@pytest.mark.parametrize("concurrent", [False, True])
def test_turn_clients_are_isolated_and_closed_on_failure(
    tmp_path: Path, api_type: str, concurrent: bool
) -> None:
    base = SqliteStore(tmp_path / "fora.sqlite3")
    settings = SqliteAgentStore(base._db)
    settings.set_settings(
        "model",
        {
            "api_type": api_type,
            "base_url": "https://example.invalid/v1",
            "api_key": "unused",
            "model": "test-model",
        },
    )
    clients = []
    loops = []

    def build(config):
        model = build_model(config)
        clients.append(model.provider._own_http_client)
        loops.append(asyncio.get_running_loop())
        return model

    runner = PydanticModelRunner(settings, build_model=build)

    def run(sequence):
        return runner.run(
            TurnRequest(
                agent_id=sequence,
                sequence=sequence,
                agent_name="Lifecycle",
                prompt="Reply OK.",
                reminder=None,
                history_json="[]",
                resident="",
                environment=lambda: "",
                ephemeral=lambda: "",
                persist=lambda messages: None,
            ),
            None,
        )

    try:
        with override_allow_model_requests(False):
            if concurrent:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    outcomes = list(executor.map(run, range(1, 5)))
            else:
                outcomes = [run(sequence) for sequence in range(1, 5)]
        assert all(
            outcome.error is not None
            and "Model requests are not allowed" in outcome.error
            for outcome in outcomes
        )
        assert len(clients) == len(outcomes)
        assert len({id(client) for client in clients}) == len(outcomes)
        assert len(set(loops)) == len(outcomes)
        assert all(loop.is_closed() for loop in loops)
        assert all(client is not None and client.is_closed for client in clients)
    finally:
        base.close()
