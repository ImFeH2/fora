import json

import pytest
from test_runtime import HUMAN, MAIN, RecordingRunner, mention

pytest_plugins = ("test_runtime",)

from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.core.errors import DomainError
from fora.runtime.reminder import TurnOutcome
from fora.runtime.scheduler import Scheduler
from fora.tools.authorize import Actor


def outcome(calls=0, error=None):
    return TurnOutcome(
        messages_json="[]", usage_json=json.dumps({"tool_calls": calls}), error=error
    )


def test_no_tool_turns_block_and_only_human_can_resume(world):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(lambda request, tools: outcome()))
    for turn in range(1, 4):
        if turn > 1:
            world.store.append_message(room, HUMAN, "@Main continue")
        assert scheduler.run_turn(MAIN).status == "completed"
        assert world.history.no_tool_streak(MAIN) == turn
    assert world.store.get_member(MAIN).state == "blocked"
    assert world.history.pause_reason(MAIN) == "no_tool_calls"
    assert SqliteAgentStore(world.store._db).pause_reason(MAIN) == "no_tool_calls"
    restarted = Scheduler(world, scheduler._runner)
    world.store.append_message(room, HUMAN, "@Main new input")
    assert restarted.runnable_agents() == ()
    assert restarted.run_turn(MAIN) is None
    with pytest.raises(DomainError, match="Human"):
        restarted.tools_for(MAIN).resume_agent(MAIN)
    assert world.history.pause_reason(MAIN) == "no_tool_calls"
    human = restarted.tools_for_actor(Actor(HUMAN, False))
    human.resume_agent(MAIN)
    assert world.history.pause_reason(MAIN) is None
    assert world.history.no_tool_streak(MAIN) == 0
    assert restarted.run_turn(MAIN).status == "completed"
    assert world.history.no_tool_streak(MAIN) == 1


def test_human_resume_allows_the_same_mentions(world):
    mention(world)
    world.settings.set_settings("agent", {"no_tool_turns_before_pause": 1})
    scheduler = Scheduler(world, RecordingRunner(lambda request, tools: outcome()))
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.store.get_member(MAIN).state == "blocked"
    scheduler.tools_for_actor(Actor(HUMAN, False)).resume_agent(MAIN)
    assert scheduler.runnable_agents() == (MAIN,)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.store.get_member(MAIN).state == "blocked"


def test_soft_failures_do_not_count_and_tools_reset_the_streak(world):
    room = mention(world)
    responses = iter(
        [
            outcome(),
            outcome(error="network unavailable"),
            outcome(1, "network unavailable"),
            outcome(),
            outcome(1),
            outcome(),
        ]
    )
    scheduler = Scheduler(
        world, RecordingRunner(lambda request, tools: next(responses))
    )
    for expected in [1, 1, 0, 1, 0, 1]:
        world.store.append_message(room, HUMAN, "@Main retry")
        result = scheduler.run_turn(MAIN)
        assert result is not None
        assert world.history.no_tool_streak(MAIN) == expected
        assert world.history.pause_reason(MAIN) is None
        assert world.store.get_member(MAIN).state == (
            "error" if result.status == "failed" else "idle"
        )
        assert scheduler.runnable_agents() == (
            () if result.status == "failed" else (MAIN,)
        )


def test_runtime_exception_records_error(world):
    mention(world)

    def fail(request, tools):
        raise RuntimeError("framework failure")

    scheduler = Scheduler(world, RecordingRunner(fail))
    assert scheduler.run_turn(MAIN).status == "failed"
    assert world.store.get_member(MAIN).state == "error"
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.agent_status(MAIN)["error"] == "RuntimeError: framework failure"


@pytest.mark.parametrize("invalid", [False, True])
def test_actual_tool_calls_include_retry_results_and_do_not_leak_between_turns(
    world, monkeypatch, invalid
):
    from pydantic_ai import models
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from fora.adapters.model.runner import PydanticModelRunner

    room = mention(world)
    world.settings.set_settings("agent", {"no_tool_turns_before_pause": 1})
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discussion",
                        {"action": "read", "discussion_id": 999 if invalid else room},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("Done")])

    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.history.no_tool_streak(MAIN) == 0
    assert json.loads(world.history.runs(MAIN)[0].usage_json)["tool_calls"] == 1
    assert world.store.get_member(MAIN).state == "idle"
    world.store.append_message(room, HUMAN, "@Main another input")
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.history.no_tool_streak(MAIN) == 1
    assert world.store.get_member(MAIN).state == "blocked"


def test_resume_finalizes_an_interrupted_turn(world):
    mention(world)
    world.history.start_run(MAIN, reminded=[(1, 1)])
    world.store.set_agent_state(MAIN, "paused")
    scheduler = Scheduler(world, RecordingRunner())
    scheduler.tools_for_actor(Actor(HUMAN, False)).resume_agent(MAIN)
    assert world.store.get_member(MAIN).state == "idle"
    assert world.history.runs(MAIN)[0].status == "interrupted"
