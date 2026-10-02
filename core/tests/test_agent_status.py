from __future__ import annotations

import asyncio
import json
import threading

import pytest
from pydantic_ai import ModelMessagesTypeAdapter, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from test_runtime import HELPER, HUMAN, MAIN, RecordingRunner, mention

pytest_plugins = ("test_runtime",)

from fora.adapters.model.runner import PydanticModelRunner
from fora.core.errors import DomainError
from fora.runtime.reminder import TurnOutcome
from fora.runtime.scheduler import Scheduler


@pytest.fixture
def model_settings(world, monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {
            "version": 1,
            "providers": [
                {
                    "id": "provider",
                    "name": "Test",
                    "api_type": "openai-chat",
                    "base_url": "https://test.invalid",
                    "api_key": "test-only",
                }
            ],
            "models": [
                {
                    "id": "model",
                    "provider_id": "provider",
                    "name": "Test",
                    "model": "gpt-5",
                }
            ],
            "default_model_id": "model",
            "default_thinking": "high",
        },
    )
    return world


def test_error_requires_new_identity_after_restart_and_partial_ack(world):
    room = mention(world)
    world.store.append_message(room, HUMAN, "@Main another")
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome("[]", error="request failed")
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "failed"
    assert scheduler.agent_status(MAIN)["state"] == "error"
    world.store.ack(room, [1], MAIN)
    assert scheduler.runnable_agents() == ()
    world.history.mark_session_start()
    restarted = Scheduler(world, runner)
    assert restarted.runnable_agents() == ()
    another = mention(world)
    assert another != room
    assert restarted.runnable_agents() == (MAIN,)
    assert restarted.run_turn(MAIN).status == "failed"
    assert restarted.runnable_agents() == ()


@pytest.mark.parametrize("preparation_failed", [False, True])
def test_error_preparation_consumption_and_restart_continuation(
    world, preparation_failed
):
    room = mention(world)
    world.settings.set_settings("agent", {"context_window_tokens": 100})

    def respond(request, tools):
        if request.sequence == 1:
            return TurnOutcome(
                "[]", usage_json='{"last_input_tokens":100}', error="initial failure"
            )
        assert request.reminder is None
        return TurnOutcome(
            "[]", error="preparation failure" if preparation_failed else None
        )

    scheduler = Scheduler(world, RecordingRunner(respond))
    assert scheduler.run_turn(MAIN).status == "failed"
    assert scheduler.run_turn(MAIN) is None
    world.store.append_message(room, HUMAN, "@Main new identity")
    assert scheduler.run_turn(MAIN).status == (
        "failed" if preparation_failed else "completed"
    )
    assert world.history.prepared_mentions(MAIN, 2) == {(room, 1), (room, 2)}
    assert world.history.new_mentions(MAIN, [(room, 1), (room, 2)]) == frozenset()
    restarted = Scheduler(world, RecordingRunner())
    restarted.recover()
    if preparation_failed:
        assert restarted.runnable_agents() == ()
        assert restarted.run_turn(MAIN) is None
        world.store.ack(room, [1], MAIN)
        assert restarted.run_turn(MAIN) is None
        world.store.append_message(room, HUMAN, "@Main new request")
        assert restarted.runnable_agents() == (MAIN,)
    else:
        assert restarted.runnable_agents() == (MAIN,)
        assert restarted.run_turn(MAIN).status == "completed"
        assert world.history.prepared_mentions(MAIN, 2) == frozenset()
        assert world.history.lifecycle(MAIN).prepared_sequence is None
        assert restarted.runnable_agents() == (MAIN,)


@pytest.mark.parametrize("change", ["archive", "remove", "ack"])
@pytest.mark.parametrize("stage", ["error", "preparation", "continuation"])
def test_start_rechecks_pending_eligibility(world, monkeypatch, change, stage):
    room = mention(world)
    if stage != "error":
        world.settings.set_settings("agent", {"context_window_tokens": 100})

    def respond(request, tools):
        if request.sequence == 1:
            return TurnOutcome(
                "[]", usage_json='{"last_input_tokens":100}', error="initial failure"
            )
        assert request.reminder is None
        return TurnOutcome("[]")

    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "failed"
    new_room = mention(world)
    if stage == "continuation":
        world.store.ack(room, [1], MAIN)
        assert scheduler.run_turn(MAIN).status == "completed"
        world.store.revoke_ack(room, [1], MAIN)
    before = world.history.runs(MAIN)
    lifecycle = world.history.lifecycle(MAIN)
    original = scheduler._eligible

    def change_after_check(agent_id):
        eligible = original(agent_id)
        assert eligible
        if change == "archive":
            world.store.set_archived(new_room, True)
        elif change == "remove":
            world.store.set_discussion_members(new_room, [HUMAN])
        else:
            world.store.ack(new_room, [1], MAIN)
        return eligible

    monkeypatch.setattr(scheduler, "_eligible", change_after_check)
    assert scheduler.run_turn(MAIN) is None
    assert len(runner.requests) == len(before)
    assert world.history.runs(MAIN) == before
    assert world.history.lifecycle(MAIN) == lifecycle
    assert scheduler.pending_keys(MAIN) == {(room, 1)}
    assert scheduler._reserved == set()
    assert scheduler._active == {}
    monkeypatch.setattr(scheduler, "_eligible", original)
    assert scheduler.runnable_agents() == ()


def test_pause_allows_current_model_and_tool_to_finish(model_settings):
    world = model_settings
    room = mention(world)
    second = world.store.create_discussion("second", [HUMAN, HELPER])
    world.store.append_message(second.id, HUMAN, "@Helper continue")
    world.settings.set_settings("agent", {"max_concurrent_turns": 1})
    entered = threading.Event()
    release = threading.Event()
    calls = []

    async def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert await asyncio.to_thread(release.wait, 10)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discussion",
                        {
                            "action": "send",
                            "discussion_id": room,
                            "body": "real effect",
                        },
                        "send-result",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("finished")])

    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    scheduler = Scheduler(world, runner)
    try:
        assert scheduler.tick() == (MAIN,)
        assert entered.wait(5)
        state = scheduler.pause(MAIN)
        assert state["state"] == "running"
        assert state["pause_requested"]
        assert scheduler.pause(MAIN) == state
        assert scheduler.tick() == ()
        world.store.append_message(room, HUMAN, "@Main next turn")
        release.set()
        scheduler._threads[MAIN].join(5)
        assert not scheduler._threads[MAIN].is_alive()
        assert len(calls) == 2
        assert world.store.messages(room)[-1].body == "real effect"
        assert scheduler.agent_status(MAIN)["state"] == "paused"
        assert world.history.runs(MAIN)[0].status == "completed"
        assert MAIN not in scheduler._reserved
        assert scheduler.tick() == (HELPER,)
        scheduler._threads[HELPER].join(5)
        assert scheduler.resume(MAIN)["state"] == "idle"
        assert MAIN in scheduler.runnable_agents()
        history = ModelMessagesTypeAdapter.validate_json(
            world.history.runs(MAIN)[0].messages_json
        )
        returns = [
            part
            for message in history
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_call_id == "send-result"
        ]
        assert len(returns) == 1
        assert returns[0].content["discussion_id"] == room
    finally:
        release.set()
        scheduler.stop()


def test_resume_during_running_turn_clears_intent_and_preserves_effects(world):
    room = mention(world)
    entered = threading.Event()
    release = threading.Event()

    def respond(request, tools):
        entered.set()
        assert release.wait(10)
        tools.send_message(room, "continued")
        return TurnOutcome("[]")

    scheduler = Scheduler(world, RecordingRunner(respond))
    try:
        assert scheduler.tick() == (MAIN,)
        assert entered.wait(5)
        assert scheduler.pause(MAIN)["pause_requested"]
        state = scheduler.resume(MAIN)
        assert state["state"] == "running"
        assert not state["pause_requested"]
        assert scheduler.resume(MAIN) == state
        assert MAIN in scheduler._reserved
        release.set()
        scheduler._threads[MAIN].join(5)
        assert scheduler.agent_status(MAIN)["state"] == "idle"
        assert world.store.messages(room)[-1].body == "continued"
    finally:
        release.set()
        scheduler.stop()


@pytest.mark.parametrize("stage", ["construct", "start"])
def test_thread_start_failure_releases_capacity_without_model_calls(
    world, monkeypatch, stage
):
    mention(world)
    calls = []
    runner = RecordingRunner(lambda request, tools: calls.append(request))
    scheduler = Scheduler(world, runner)

    def fail(*args, **kwargs):
        raise RuntimeError("thread unavailable")

    if stage == "construct":
        monkeypatch.setattr(threading, "Thread", fail)
    else:
        monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        scheduler.tick()
    assert calls == []
    assert scheduler._reserved == set()
    assert scheduler._active == {}
    assert world.history.runs(MAIN)[0].status == "failed"
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert scheduler.runnable_agents() == ()


def test_start_transaction_failure_keeps_mention_unconsumed(world):
    room = mention(world)
    world.store._db.execute(
        "CREATE TEMP TRIGGER reject_start BEFORE UPDATE OF state ON members WHEN NEW.state = 'running' BEGIN SELECT RAISE(ABORT, 'state rejected'); END"
    )
    scheduler = Scheduler(world, RecordingRunner())
    with pytest.raises(Exception, match="state rejected"):
        scheduler.run_turn(MAIN)
    assert world.history.runs(MAIN) == ()
    assert world.history.new_mentions(MAIN, [(room, 1)]) == {(room, 1)}
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert MAIN not in scheduler._reserved
    world.store._db.execute("DROP TRIGGER reject_start")
    assert scheduler.resume(MAIN)["state"] == "idle"


def test_failed_finalization_recovery_preserves_human_history(world):
    mention(world)
    original = json.dumps([{"kind": "response", "parts": []}])
    repaired = json.dumps(
        [
            {
                "kind": "response",
                "parts": [{"part_kind": "text", "content": "human repair"}],
            }
        ]
    )

    def respond(request, tools):
        request.persist(original)
        world.store._db.execute(
            "CREATE TEMP TRIGGER reject_finish BEFORE UPDATE OF completed_at ON agent_runs BEGIN SELECT RAISE(ABORT, 'finish rejected'); END"
        )
        return TurnOutcome(original)

    scheduler = Scheduler(world, RecordingRunner(respond))
    with pytest.raises(Exception, match="finish rejected"):
        scheduler.run_turn(MAIN)
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert MAIN not in scheduler._reserved
    world.history.save_progress(MAIN, 1, repaired)
    world.store._db.execute("DROP TRIGGER reject_finish")
    assert scheduler.resume(MAIN)["state"] == "idle"
    assert world.history.runs(MAIN)[0].messages_json == repaired


def test_resume_repairs_interrupted_run_after_history_correction(world):
    mention(world)
    run = world.history.start_run(MAIN)
    world.history.save_progress(MAIN, run.sequence, "invalid history")
    world.store.set_agent_state(MAIN, "running")
    scheduler = Scheduler(world, PydanticModelRunner(world.settings))
    scheduler.recover()
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert world.history.runs(MAIN)[0].status == "running"
    repaired = ModelMessagesTypeAdapter.dump_json(
        [
            ModelResponse(parts=[TextPart("human repair")]),
        ]
    ).decode()
    world.history.save_progress(MAIN, run.sequence, repaired)
    scheduler.resume(MAIN)
    recovered = world.history.runs(MAIN)[0]
    assert recovered.status == "interrupted"
    assert recovered.messages_json == repaired
    assert world.history.pause_reason(MAIN) is None


def test_resume_failure_rolls_back_boundary(world):
    mention(world)
    scheduler = Scheduler(
        world,
        RecordingRunner(lambda request, tools: TurnOutcome("[]", error="model failed")),
    )
    scheduler.run_turn(MAIN)
    scheduler.pause(MAIN)
    before = world.history.lifecycle(MAIN)
    world.store._db.execute(
        "CREATE TEMP TRIGGER reject_resume BEFORE UPDATE OF state ON members"
        " WHEN NEW.state = 'idle' BEGIN SELECT RAISE(ABORT, 'resume rejected'); END"
    )
    with pytest.raises(Exception, match="resume rejected"):
        scheduler.resume(MAIN)
    assert world.history.lifecycle(MAIN) == before
    assert (
        list(
            world.store._db.execute(
                "SELECT resumed_after FROM agent_safety WHERE agent_id = ?", (MAIN,)
            )
        )
        == []
    )
    assert scheduler.agent_status(MAIN)["pause_requested"]
    assert scheduler.runnable_agents() == ()


def test_pause_and_resume_storage_failures_preserve_intent(world):
    mention(world)
    entered = threading.Event()
    release = threading.Event()

    def respond(request, tools):
        entered.set()
        assert release.wait(10)
        return TurnOutcome("[]")

    scheduler = Scheduler(world, RecordingRunner(respond))
    try:
        assert scheduler.tick() == (MAIN,)
        assert entered.wait(5)
        world.store._db.execute(
            "CREATE TEMP TRIGGER reject_intent BEFORE UPDATE ON agent_lifecycle BEGIN SELECT RAISE(ABORT, 'intent rejected'); END"
        )
        with pytest.raises(DomainError, match="saving the pause intent failed"):
            scheduler.pause(MAIN)
        assert scheduler.agent_status(MAIN)["pause_requested"]
        assert scheduler.agent_status(MAIN)["state"] == "running"
        with pytest.raises(Exception, match="intent rejected"):
            scheduler.resume(MAIN)
        assert scheduler.agent_status(MAIN)["pause_requested"]
        assert MAIN in scheduler._reserved
        world.store._db.execute("DROP TRIGGER reject_intent")
        release.set()
        scheduler._threads[MAIN].join(5)
        assert scheduler.agent_status(MAIN)["state"] == "paused"
        assert world.history.lifecycle(MAIN).pause_requested
        assert MAIN not in scheduler._reserved
    finally:
        release.set()
        scheduler.stop()
